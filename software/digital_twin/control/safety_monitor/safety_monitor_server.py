"""Safety monitoring service for the 2D thaw-front digital twin."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.utils.logging_setup import get_logger

THAW_FRONT_QUEUE = "permafrost.record.thaw_front.2d"
FORECAST_QUEUE = "permafrost.record.fem_forecast.2d"
ALERT_QUEUE = "permafrost.record.safety_alerts.2d"

THAW_FRONT_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "thaw_front_message.json"
FORECAST_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "fem_forecast_message.json"
ALERT_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "safety_alert_message.json"

ALLOWED_METRICS = {"max", "avg"}
ALLOWED_MODES = {"single", "dual"}


class SafetyMonitorServer:
    """Evaluate thaw-front metrics against a safety limit and emit alerts.

    Two modes:

    * ``single`` (legacy) — compare one reduced metric (``max``/``avg``) against a
      single ``limit_radius_m`` and trigger when it is exceeded.
    * ``dual`` — the closed-loop thaw-control rule engine. It uses both the outer
      thaw radius ``r_max`` and the least-thawed radius ``r_min`` against an upper
      and lower limit:

        - ``r_max >= limit_max``           -> STOP heating (status ``alert``). This
          is the safety cut and overrides everything, including a low ``r_min``.
        - otherwise                        -> keep HEATING ("keep thawing"), whether
          ``r_min`` is above or below its floor.
        - ``r_max`` in [warn, critical)·limit_max -> status ``warning`` (still heating)
        - ``r_max`` in [critical, 1)·limit_max     -> status ``critical`` (still heating)
        - ``r_min < limit_min``            -> flagged ``rmin_below_floor`` (informational;
          reinforces that heating should continue — never stops the heater).

      The emitted alert carries a ``recommended_action`` (``stop``/``heat``) that the
      heater-actuation service maps directly onto the PolyScience power state.
    """

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        thaw_input_config: RabbitMQConfig | None = None,
        forecast_input_config: RabbitMQConfig | None = None,
        output_config: RabbitMQConfig | None = None,
        *,
        thaw_input_queue: str | None = None,
        forecast_input_queue: str | None = None,
        output_queue: str | None = None,
        limit_radius_m: float = 0.1,
        metric: str = "max",
        mode: str = "single",
        limit_min_radius_m: float = 0.0,
        warn_fraction: float = 0.6,
        critical_fraction: float = 0.9,
        publish_on_change: bool = False,
        site_id: str = "default",
    ) -> None:
        self.logger = get_logger("SafetyMonitorServer")
        self.influx_config = influx_config or InfluxConfig()
        self.thaw_input_config = resolve_queue_config(
            thaw_input_config,
            queue=thaw_input_queue or THAW_FRONT_QUEUE,
            schema_path=THAW_FRONT_SCHEMA,
        )
        self.forecast_input_config = resolve_queue_config(
            forecast_input_config,
            queue=forecast_input_queue or FORECAST_QUEUE,
            schema_path=FORECAST_SCHEMA,
        )
        self.output_config = resolve_queue_config(
            output_config,
            queue=output_queue or ALERT_QUEUE,
            schema_path=ALERT_SCHEMA,
        )
        mode = mode.lower().strip()
        if mode not in ALLOWED_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Allowed: {sorted(ALLOWED_MODES)}")
        metric = metric.lower().strip()
        if mode == "single" and metric not in ALLOWED_METRICS:
            raise ValueError(f"Unsupported metric '{metric}'. Allowed: {sorted(ALLOWED_METRICS)}")

        self.mode = mode
        self.metric = metric
        self.limit_radius_m = float(limit_radius_m)
        self.limit_min_radius_m = float(limit_min_radius_m)
        self.warn_fraction = float(warn_fraction)
        self.critical_fraction = float(critical_fraction)
        self.publish_on_change = bool(publish_on_change)
        self.site_id = site_id

        self._lock = threading.Lock()
        self._last_thaw: Optional[dict] = None
        self._last_forecast: Optional[dict] = None
        self._last_triggered: Optional[bool] = None
        self._last_change_key: Optional[tuple] = None
        self._threads: list[threading.Thread] = []

        self.mq_thaw: Optional[RabbitMQClient] = None
        self.mq_forecast: Optional[RabbitMQClient] = None
        self.mq_out: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None

    def _metric_field(self) -> str:
        return "radius_max_m" if self.metric == "max" else "radius_avg_m"

    def _parse_metric(self, msg: dict, *, source: str) -> dict:
        try:
            time_hours = float(msg["time_hours"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Missing time_hours in {source} message.")

        metric_field = self._metric_field()
        raw_value = msg.get(metric_field)
        radius_m = None if raw_value is None else float(raw_value)
        timestamp = msg.get("timestamp") or datetime.utcnow().isoformat()

        return {
            "time_hours": time_hours,
            "radius_m": radius_m,
            "timestamp": timestamp,
        }

    @staticmethod
    def _opt_float(value) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_dual(self, msg: dict, *, source: str) -> dict:
        try:
            time_hours = float(msg["time_hours"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Missing time_hours in {source} message.")

        return {
            "time_hours": time_hours,
            "r_max": self._opt_float(msg.get("radius_max_m")),
            "r_min": self._opt_float(msg.get("radius_min_m")),
            "timestamp": msg.get("timestamp") or datetime.utcnow().isoformat(),
        }

    def _evaluate_and_publish_dual(self, *, event_time_hours: float, event_timestamp: str) -> None:
        """Apply the dual-limit thaw-control rules and publish the decision.

        Safety-first priority: if the outer thaw radius (measured or forecast)
        reaches ``limit_max`` the heater is stopped regardless of ``r_min``.
        Otherwise heating continues, with graded warning statuses as ``r_max``
        approaches the limit and an informational flag when ``r_min`` sits below
        its floor.
        """
        if self.mq_out is None or self.influx is None:
            raise RuntimeError("Dependencies not initialised. Did you call setup()?")  # pragma: no cover

        with self._lock:
            thaw = self._last_thaw
            forecast = self._last_forecast

        m_rmax = thaw["r_max"] if thaw else None
        m_rmin = thaw["r_min"] if thaw else None
        f_rmax = forecast["r_max"] if forecast else None

        limit_max = self.limit_radius_m
        limit_min = self.limit_min_radius_m

        over_measured = m_rmax is not None and m_rmax >= limit_max
        over_forecast = f_rmax is not None and f_rmax >= limit_max
        triggered = bool(over_measured or over_forecast)

        ratio = (m_rmax / limit_max) if (m_rmax is not None and limit_max > 0) else None
        if triggered:
            status = "alert"
        elif ratio is not None and ratio >= self.critical_fraction:
            status = "critical"
        elif ratio is not None and ratio >= self.warn_fraction:
            status = "warning"
        else:
            status = "ok"

        rmin_below_floor = bool(m_rmin is not None and m_rmin < limit_min)
        recommended_action = "stop" if triggered else "heat"

        if over_measured:
            reason = "rmax_over_limit"
        elif over_forecast:
            reason = "forecast_rmax_over_limit"
        elif status == "critical":
            reason = "rmax_critical"
        elif status == "warning":
            reason = "rmax_warning"
        elif rmin_below_floor:
            reason = "rmin_below_floor"
        else:
            reason = "nominal"

        change_key = (status, recommended_action)
        if self.publish_on_change and self._last_change_key is not None and change_key == self._last_change_key:
            return
        self._last_change_key = change_key

        payload = {
            "timestamp": event_timestamp,
            "time_hours": event_time_hours,
            "limit_radius_m": limit_max,
            "measured_radius_m": m_rmax,
            "forecast_radius_m": f_rmax,
            "triggered": triggered,
            "status": status,
            "reason": reason,
            "metric": "dual",
            "r_max_m": m_rmax,
            "r_min_m": m_rmin,
            "limit_min_radius_m": limit_min,
            "warn_fraction": self.warn_fraction,
            "critical_fraction": self.critical_fraction,
            "rmin_below_floor": rmin_below_floor,
            "recommended_action": recommended_action,
        }

        self.mq_out.publish(payload)
        self.influx.write_safety_alert(
            time_hours=event_time_hours,
            limit_radius_m=limit_max,
            measured_radius_m=m_rmax,
            forecast_radius_m=f_rmax,
            triggered=triggered,
            reason=reason,
            metric="dual",
            status=status,
            r_max_m=m_rmax,
            r_min_m=m_rmin,
            limit_min_radius_m=limit_min,
            rmin_below_floor=rmin_below_floor,
            recommended_action=recommended_action,
            site=self.site_id,
        )
        self.logger.info(
            "Thaw-control status (t=%.2fh, r_max=%s, r_min=%s, limit_max=%.3fm, "
            "limit_min=%.3fm, status=%s, action=%s%s)",
            event_time_hours,
            "n/a" if m_rmax is None else f"{m_rmax:.3f}m",
            "n/a" if m_rmin is None else f"{m_rmin:.3f}m",
            limit_max,
            limit_min,
            status,
            recommended_action,
            ", rmin<floor" if rmin_below_floor else "",
        )

    def _evaluate_and_publish(self, *, event_time_hours: float, event_timestamp: str) -> None:
        if self.mq_out is None or self.influx is None:
            raise RuntimeError("Dependencies not initialised. Did you call setup()?")  # pragma: no cover

        with self._lock:
            thaw = self._last_thaw
            forecast = self._last_forecast

        measured_radius = thaw["radius_m"] if thaw else None
        forecast_radius = forecast["radius_m"] if forecast else None

        reasons: list[str] = []
        if measured_radius is not None and measured_radius >= self.limit_radius_m:
            reasons.append("measured")
        if forecast_radius is not None and forecast_radius >= self.limit_radius_m:
            reasons.append("forecast")

        triggered = bool(reasons)
        if self.publish_on_change and self._last_triggered is not None and triggered == self._last_triggered:
            return
        self._last_triggered = triggered

        if not reasons:
            reason = "none"
        elif len(reasons) == 2:
            reason = "both"
        else:
            reason = reasons[0]

        payload = {
            "timestamp": event_timestamp,
            "time_hours": event_time_hours,
            "limit_radius_m": self.limit_radius_m,
            "measured_radius_m": measured_radius,
            "forecast_radius_m": forecast_radius,
            "triggered": triggered,
            "status": "alert" if triggered else "ok",
            "reason": reason,
            "metric": self.metric,
        }

        self.mq_out.publish(payload)
        self.influx.write_safety_alert(
            time_hours=event_time_hours,
            limit_radius_m=self.limit_radius_m,
            measured_radius_m=measured_radius,
            forecast_radius_m=forecast_radius,
            triggered=triggered,
            reason=reason,
            metric=self.metric,
            site=self.site_id,
        )
        self.logger.info(
            "Safety status (t=%.2fh, measured=%s, forecast=%s, limit=%.3fm, status=%s)",
            event_time_hours,
            "n/a" if measured_radius is None else f"{measured_radius:.3f}m",
            "n/a" if forecast_radius is None else f"{forecast_radius:.3f}m",
            self.limit_radius_m,
            payload["status"],
        )

    def _handle_thaw(self, msg: dict) -> None:
        if self.mode == "dual":
            data = self._parse_dual(msg, source="thaw_front")
            with self._lock:
                self._last_thaw = data
            self._evaluate_and_publish_dual(
                event_time_hours=data["time_hours"],
                event_timestamp=data["timestamp"],
            )
            return
        data = self._parse_metric(msg, source="thaw_front")
        with self._lock:
            self._last_thaw = data
        self._evaluate_and_publish(
            event_time_hours=data["time_hours"],
            event_timestamp=data["timestamp"],
        )

    def _handle_forecast(self, msg: dict) -> None:
        if self.mode == "dual":
            data = self._parse_dual(msg, source="forecast")
            with self._lock:
                self._last_forecast = data
            self._evaluate_and_publish_dual(
                event_time_hours=data["time_hours"],
                event_timestamp=data["timestamp"],
            )
            return
        data = self._parse_metric(msg, source="forecast")
        with self._lock:
            self._last_forecast = data
        self._evaluate_and_publish(
            event_time_hours=data["time_hours"],
            event_timestamp=data["timestamp"],
        )

    def _consume_thaw(self) -> None:
        if self.mq_thaw is None:
            raise RuntimeError("Thaw input client not initialised.")  # pragma: no cover
        self.mq_thaw.consume(callback=self._handle_thaw)

    def _consume_forecast(self) -> None:
        if self.mq_forecast is None:
            raise RuntimeError("Forecast input client not initialised.")  # pragma: no cover
        self.mq_forecast.consume(callback=self._handle_forecast)

    def setup(self) -> None:
        if self.mq_thaw is None:
            self.mq_thaw = RabbitMQClient(self.thaw_input_config)
        if self.mq_forecast is None:
            self.mq_forecast = RabbitMQClient(self.forecast_input_config)
        if self.mq_out is None:
            self.mq_out = RabbitMQClient(self.output_config)
        if self.influx is None:
            self.influx = InfluxHelper(self.influx_config)
        self.logger.info(
            "Dependencies initialised (thaw=%s, forecast=%s, out=%s)",
            self.thaw_input_config.queue,
            self.forecast_input_config.queue,
            self.output_config.queue,
        )
        if self.mode == "dual":
            self.logger.info(
                "Dual-limit thaw control (limit_max=%.3fm, limit_min=%.3fm, "
                "warn=%.0f%%, critical=%.0f%%, publish_on_change=%s)",
                self.limit_radius_m,
                self.limit_min_radius_m,
                self.warn_fraction * 100.0,
                self.critical_fraction * 100.0,
                self.publish_on_change,
            )
        else:
            self.logger.info(
                "Safety limit configured (limit=%.3fm, metric=%s, publish_on_change=%s)",
                self.limit_radius_m,
                self.metric,
                self.publish_on_change,
            )

    def start(self) -> None:
        if self.mq_thaw is None or self.mq_forecast is None or self.mq_out is None or self.influx is None:
            self.setup()

        self.logger.info("Listening for thaw-front updates on %s", self.thaw_input_config.queue)
        self.logger.info("Listening for FEM forecasts on %s", self.forecast_input_config.queue)

        thaw_thread = threading.Thread(target=self._consume_thaw, name="safety_monitor_thaw", daemon=True)
        forecast_thread = threading.Thread(
            target=self._consume_forecast,
            name="safety_monitor_forecast",
            daemon=True,
        )
        self._threads = [thaw_thread, forecast_thread]
        for thread in self._threads:
            thread.start()

        try:
            while any(thread.is_alive() for thread in self._threads):
                for thread in self._threads:
                    thread.join(timeout=0.5)
        except KeyboardInterrupt:  # pragma: no cover - runtime behaviour
            self.logger.info("Interrupt received; shutting down")
        finally:
            self.close()

    def stop(self) -> None:
        if self.mq_thaw and self.mq_thaw.channel:
            self.mq_thaw.channel.stop_consuming()
        if self.mq_forecast and self.mq_forecast.channel:
            self.mq_forecast.channel.stop_consuming()

    def close(self) -> None:
        self.stop()
        if self.mq_thaw is not None:
            self.mq_thaw.disconnect()
        if self.mq_forecast is not None:
            self.mq_forecast.disconnect()
        if self.mq_out is not None:
            self.mq_out.disconnect()
        if self.influx is not None:
            self.influx.close()
        self.logger.info("Shutdown complete")


if __name__ == "__main__":
    SafetyMonitorServer().start()
