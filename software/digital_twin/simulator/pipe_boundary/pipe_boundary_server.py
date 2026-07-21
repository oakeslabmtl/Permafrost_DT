"""Pipe boundary enrichment service.

Sits between the boundary-condition builder and the FEM forecast. It calibrates
the U-shaped heat-exchanger loop model against the live inlet/outlet fluid
temperature stream (slow loop), and on every boundary message it overrides the
pipe boundary set with a physics-based Robin condition derived from the
calibrated fluid state (fast loop), then forwards the enriched message to the
FEM.

This replaces the crude "stamp one raw sensor onto the whole pipe wall"
boundary that made the FEM soil field inaccurate: the pipe boundary now carries
a calibrated, transport-delayed effective fluid temperature that respects the
finite fluid->wall heat transfer.

Two RabbitMQ consumers run concurrently, each on its own connection:

* fluid stream (background thread) -> maintain a rolling window, recalibrate the
  six pipe parameters periodically (warm-started), and cache the current
  effective interface temperature.
* boundary stream (main thread) -> enrich + forward to the FEM boundary queue.
"""

from __future__ import annotations

import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Optional

import numpy as np

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.tools.run_loop_thermal_case import (
    DEFAULT_P0,
    PARAM_NAMES,
    PipeModelConfig,
    calibrate_pipe,
    effective_interface_temperature,
    interface_temperature_profile,
)
from software.utils.logging_setup import get_logger

BOUNDARY_IN_QUEUE = "permafrost.record.boundary.raw.2d"
BOUNDARY_OUT_QUEUE = "permafrost.record.boundary.2d"
FLUID_QUEUE = "permafrost.record.fluid.pipe"
FLUID_EXCHANGE = "permafrost.fluid"
CALIBRATION_QUEUE = "permafrost.record.pipe_calibration"

_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "communication" / "schemas"
BOUNDARY_SCHEMA = _SCHEMA_DIR / "boundary_condition_message.json"
FLUID_SCHEMA = _SCHEMA_DIR / "fluid_message.json"
CALIBRATION_SCHEMA = _SCHEMA_DIR / "pipe_calibration_output_message.json"


class PipeBoundaryServer:
    """Calibrates the pipe model and enriches FEM boundary messages."""

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        boundary_in_config: RabbitMQConfig | None = None,
        boundary_out_config: RabbitMQConfig | None = None,
        fluid_config: RabbitMQConfig | None = None,
        calibration_config: RabbitMQConfig | None = None,
        *,
        boundary_in_queue: str | None = None,
        boundary_out_queue: str | None = None,
        fluid_queue: str | None = None,
        calibration_queue: str | None = None,
        site_id: str = "default",
        # Pipe geometry / discretisation.
        h_sample_m: float = 0.30,
        r_bend_m: float = 0.0,
        n_cells: int = 120,
        max_dt_s: float = 5.0,
        max_steps: int = 3000,
        # Boundary mapping.
        pipe_boundary_set: str = "left",
        pipe_bc_type: str = "robin",
        pipe_h_coeff: float = 5.0,
        interface_source: str = "fluid",
        interface_reduction: str = "mean",
        depth_resolved: bool = True,
        profile_points: int = 17,
        # Windowing / calibration cadence.
        window_seconds: float = 7200.0,
        min_points: int = 5,
        max_points: int = 240,
        recalibrate_seconds: float = 300.0,
        maxiter: int = 40,
        maxfun: int = 80,
        initial_params: dict[str, float] | None = None,
        lower_bounds: dict[str, float] | None = None,
        upper_bounds: dict[str, float] | None = None,
        fixed_params: dict[str, float] | None = None,
        warm_start: bool = True,
        passthrough_until_ready: bool = True,
        publish_calibration: bool = True,
    ) -> None:
        self.logger = get_logger("PipeBoundaryServer")
        self.influx_config = influx_config or InfluxConfig()

        self.boundary_in_config = resolve_queue_config(
            boundary_in_config,
            queue=boundary_in_queue or BOUNDARY_IN_QUEUE,
            schema_path=BOUNDARY_SCHEMA,
        )
        self.boundary_out_config = resolve_queue_config(
            boundary_out_config,
            queue=boundary_out_queue or BOUNDARY_OUT_QUEUE,
            schema_path=BOUNDARY_SCHEMA,
        )
        self.fluid_config = resolve_queue_config(
            fluid_config,
            queue=fluid_queue or FLUID_QUEUE,
            schema_path=FLUID_SCHEMA,
        )
        self.calibration_config = resolve_queue_config(
            calibration_config,
            queue=calibration_queue or CALIBRATION_QUEUE,
            schema_path=CALIBRATION_SCHEMA,
        )

        self.site_id = site_id
        self.pipe_config = PipeModelConfig(
            h_sample_m=float(h_sample_m),
            r_bend_m=float(r_bend_m),
            n_cells=max(int(n_cells), 4),
            max_dt_s=float(max_dt_s),
            max_steps=max(int(max_steps), 10),
        )

        self.pipe_boundary_set = str(pipe_boundary_set)
        self.pipe_bc_type = str(pipe_bc_type).strip().lower()
        if self.pipe_bc_type not in {"robin", "dirichlet"}:
            raise ValueError("pipe_bc_type must be 'robin' or 'dirichlet'.")
        self.pipe_h_coeff = float(pipe_h_coeff)
        self.interface_source = str(interface_source)
        self.interface_reduction = str(interface_reduction)
        self.depth_resolved = bool(depth_resolved)
        self.profile_points = max(int(profile_points), 2)
        # Depths (m) spanning the pipe axis, used to build the boundary profile.
        self._profile_depths = np.linspace(0.0, self.pipe_config.h_sample_m, self.profile_points)

        self.window_seconds = max(float(window_seconds), 0.0)
        self.min_points = max(int(min_points), 2)
        self.max_points = max(int(max_points), self.min_points)
        self.recalibrate_seconds = max(float(recalibrate_seconds), 0.0)
        self.maxiter = max(int(maxiter), 1)
        self.maxfun = max(int(maxfun), 1)
        self.initial_params = initial_params or None
        self.lower_bounds = lower_bounds or None
        self.upper_bounds = upper_bounds or None
        self.fixed_params = fixed_params or None
        self.use_warm_start = bool(warm_start)
        self.passthrough_until_ready = bool(passthrough_until_ready)
        self.publish_calibration = bool(publish_calibration)

        # Shared state (guarded by _lock).
        self._lock = Lock()
        self._window: list[tuple[float, float, float]] = []  # (time_hours, t_in, t_out)
        self._best_params: dict[str, float] | None = None
        self._effective_temp_c: float | None = None
        self._effective_profile: list[dict[str, float]] | None = None
        self._effective_time_hours: float | None = None
        self._last_calibration_monotonic: float | None = None
        self._last_calib_result: dict[str, object] | None = None

        self.mq_boundary_in: Optional[RabbitMQClient] = None
        self.mq_boundary_out: Optional[RabbitMQClient] = None
        self.mq_fluid: Optional[RabbitMQClient] = None
        self.mq_calibration: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None
        self._fluid_thread: Optional[Thread] = None

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    def setup(self) -> None:
        if self.mq_boundary_in is None:
            self.mq_boundary_in = RabbitMQClient(self.boundary_in_config)
        if self.mq_boundary_out is None:
            self.mq_boundary_out = RabbitMQClient(self.boundary_out_config)
        if self.mq_fluid is None:
            self.mq_fluid = RabbitMQClient(self.fluid_config)
        if self.publish_calibration and self.mq_calibration is None:
            self.mq_calibration = RabbitMQClient(self.calibration_config)
        if self.influx is None:
            self.influx = InfluxHelper(self.influx_config)
        self.logger.info(
            "Dependencies initialised (boundary_in=%s, boundary_out=%s, fluid=%s, pipe_boundary_set=%s, bc=%s)",
            self.boundary_in_config.queue,
            self.boundary_out_config.queue,
            self.fluid_config.queue,
            self.pipe_boundary_set,
            self.pipe_bc_type,
        )

    def start(self) -> None:
        if self.mq_boundary_in is None:
            self.setup()

        self._fluid_thread = Thread(target=self._consume_fluid, name="pipe-fluid-consumer", daemon=True)
        self._fluid_thread.start()

        self.logger.info("Listening for boundary snapshots on %s", self.boundary_in_config.queue)
        try:
            self.mq_boundary_in.consume(callback=self._on_boundary)
        except KeyboardInterrupt:  # pragma: no cover - runtime behaviour
            self.logger.info("Interrupt received; shutting down")
        finally:
            self.close()

    def _consume_fluid(self) -> None:
        try:
            self.mq_fluid.consume(callback=self._on_fluid)
        except Exception as exc:  # noqa: BLE001 - surfaced, thread exits
            self.logger.error("Fluid consumer stopped: %s", exc, exc_info=True)

    def stop(self) -> None:
        if self.mq_boundary_in and self.mq_boundary_in.channel:
            self.mq_boundary_in.channel.stop_consuming()
        if self.mq_fluid and self.mq_fluid.channel:
            try:
                self.mq_fluid.channel.stop_consuming()
            except Exception:  # noqa: BLE001 - best effort across threads
                pass

    def close(self) -> None:
        self.stop()
        for client in (self.mq_boundary_in, self.mq_boundary_out, self.mq_fluid, self.mq_calibration):
            if client is not None:
                try:
                    client.disconnect()
                except Exception:  # noqa: BLE001 - best effort
                    pass
        if self.influx is not None:
            self.influx.close()
        self.logger.info("Shutdown complete")

    # ------------------------------------------------------------------
    # FLUID STREAM  (slow loop: calibrate + cache effective temperature)
    # ------------------------------------------------------------------

    def _on_fluid(self, msg: dict) -> None:
        try:
            time_hours = float(msg["time_hours"])
            t_in = float(msg["t_inlet_c"])
            t_out = float(msg["t_outlet_c"])
        except (KeyError, TypeError, ValueError):
            self.logger.warning("Skipping malformed fluid message")
            return

        with self._lock:
            self._upsert_locked(time_hours, t_in, t_out)
            window = list(self._window)
            params = dict(self._best_params) if self._best_params else None
            do_calib = self._should_recalibrate_locked(len(window))
            if do_calib:
                self._last_calibration_monotonic = time.monotonic()

        if len(window) < self.min_points:
            self.logger.debug("Fluid window warming up (%d/%d points)", len(window), self.min_points)
            return

        times_s, tin, tout, latest_th = self._window_arrays(window)

        calib_result: dict[str, object] | None = None
        if do_calib:
            try:
                calib_result = calibrate_pipe(
                    times_s,
                    tin,
                    tout,
                    self.pipe_config,
                    p0=self.initial_params,
                    warm_start=params if self.use_warm_start else None,
                    fixed=self.fixed_params,
                    lower=self.lower_bounds,
                    upper=self.upper_bounds,
                    maxiter=self.maxiter,
                    maxfun=self.maxfun,
                )
                params = dict(calib_result["parameters"])  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001 - keep last good params
                self.logger.error("Pipe calibration failed: %s", exc, exc_info=True)

        if params is None:
            params = {**DEFAULT_P0, "Tw0": float(tout[0])}

        try:
            eff = float(
                effective_interface_temperature(
                    params,
                    times_s,
                    tin,
                    self.pipe_config,
                    source=self.interface_source,
                    reduction=self.interface_reduction,
                    tf_initial_c=float(tout[0]),
                )
            )
        except Exception as exc:  # noqa: BLE001 - skip this update
            self.logger.error("Effective interface temperature failed: %s", exc)
            return

        profile: list[dict[str, float]] | None = None
        if self.depth_resolved:
            try:
                pts = interface_temperature_profile(
                    params,
                    times_s,
                    tin,
                    self.pipe_config,
                    depths_m=self._profile_depths,
                    source=self.interface_source,
                    tf_initial_c=float(tout[0]),
                )
                profile = [{"z_m": float(z), "temperature": float(t)} for z, t in pts]
            except Exception as exc:  # noqa: BLE001 - fall back to scalar
                self.logger.error("Interface temperature profile failed: %s", exc)
                profile = None

        with self._lock:
            if calib_result is not None:
                self._best_params = dict(params)
                self._last_calib_result = calib_result
            self._effective_temp_c = eff
            self._effective_profile = profile
            self._effective_time_hours = latest_th

        if calib_result is not None:
            self.logger.info(
                "Pipe calibrated (t=%.3fh, rmse=%.4f, mae=%.4f, evals=%s, eff_interface=%.3f C, v=%.4e, kf=%.4e)",
                latest_th,
                float(calib_result.get("rmse", float("nan"))),
                float(calib_result.get("mae", float("nan"))),
                calib_result.get("eval_count"),
                eff,
                float(params["v"]),
                float(params["kf"]),
            )
            self._persist_calibration(calib_result, latest_th, eff)

    def _upsert_locked(self, time_hours: float, t_in: float, t_out: float) -> None:
        for idx, (existing_th, _, _) in enumerate(self._window):
            if abs(existing_th - time_hours) < 1e-9:
                self._window[idx] = (time_hours, t_in, t_out)
                break
        else:
            self._window.append((time_hours, t_in, t_out))

        self._window.sort(key=lambda item: item[0])

        if self.window_seconds > 0.0 and self._window:
            latest = self._window[-1][0]
            cutoff = latest - self.window_seconds / 3600.0
            self._window = [row for row in self._window if row[0] >= cutoff]
        if len(self._window) > self.max_points:
            self._window = self._window[-self.max_points :]

    def _should_recalibrate_locked(self, n_points: int) -> bool:
        if n_points < self.min_points:
            return False
        if self._last_calibration_monotonic is None:
            return True
        return (time.monotonic() - self._last_calibration_monotonic) >= self.recalibrate_seconds

    def _window_arrays(self, window: list[tuple[float, float, float]]):
        times_h = [row[0] for row in window]
        tin = [row[1] for row in window]
        tout = [row[2] for row in window]
        t0 = times_h[0]
        times_s = [max((th - t0) * 3600.0, 0.0) for th in times_h]
        return times_s, tin, tout, times_h[-1]

    def _persist_calibration(self, result: dict, time_hours: float, eff_temp_c: float) -> None:
        parameters = {name: float(result["parameters"][name]) for name in PARAM_NAMES}  # type: ignore[index]

        if self.influx is not None:
            try:
                self.influx.write_pipe_calibration(
                    parameters=parameters,
                    rmse=float(result.get("rmse", float("nan"))),
                    mae=float(result.get("mae", float("nan"))),
                    interface_temperature_c=eff_temp_c,
                    site=self.site_id,
                )
            except Exception as exc:  # noqa: BLE001 - infra
                self.logger.error("Failed to write pipe calibration to InfluxDB: %s", exc)

        if self.mq_calibration is not None:
            payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "status": "calibrated",
                "time_hours": float(time_hours),
                "site_id": self.site_id,
                "parameters": parameters,
                "interface_temperature_c": eff_temp_c,
                "pipe_boundary_set": self.pipe_boundary_set,
                "rmse": float(result.get("rmse", float("nan"))),
                "mae": float(result.get("mae", float("nan"))),
                "success": bool(result.get("success", False)),
                "message": str(result.get("message", "")),
                "eval_count": int(result.get("eval_count", 0)),
                "n_samples": int(result.get("n_samples", 0)),
                "travel_time_min": float(result.get("travel_time_min", float("nan"))),
            }
            try:
                self.mq_calibration.publish(payload)
            except Exception as exc:  # noqa: BLE001 - infra
                self.logger.error("Failed to publish pipe calibration: %s", exc)

    # ------------------------------------------------------------------
    # BOUNDARY STREAM  (fast loop: enrich + forward)
    # ------------------------------------------------------------------

    def _on_boundary(self, msg: dict) -> None:
        with self._lock:
            eff = self._effective_temp_c
            profile = list(self._effective_profile) if self._effective_profile else None

        enriched = deepcopy(msg)
        if eff is None:
            if not self.passthrough_until_ready:
                self.logger.debug("Pipe boundary not ready; dropping boundary message")
                return
            self.logger.debug("Pipe boundary not ready; forwarding boundary message unchanged")
        else:
            self._apply_pipe_boundary(enriched, float(eff), profile)

        if self.mq_boundary_out is None:
            raise RuntimeError("Boundary output client not initialised. Did you call setup()?")
        self.mq_boundary_out.publish(enriched)

    def _apply_pipe_boundary(
        self,
        msg: dict,
        eff_temp_c: float,
        profile: list[dict[str, float]] | None = None,
    ) -> None:
        boundaries = msg.get("boundaries")
        if not isinstance(boundaries, list):
            boundaries = []

        entry: dict[str, object] = {
            "name": "pipe_fluid",
            "boundary_set": self.pipe_boundary_set,
            "bc_type": self.pipe_bc_type,
            # Scalar kept as fallback / diagnostic; the profile takes precedence
            # in the solver when present.
            "temperature": eff_temp_c,
        }
        if self.pipe_bc_type == "robin":
            entry["h_coeff"] = self.pipe_h_coeff
        if profile:
            entry["temperature_profile"] = profile

        replaced = False
        for idx, existing in enumerate(boundaries):
            if isinstance(existing, dict) and existing.get("boundary_set") == self.pipe_boundary_set:
                # Preserve the configured name if present.
                if existing.get("name"):
                    entry["name"] = existing["name"]
                boundaries[idx] = entry
                replaced = True
                break
        if not replaced:
            boundaries.append(entry)

        msg["boundaries"] = boundaries


if __name__ == "__main__":
    PipeBoundaryServer().start()
