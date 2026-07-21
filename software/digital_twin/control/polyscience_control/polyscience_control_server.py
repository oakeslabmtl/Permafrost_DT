"""PolyScience circulator actuation service (RS232 hardware bridge).

This is the *output gate* of the digital twin's control loop, the mirror image
of ``cr1000_publisher_server`` on the sensing side: where the CR1000 bridge
turns hardware readings into RabbitMQ messages, this bridge turns RabbitMQ
heater commands into RS232 commands on the physical PolyScience Advanced Digital
Temperature Controller (the circulator that heats/cools the fluid loop, i.e. the
physical twin of the ``left`` heater boundary in the FEM).

Flow::

    safety_monitor --alert--> heater_actuation --heater_command--> [THIS SERVICE] --RS232--> PolyScience

It consumes ``heater_command_message`` payloads from ``permafrost.command.heater.2d``
(the same queue ``heater_actuation`` publishes to) and translates each into
concrete controller commands:

* ``action`` -> ON (``SO1``) / OFF (``SO0``) / hold, via a configurable action map
* ``setpoint`` (C) -> ``SS`` command (range-clamped for safety)

Applied actions are read back from the controller and recorded to InfluxDB
(``heater_actions`` measurement, ``source=polyscience``) so the dashboard/history
reflect the *actual* hardware state, not just the intended command.

**Serial-port exclusivity** — pyserial opens the port exclusively, so this
service and the CR1000 publisher must be configured on *different* COM ports.
The launcher (``start_all_services``) enforces this at startup; at runtime a
failed open is reported with an explicit diagnostic and retried via the
reconnect loop.

The RS232 command set / response protocol mirrors the validated test tool
``software/tools/polyscience_control_serial.py``.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    import serial
    _SERIAL_AVAILABLE = True
    _SERIAL_SETTINGS = {
        "bytesize": serial.EIGHTBITS,
        "parity": serial.PARITY_NONE,
        "stopbits": serial.STOPBITS_ONE,
        "xonxoff": False,
        "rtscts": False,
    }
except ImportError:  # pragma: no cover - exercised only without pyserial
    _SERIAL_AVAILABLE = False
    serial = None  # type: ignore[assignment]
    _SERIAL_SETTINGS = {}

from software.digital_twin.communication.messaging import (
    RabbitMQClient,
    RabbitMQConfig,
    resolve_queue_config,
)
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.utils.logging_setup import get_logger

COMMAND_QUEUE = "permafrost.command.heater.2d"
COMMAND_SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "communication" / "schemas" / "heater_command_message.json"
)

# Maps the abstract control action (as emitted by heater_actuation) to the
# PolyScience power state. "hold" leaves the ON/OFF state untouched.
DEFAULT_ACTION_MAP: Dict[str, str] = {
    "start": "on",
    "run": "on",
    "on": "on",
    "heat": "on",
    "stop": "off",
    "off": "off",
    "shutdown": "off",
    "hold": "hold",
    "maintain": "hold",
    "none": "hold",
}


class PolyScienceControlServer:
    """Consume heater commands and drive a PolyScience circulator over RS232.

    The serial connection is opened lazily and re-established on demand so a
    transient hardware fault (unplugged cable, controller reboot) never crashes
    the consumer — a command that arrives while the port is down is retried
    against a freshly opened port before being reported as failed.
    """

    def __init__(
        self,
        port: str,
        baud_rate: int = 57600,
        command_config: Optional[RabbitMQConfig] = None,
        influx_config: Optional[InfluxConfig] = None,
        *,
        command_queue: Optional[str] = None,
        action_map: Optional[Dict[str, str]] = None,
        setpoint_min_c: float = -30.0,
        setpoint_max_c: float = 50.0,
        serial_timeout_s: float = 3.0,
        inter_command_delay_s: float = 0.2,
        open_retries: int = 3,
        open_retry_delay_s: float = 2.0,
        verify_after_apply: bool = True,
        stop_on_shutdown: bool = True,
        site_id: str = "default",
        dry_run: bool = False,
    ) -> None:
        self.logger = get_logger("PolyScienceControlServer")
        self.port = port
        self.baud_rate = int(baud_rate)
        self.setpoint_min_c = float(setpoint_min_c)
        self.setpoint_max_c = float(setpoint_max_c)
        self.serial_timeout_s = float(serial_timeout_s)
        # Minimum gap enforced between consecutive RS232 commands. The controller
        # drops a command that arrives while it is still processing the previous
        # one (observed as a TIMEOUT on the SO1/SO0 power command sent right after
        # an SS setpoint ack), so back-to-back writes must be spaced out.
        self.inter_command_delay_s = max(0.0, float(inter_command_delay_s))
        self._last_command_monotonic: float = 0.0
        self.open_retries = max(1, int(open_retries))
        self.open_retry_delay_s = float(open_retry_delay_s)
        self.verify_after_apply = bool(verify_after_apply)
        self.stop_on_shutdown = bool(stop_on_shutdown)
        self.site_id = site_id
        # dry_run drives the full pipeline without touching hardware — useful for
        # testing integration without the physical twin, or when pyserial is
        # unavailable. It is auto-enabled if pyserial cannot be imported.
        self.dry_run = bool(dry_run) or not _SERIAL_AVAILABLE
        if self.dry_run and not _SERIAL_AVAILABLE:
            self.logger.warning(
                "pyserial not installed — running in DRY-RUN mode (commands logged, "
                "not sent). Run: pip install pyserial"
            )

        self.action_map = {k.lower(): v for k, v in (action_map or DEFAULT_ACTION_MAP).items()}

        self.command_config = resolve_queue_config(
            command_config,
            queue=command_queue or COMMAND_QUEUE,
            schema_path=COMMAND_SCHEMA,
        )
        self.influx_config = influx_config

        self.mq_in: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None
        self._serial = None
        self._serial_lock = threading.Lock()
        # The PolyScience holds its setpoint once written, so there is no need to
        # re-send SS on every heater command. Track the last value we applied and
        # skip the SS write when it is unchanged (cuts serial traffic in half and
        # keeps the power command from being crowded out).
        self._last_setpoint_c: Optional[float] = None

    # ------------------------------------------------------------------
    # SERIAL LINK
    # ------------------------------------------------------------------

    def _open_serial(self) -> None:
        """(Re)open the serial port, retrying a few times before giving up."""
        if self.dry_run:
            return
        if self._serial is not None and getattr(self._serial, "is_open", False):
            return

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.open_retries + 1):
            try:
                self._serial = serial.Serial(
                    port=self.port,
                    baudrate=self.baud_rate,
                    timeout=self.serial_timeout_s,
                    **_SERIAL_SETTINGS,
                )
                self.logger.info(
                    "PolyScience serial port %s opened at %d baud", self.port, self.baud_rate
                )
                return
            except Exception as exc:  # noqa: BLE001 - surfaced below
                last_exc = exc
                self.logger.warning(
                    "Could not open %s (attempt %d/%d): %s",
                    self.port, attempt, self.open_retries, exc,
                )
                if attempt < self.open_retries:
                    threading.Event().wait(self.open_retry_delay_s)

        self.logger.error(
            "Failed to open PolyScience port %s after %d attempts. Check: port correct "
            "(not shared with the CR1000 service — pyserial opens ports exclusively), "
            "controller in RS232 mode, baud matches, cable connected.",
            self.port, self.open_retries,
        )
        raise ConnectionError(f"Cannot open serial port {self.port}: {last_exc}")

    def _close_serial(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
            self._serial = None

    def _send_command(self, command: str) -> str:
        """Send one RS232 command and return the trimmed response.

        ``!`` prefix = success, ``?`` prefix = rejected. ``TIMEOUT`` / ``ERROR:``
        indicate a link problem. In dry-run mode a synthetic ``!`` ack is
        returned so the pipeline can be exercised without hardware.
        """
        if self.dry_run:
            self.logger.info("[DRY-RUN] would send '%s'", command)
            return "!DRYRUN"

        with self._serial_lock:
            self._open_serial()
            # Space this command from the previous one so the controller isn't
            # still busy when it arrives (otherwise the reply is dropped).
            if self.inter_command_delay_s > 0.0:
                elapsed = time.monotonic() - self._last_command_monotonic
                remaining = self.inter_command_delay_s - elapsed
                if remaining > 0.0:
                    time.sleep(remaining)
            try:
                self._serial.reset_input_buffer()
                self._serial.write((command + "\r").encode("ascii"))
                response_bytes = self._serial.read_until(b"\r", size=64)
            except Exception as exc:  # noqa: BLE001 - drop link, reopen next time
                self.logger.error("Serial I/O error on '%s': %s", command, exc)
                self._close_serial()
                return f"ERROR: {exc}"
            finally:
                self._last_command_monotonic = time.monotonic()

        if not response_bytes:
            self.logger.error("No response to '%s' (timeout) — link or baud problem", command)
            return "TIMEOUT"
        try:
            return response_bytes.decode("ascii").strip()
        except UnicodeDecodeError:
            self.logger.error("Undecodable response to '%s' — likely baud mismatch", command)
            return "DECODE_ERROR"

    @staticmethod
    def _is_ok(response: str) -> bool:
        return response.startswith("!")

    # ------------------------------------------------------------------
    # COMMAND -> ACTUATION
    # ------------------------------------------------------------------

    def _clamp_setpoint(self, value: float) -> float:
        clamped = max(self.setpoint_min_c, min(self.setpoint_max_c, value))
        if clamped != value:
            self.logger.warning(
                "Setpoint %.2fC out of range [%.1f, %.1f] — clamped to %.2fC",
                value, self.setpoint_min_c, self.setpoint_max_c, clamped,
            )
        return clamped

    def _apply_setpoint(self, value: float) -> bool:
        setpoint = self._clamp_setpoint(float(value))
        # Only write when the target actually moved — the controller retains the
        # last setpoint, so re-sending it every command is pointless serial churn.
        if self._last_setpoint_c is not None and setpoint == self._last_setpoint_c:
            self.logger.debug("Setpoint already at %.2fC — not resending", setpoint)
            return True
        response = self._send_command(f"SS{setpoint:06.2f}")
        if self._is_ok(response):
            self._last_setpoint_c = setpoint
            self.logger.info("Setpoint set to %.2fC", setpoint)
            return True
        self.logger.error("Setpoint command failed — response: '%s'", response)
        return False

    def _apply_power(self, mode: str) -> bool:
        if mode == "hold":
            self.logger.info("Action 'hold' — leaving heater power state unchanged")
            return True
        cmd = "SO1" if mode == "on" else "SO0"
        response = self._send_command(cmd)
        if self._is_ok(response):
            self.logger.info("Heater turned %s", mode.upper())
            return True
        self.logger.error("Power command %s failed — response: '%s'", cmd, response)
        return False

    def _read_status(self) -> Optional[str]:
        """Return 'on'/'off'/None for the controller's current power state."""
        response = self._send_command("RO")
        if response == "!1":
            return "on"
        if response == "!0":
            return "off"
        return None

    def _process_command(self, msg: dict) -> None:
        action = str(msg.get("action", "")).lower()
        setpoint = msg.get("setpoint")
        time_hours = float(msg.get("time_hours", 0.0))
        mode = self.action_map.get(action)
        if mode is None:
            self.logger.warning(
                "Unknown action '%s' — treating as 'hold' (no power change)", action
            )
            mode = "hold"

        self.logger.info(
            "Command received (t=%.2fh, action=%s -> %s, setpoint=%s, reason=%s)",
            time_hours, action, mode,
            "n/a" if setpoint is None else f"{float(setpoint):.2f}C",
            msg.get("reason", "n/a"),
        )

        ok = True
        # Apply the setpoint before power so the controller heats toward the
        # right target the moment it is switched on.
        if setpoint is not None:
            ok = self._apply_setpoint(setpoint) and ok
        ok = self._apply_power(mode) and ok

        applied_state = self._read_status() if (self.verify_after_apply and not self.dry_run) else None
        status = "applied" if ok else "failed"
        if applied_state is not None:
            status = f"{status}:{applied_state}"

        self._record_action(msg, action, setpoint, status)

        if ok:
            self.logger.info("Command applied successfully (status=%s)", status)
        else:
            self.logger.error(
                "Command did NOT fully apply (status=%s) — check controller/link. "
                "If this was a safety STOP, verify the heater manually.",
                status,
            )

    def _record_action(self, msg: dict, action: str, setpoint, status: str) -> None:
        if self.influx is None:
            return
        try:
            self.influx.write_heater_action(
                time_hours=float(msg.get("time_hours", 0.0)),
                action=action,
                setpoint=float(setpoint) if setpoint is not None else None,
                triggered=bool(msg.get("triggered", False)),
                reason=str(msg.get("reason", "unknown")),
                source="polyscience",
                site=self.site_id,
            )
        except Exception as exc:  # noqa: BLE001 - recording must not stop actuation
            self.logger.error("Failed to record heater action to InfluxDB: %s", exc)

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    def setup(self) -> None:
        if self.mq_in is None:
            self.mq_in = RabbitMQClient(self.command_config)
        if self.influx is None and self.influx_config is not None:
            self.influx = InfluxHelper(self.influx_config)
        if not self.dry_run:
            try:
                self._open_serial()
                state = self._read_status()
                self.logger.info(
                    "PolyScience controller reachable (current power state: %s)",
                    state or "unknown",
                )
            except Exception as exc:  # noqa: BLE001 - allow start; reconnect on demand
                self.logger.error(
                    "PolyScience controller not reachable at startup: %s. "
                    "Service will keep running and retry on the next command.", exc,
                )
        self.logger.info(
            "PolyScienceControlServer ready (port=%s, baud=%d, queue=%s, dry_run=%s)",
            self.port, self.baud_rate, self.command_config.queue, self.dry_run,
        )

    def start(self) -> None:
        if self.mq_in is None:
            self.setup()
        self.logger.info("Listening for heater commands on %s", self.command_config.queue)
        try:
            self.mq_in.consume(callback=self._process_command)
        except KeyboardInterrupt:  # pragma: no cover - runtime behaviour
            self.logger.info("Interrupt received; shutting down")
        finally:
            self.close()

    def stop(self) -> None:
        if self.mq_in and self.mq_in.channel:
            self.mq_in.channel.stop_consuming()

    def close(self) -> None:
        self.stop()
        # Fail safe: leave the heater OFF when the actuator is not supervising it.
        if self.stop_on_shutdown and not self.dry_run and self._serial is not None:
            try:
                self.logger.info("Turning heater OFF on shutdown (fail-safe)")
                self._apply_power("off")
            except Exception as exc:  # noqa: BLE001 - best effort
                self.logger.error("Could not turn heater off on shutdown: %s", exc)
        self._close_serial()
        if self.mq_in is not None:
            self.mq_in.disconnect()
        if self.influx is not None:
            self.influx.close()
        self.logger.info("Shutdown complete")


if __name__ == "__main__":
    PolyScienceControlServer(port="COM3").start()
