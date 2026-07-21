"""Start the PolyScience circulator actuation service (RS232 hardware bridge)."""

from __future__ import annotations

from software.startup.service_runners import (
    build_influx_config,
    create_polyscience_control_server,
)
from software.startup.utils.config import configure_logging, load_startup_config


def main() -> None:
    configure_logging()
    config = load_startup_config()
    influx_defaults = build_influx_config(config.get("influxdb", {}))
    server = create_polyscience_control_server(
        config.get("rabbitmq", {}),
        config.get("polyscience_control", {}),
        influx_defaults,
    )
    server.start()


if __name__ == "__main__":
    main()
