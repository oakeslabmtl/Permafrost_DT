"""Start the pipe boundary enrichment service."""

from __future__ import annotations

from software.startup.service_runners import build_influx_config, create_pipe_boundary_server
from software.startup.utils.config import configure_logging, load_startup_config


def main() -> None:
    configure_logging()
    config = load_startup_config()
    influx_defaults = build_influx_config(config.get("influxdb", {}))
    server = create_pipe_boundary_server(
        influx_defaults,
        config.get("rabbitmq", {}),
        config.get("pipe_boundary", {}),
    )
    server.start()


if __name__ == "__main__":
    main()
