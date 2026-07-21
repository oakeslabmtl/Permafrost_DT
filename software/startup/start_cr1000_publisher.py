"""Start the CR1000 hardware publisher service."""

from __future__ import annotations

from software.startup.service_runners import create_cr1000_publisher_server
from software.startup.utils.config import configure_logging, load_startup_config


def main() -> None:
    configure_logging()
    config = load_startup_config()
    server = create_cr1000_publisher_server(
        config.get("rabbitmq", {}),
        config.get("cr1000_publisher", {}),
        config.get("influxdb", {}),
    )
    server.start()


if __name__ == "__main__":
    main()
