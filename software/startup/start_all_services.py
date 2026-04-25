"""Convenience launcher for the permafrost digital twin.

Uncomment the services you want to start and run:

    python -m software.startup.start_all_services

Each selected service runs in its own daemon process, so stopping the launcher
will shut everything down. Docker helpers for RabbitMQ/InfluxDB are provided
but commented out so you can decide whether to trigger them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from software.startup.docker_services.start_influxdb import start_docker_influxdb
from software.startup.docker_services.start_rabbitmq import start_docker_rabbitmq
from software.startup.service_runners import (
    build_influx_config,
    initialize_service_config,
    _observation_ingestion_2d_runner,
    _boundary_condition_builder_runner,
    _fem_forecast_runner,
    _heater_actuation_runner,
    _fem_inversion_runner,
    _safety_monitor_runner,
    _thaw_front_reconstruction_runner,
    _viz_dashboard_runner,
)
from software.startup.utils.daemon import start_as_daemon
from software.startup.utils.config import configure_logging, load_startup_config


def main() -> None:
    configure_logging()
    fem_fields_dir = Path("artifacts/fem_fields")
    if fem_fields_dir.exists():
        for csv_file in fem_fields_dir.glob("*.csv"):
            csv_file.unlink()

    config = load_startup_config()

    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg_values = config.get("influxdb", {})
    obs_2d_cfg = config.get("observation_ingestion_2d", {})
    bc_cfg = config.get("boundary_condition_builder", {})
    thaw_cfg = config.get("thaw_front_reconstruction", {})
    inversion_cfg = config.get("fem_inversion", {})
    fem_cfg = config.get("fem_forecast", {})
    safety_cfg = config.get("safety_monitor", {})
    heater_cfg = config.get("heater_actuation", {})

    influx_defaults = build_influx_config(influx_cfg_values)

    # Initialize the shared config dict (works across process boundaries on both Linux and Windows)
    config_dict = initialize_service_config(
        influx_config=influx_defaults,
        rabbit_cfg=rabbit_cfg,
        obs_2d_cfg=obs_2d_cfg,
        bc_cfg=bc_cfg,
        thaw_cfg=thaw_cfg,
        inversion_cfg=inversion_cfg,
        fem_cfg=fem_cfg,
        safety_cfg=safety_cfg,
        heater_cfg=heater_cfg,
    )

    # Map service names to runner functions
    available_services = {
        "observation_ingestion_2d": _observation_ingestion_2d_runner,
        "boundary_condition_builder": _boundary_condition_builder_runner,
        "thaw_front_reconstruction": _thaw_front_reconstruction_runner,
        "fem_inversion": _fem_inversion_runner,
        "fem_forecast": _fem_forecast_runner,
        "safety_monitor": _safety_monitor_runner,
        "heater_actuation": _heater_actuation_runner,
        "viz_dashboard": _viz_dashboard_runner,
    }

    services_to_start = [
        "observation_ingestion_2d",
        "boundary_condition_builder",
        "thaw_front_reconstruction",
        "fem_inversion",
        "fem_forecast",
        "safety_monitor",
        "heater_actuation",
        "viz_dashboard",
    ]

    # Optionally start local Docker dependencies first.
    start_docker_rabbitmq()
    start_docker_influxdb()

    processes = []

    for service_name in services_to_start:
        runner = available_services.get(service_name)
        if runner is None:
            raise KeyError(f"Unknown service '{service_name}'. Available: {sorted(available_services)}")
        processes.append(
            start_as_daemon(runner, process_name=service_name, kwargs={"config_dict": config_dict})
        )

    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
