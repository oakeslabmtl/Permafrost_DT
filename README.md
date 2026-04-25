# Permafrost Digital Twin (permafrost_DT)

This repository implements a digital twin system for permafrost thermal monitoring and control. The system uses finite element methods (FEM) for both forward simulation and material parameter inversion, coupled with real-time sensor ingestion, thaw-front reconstruction, safety monitoring, and heater actuation.

# Contents
- [About this Project](#about-this-project)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Quick service bootstrap](#quick-service-bootstrap)
- [Running individual services](#running-individual-services)
- [Tools](#tools)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)

# About this Project

**Goal:** Provide accurate real-time thermal modeling and closed-loop control for permafrost, enabling continuous monitoring, parameter estimation, thaw-front tracking, and automated heater control.

**Key Features:**
- Real-time 2D temperature field simulation using FEM
- FEM-based material parameter inversion from sensor data
- Boundary condition estimation from 2D sensor arrays
- Thaw-front reconstruction and safety limit enforcement
- Automated heater actuation based on safety alerts
- Streamlit visualization dashboard backed by InfluxDB
- Message-based microservice architecture via RabbitMQ

# System Architecture

Services communicate asynchronously via RabbitMQ queues. InfluxDB stores all time-series data.

## Data Flow

```
2D Sensors
    │
    ▼
Observation Ingestion ──► InfluxDB
    │ (fanout: permafrost.sensors.2d)
    ├──► Boundary Condition Builder ──► boundary.2d ──► FEM Forecast ──► fem_forecast.2d ──► Safety Monitor
    │                                                └──► FEM Inversion ──► fem_inversion ──► InfluxDB          │
    ├──► Thaw Front Reconstruction ──► thaw_front.2d ──────────────────────────────────────► Safety Monitor     │
    │                                                                                              │             │
    │                                                                                              ▼             │
    │                                                                                       safety_alerts.2d     │
    │                                                                                              │             │
    │                                                                                              ▼             │
    │                                                                                      Heater Actuation       │
    │                                                                                    (command + action queues)│
    └──────────────────────────────────────────────────────────────────────────────────────────────────────────►  Viz Dashboard (reads InfluxDB directly)
```

## Services

### Monitoring
1. **Observation Ingestion** (`monitoring/observation_ingestion`)
   - Consumes 2D sensor snapshots from the `permafrost.sensors.2d` fanout exchange
   - Writes temperature observations to InfluxDB
   - Entry point: `start_observation_ingestion_2d.py`

2. **Boundary Condition Builder** (`monitoring/boundary_condition_builder`)
   - Derives FEM boundary conditions (Robin/Neumann) from sensor data
   - Publishes structured boundary messages to `permafrost.record.boundary.2d`
   - Entry point: `start_boundary_condition_builder.py`

3. **Thaw Front Reconstruction** (`monitoring/thaw_front_reconstruction`)
   - Interpolates 2D sensor fields and extracts the 0 °C isotherm
   - Publishes thaw-front metrics and contour points
   - Entry point: `start_thaw_front_reconstruction.py`

### Simulators
4. **FEM Forecast** (`simulator/fem_forecast`)
   - 2D finite element solver over a triangle mesh
   - Runs a thermal forecast horizon from boundary conditions
   - Publishes thaw-front radius and node-level field snapshots
   - Mesh files live in `software/models/fem_mesh/`
   - Entry point: `start_fem_forecast.py`

5. **FEM Inversion** (`simulator/fem_inversion`)
   - Estimates material parameters (porosity, thermal conductivity, heat capacity) from sensor data
   - Optimises against FEM simulations using `scipy.optimize`
   - Publishes inverted parameters and validation metrics
   - Entry point: `start_fem_inversion.py`

### Control
6. **Safety Monitor** (`control/safety_monitor`)
   - Compares thaw-front radius against a configurable safety limit
   - Fuses thaw-front reconstruction and FEM forecast inputs
   - Emits safety alerts to `permafrost.record.safety_alerts.2d`
   - Entry point: `start_safety_monitor.py`

7. **Heater Actuation** (`control/heater_actuation`)
   - Translates safety alerts into heater commands (stop / hold)
   - Publishes both the command and a logged action record
   - Entry point: `start_heater_actuation.py`

### Visualization
8. **Viz Dashboard** (`visualization/viz_dashboard`)
   - Streamlit app that queries InfluxDB directly
   - Displays temperature fields, thaw-front evolution, inversion results, and safety status
   - Entry point: `start_viz_dashboard.py`

## Infrastructure

- **Communication** (`digital_twin/communication/`) — RabbitMQ client, JSON schema validation, logging
- **Data Access** (`digital_twin/data_access/`) — InfluxDB helpers, Docker compose assets
- **Startup** (`software/startup/`) — service launchers, daemon runner, config loader
- **Tools** (`software/tools/`) — standalone utilities (see [Tools](#tools))

## Repository Structure

```
permafrost_DT/
├── startup.conf                  # Service configuration
├── requirements.txt              # Python dependencies
├── pytest.ini                   # Test configuration
├── artifacts/                   # Runtime outputs (FEM fields, case results)
├── data/                        # Data storage and exports
├── docs/                        # Architecture diagrams
├── logs/                        # Service logs (optional)
└── software/
    ├── digital_twin/
    │   ├── communication/        # RabbitMQ client + JSON schemas
    │   ├── control/              # Safety monitor, heater actuation
    │   ├── data_access/          # InfluxDB helpers + Docker assets
    │   ├── monitoring/           # Observation ingestion, BC builder, thaw front
    │   ├── simulator/            # FEM forecast, FEM inversion
    │   └── visualization/        # Streamlit dashboard
    ├── models/
    │   └── fem_mesh/             # Triangle mesh files (nodes, elements, edge sets)
    ├── startup/                  # Service launchers and daemon utilities
    ├── tests/                    # Unit and integration tests
    ├── tools/                    # CLI utilities
    └── utils/                    # Shared logging setup
```

# Prerequisites

- Python 3.10+
- pip
- Docker & Docker Compose
- Git

# Getting Started

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/simplyadil/permafrost_DT.git
   cd permafrost_DT
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Upgrade pip tooling**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install the package in editable mode**
   ```bash
   pip install -e .
   ```

## Configuration

All service parameters live in `startup.conf`. The defaults target a fully local developer setup. Review and adjust before starting:

- `rabbitmq` / `influxdb` — connection settings
- `fem_forecast` — mesh paths, material properties, solver settings
- `fem_inversion` — inversion parameters, bounds, convergence settings
- `boundary_condition_builder` — boundary types (Robin/Neumann) and sensor overrides
- `safety_monitor` — thaw-front radius limit and metric (`max` or `avg`)
- `heater_actuation` — actions on alert/OK and setpoints
- `viz_dashboard` — host, port, refresh interval, history depth

### Infrastructure Setup

Start InfluxDB and RabbitMQ via Docker:

```bash
# Using helper scripts
python software/startup/docker_services/start_influxdb.py
python software/startup/docker_services/start_rabbitmq.py

# Or directly with docker compose
cd software/digital_twin/data_access/influxdbserver && docker compose up -d
cd software/digital_twin/communication/installation/rabbitmq && docker compose up -d
```

### Quick service bootstrap

Once Docker services are up:

```bash
python -m software.startup.start_all_services
```

This launches all services as daemon processes. Comment out any services you don't need inside `start_all_services.py` before running.

## Running individual services

Each service can be launched independently:

```bash
python -m software.startup.start_observation_ingestion_2d
python -m software.startup.start_boundary_condition_builder
python -m software.startup.start_thaw_front_reconstruction
python -m software.startup.start_fem_forecast
python -m software.startup.start_fem_inversion
python -m software.startup.start_safety_monitor
python -m software.startup.start_heater_actuation
python -m software.startup.start_viz_dashboard
```

The dashboard can also be launched directly via Streamlit:

```bash
streamlit run software/digital_twin/visualization/viz_dashboard/viz_dashboard_app.py
```

# Tools

Standalone utilities in `software/tools/`:

- **`replay_csv.py`** — Replay a CSV of sensor readings into the RabbitMQ sensor queue (useful for testing without live hardware)
  ```bash
  python -m software.tools.replay_csv --file data/exports/my_sensors.csv
  ```

- **`run_fem_case.py`** — Run a standalone FEM simulation case and write results to `artifacts/fem_cases/`
  ```bash
  python -m software.tools.run_fem_case
  ```

- **`export_fem_results.py`** — Export FEM field results from InfluxDB to CSV
  ```bash
  python -m software.tools.export_fem_results
  ```

# Tests

Tests live in `software/tests/`. Run with:

```bash
pytest -q
```

# Troubleshooting

### RabbitMQ Connection Failed
- Check container: `docker ps | grep rabbitmq`
- Management UI: http://localhost:15672 (credentials: `permafrost` / `permafrost`)
- Verify `host` in `startup.conf` matches your setup

### InfluxDB Not Reachable
- Check health: http://localhost:8086/health
- Check container: `docker ps | grep influxdb`
- Verify `url` and `token` in `startup.conf`

### Import Errors
- Ensure the package is installed: `pip install -e .`
- Run scripts from the repository root using module notation: `python -m software.startup.<service>`
- Verify the virtual environment is activated

### FEM Mesh Not Found
- Confirm mesh files exist in `software/models/fem_mesh/`
- Check `fem_forecast.mesh_dir`, `node_file`, `element_file`, and `edge_sets` paths in `startup.conf`
