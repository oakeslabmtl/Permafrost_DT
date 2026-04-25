"""Export FEM forecast outputs (and optional measured comparisons) to CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from software.digital_twin.data_access.influx_utils import InfluxHelper
from software.startup.service_runners import build_influx_config
from software.startup.utils.config import load_startup_config


def _range_start(hours: float) -> str:
    seconds = max(int(hours * 3600), 60)
    return f"-{seconds}s"


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _filter_horizon(df: pd.DataFrame, horizon_hours: Optional[float], tolerance: float = 1e-3) -> pd.DataFrame:
    if horizon_hours is None or df.empty or "horizon_hours" not in df.columns:
        return df
    series = pd.to_numeric(df["horizon_hours"], errors="coerce")
    return df[series.sub(horizon_hours).abs() <= tolerance]


def _merge_forecast_with_measured(
    forecast_df: pd.DataFrame,
    measured_df: pd.DataFrame,
    *,
    tolerance_hours: float,
) -> pd.DataFrame:
    if forecast_df.empty or measured_df.empty:
        return pd.DataFrame()

    forecast = forecast_df.copy()
    measured = measured_df.copy()

    forecast["time_hours"] = pd.to_numeric(forecast["time_hours"], errors="coerce")
    forecast["horizon_hours"] = pd.to_numeric(forecast["horizon_hours"], errors="coerce")
    forecast["radius_max_m"] = pd.to_numeric(forecast.get("radius_max_m"), errors="coerce")
    forecast["radius_avg_m"] = pd.to_numeric(forecast.get("radius_avg_m"), errors="coerce")

    measured["time_hours"] = pd.to_numeric(measured["time_hours"], errors="coerce")
    measured["radius_max_m"] = pd.to_numeric(measured.get("radius_max_m"), errors="coerce")
    measured["radius_avg_m"] = pd.to_numeric(measured.get("radius_avg_m"), errors="coerce")

    forecast = forecast.dropna(subset=["time_hours", "horizon_hours"])
    measured = measured.dropna(subset=["time_hours"])
    forecast["time_target"] = forecast["time_hours"] + forecast["horizon_hours"]
    measured = measured.sort_values("time_hours")

    merged = pd.merge_asof(
        forecast.sort_values("time_target"),
        measured,
        left_on="time_target",
        right_on="time_hours",
        direction="nearest",
        tolerance=tolerance_hours,
        suffixes=("_forecast", "_measured"),
    )
    merged = merged.dropna(subset=["time_hours_measured"])
    if merged.empty:
        return merged

    merged["r_max_error_m"] = (merged["radius_max_m_measured"] - merged["radius_max_m_forecast"]).abs()
    merged["r_avg_error_m"] = (merged["radius_avg_m_measured"] - merged["radius_avg_m_forecast"]).abs()
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FEM forecast outputs to CSV.")
    parser.add_argument("--out-dir", default="data/exports", help="Directory for CSV outputs.")
    parser.add_argument("--site-id", default="default", help="Site ID tag to filter.")
    parser.add_argument("--range-hours", type=float, default=8.0, help="Lookback window in hours.")
    parser.add_argument("--horizon-hours", type=float, default=None, help="Optional horizon filter.")
    parser.add_argument(
        "--match-tolerance-hours",
        type=float,
        default=0.02,
        help="Allowed time mismatch for forecast vs measured comparison (hours).",
    )
    parser.add_argument(
        "--prefix",
        default="fem_export",
        help="Filename prefix for exported CSV files.",
    )
    args = parser.parse_args()

    config = load_startup_config()
    influx_cfg = build_influx_config(config.get("influxdb", {}))
    influx = InfluxHelper(influx_cfg)

    range_start = _range_start(args.range_hours)
    forecast_metrics = influx.query_fem_forecast_metrics(
        site=args.site_id,
        range_start=range_start,
        limit=20000,
    )
    forecast_points = influx.query_fem_forecast_points(
        site=args.site_id,
        range_start=range_start,
        limit=200000,
    )
    thaw_metrics = influx.query_thaw_front_metrics(
        site=args.site_id,
        range_start=range_start,
        limit=20000,
    )

    influx.close()

    forecast_metrics = _filter_horizon(forecast_metrics, args.horizon_hours)
    forecast_points = _filter_horizon(forecast_points, args.horizon_hours)

    out_dir = Path(args.out_dir)
    metrics_path = out_dir / f"{args.prefix}_metrics.csv"
    points_path = out_dir / f"{args.prefix}_points.csv"
    comparison_path = out_dir / f"{args.prefix}_comparison.csv"

    _write_csv(forecast_metrics, metrics_path)
    _write_csv(forecast_points, points_path)

    comparison = _merge_forecast_with_measured(
        forecast_metrics,
        thaw_metrics,
        tolerance_hours=float(args.match_tolerance_hours),
    )
    if not comparison.empty:
        _write_csv(comparison, comparison_path)

    print(f"Saved forecast metrics to: {metrics_path}")
    print(f"Saved forecast points to: {points_path}")
    if not comparison.empty:
        print(f"Saved forecast vs measured comparison to: {comparison_path}")
    else:
        print("No forecast/measured matches found; comparison CSV not written.")


if __name__ == "__main__":
    main()
