"""Convert a raw CR1000 datalogger CSV into replay-ready inputs.

The raw logger table uses a datetime ``TIMESTAMP`` column and carries the fluid
``Inlet``/``Outlet`` channels inline. ``replay_csv`` instead needs a numeric
``Time (h)`` column (it float-parses the time and silently drops non-numeric
rows) and reads fluid temps from a separate ``.xlsx`` with fixed column names.

This tool writes two files next to the source (or under ``--out-dir``):

* ``<stem>_replay.csv``  — original columns + a leading ``Time (h)`` column,
  for ``replay_csv --path``.
* ``<stem>_fluid.xlsx``  — ``Time (h)`` + ``Fluid Temperature_Inlet`` +
  ``Fluid Temperatue_Outlet`` (note the logger-matching spelling replay expects),
  for ``replay_csv --fluid-path``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Sensors excluded from the live DT (see cr1000_publisher config). Dropped here
# so a replay run matches the live pipeline instead of injecting bad channels.
DEFAULT_EXCLUDE = ["H10D75"]


def convert(
    path: Path,
    *,
    out_dir: Path | None,
    timestamp_col: str,
    inlet_col: str,
    outlet_col: str,
    exclude: list[str],
) -> tuple[Path, Path | None]:
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(
            f"Timestamp column {timestamp_col!r} not found. Columns: {list(df.columns)[:8]}..."
        )

    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    if ts.isna().all():
        raise ValueError(f"Could not parse any datetimes from {timestamp_col!r}.")
    df = df.loc[~ts.isna()].copy()
    ts = ts.loc[~ts.isna()]
    time_h = (ts - ts.iloc[0]).dt.total_seconds() / 3600.0

    for col in exclude:
        if col in df.columns:
            df = df.drop(columns=col)

    out_dir = out_dir or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    sensor_df = df.copy()
    sensor_df.insert(0, "Time (h)", time_h.to_numpy())
    sensor_path = out_dir / f"{stem}_replay.csv"
    sensor_df.to_csv(sensor_path, index=False)

    fluid_path: Path | None = None
    if inlet_col in df.columns and outlet_col in df.columns:
        fluid_df = pd.DataFrame(
            {
                "Time (h)": time_h.to_numpy(),
                "Fluid Temperature_Inlet": df[inlet_col].to_numpy(),
                "Fluid Temperatue_Outlet": df[outlet_col].to_numpy(),
            }
        ).dropna()
        fluid_path = out_dir / f"{stem}_fluid.xlsx"
        fluid_df.to_excel(fluid_path, index=False)

    print(f"Rows: {len(sensor_df)}  |  duration: {time_h.iloc[-1]:.3f} h")
    print(f"Excluded sensors: {', '.join(exclude) or '(none)'}")
    print(f"Sensor replay CSV -> {sensor_path}")
    if fluid_path is not None:
        print(f"Fluid XLSX        -> {fluid_path}")
    else:
        print(f"Fluid columns {inlet_col!r}/{outlet_col!r} not found — no fluid file written.")
    return sensor_path, fluid_path


def main() -> None:
    p = argparse.ArgumentParser(description="Prep a raw CR1000 CSV for replay_csv.")
    p.add_argument("--path", required=True, help="Raw CR1000 CSV.")
    p.add_argument("--out-dir", default=None, help="Output dir (default: alongside source).")
    p.add_argument("--timestamp-col", default="TIMESTAMP")
    p.add_argument("--inlet-col", default="Inlet")
    p.add_argument("--outlet-col", default="Outlet")
    p.add_argument(
        "--exclude",
        nargs="*",
        default=DEFAULT_EXCLUDE,
        help="Sensor columns to drop (default: broken H10D75).",
    )
    args = p.parse_args()
    convert(
        Path(args.path),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        timestamp_col=args.timestamp_col,
        inlet_col=args.inlet_col,
        outlet_col=args.outlet_col,
        exclude=args.exclude,
    )


if __name__ == "__main__":
    main()
