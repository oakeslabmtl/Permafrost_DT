#!/usr/bin/env python3
"""
Thermal energy and thaw efficiency metrics — Heat Exchanger experiment.

Saves four PNG images to artifacts/:
  1. ein_vs_vthaw.png   — E_in vs V_thaw
  2. specific_energy.png — E_r vs time
  3. vthaw_vs_time.png  — thaw volume vs time
  4. eta_melt.png       — melting efficiency vs time

Usage
-----
    python energy_thaw_metrics.py
    python energy_thaw_metrics.py --tmax 10
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay
import plotly.graph_objects as go

# ── Physical constants ────────────────────────────────────────────────────────
RHO_FLUID = 1110.0
CP_FLUID  = 2400.0
V_DOT     = 0.5 / 1000 / 60   # m³/s  (0.5 L/min)
RHO_ICE   = 917.0
ICE_FRAC  = 0.39
LF        = 334_000.0          # J/kg

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
DATA_DIR  = HERE.parent.parent / "data"
OUT_DIR   = HERE.parent.parent / "artifacts"
XLSX      = DATA_DIR / "Test_Heat Exchanger_.xlsx"

# ── Style ─────────────────────────────────────────────────────────────────────
BG      = "#05101e"
GRID    = "#1a2c3e"
TICK    = "#7a98b0"
COLORS  = ["#38bdf8", "#fb923c", "#4ade80", "#a78bfa"]

LAYOUT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color="#cbd5e1", family="DejaVu Sans", size=13),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=TICK)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=TICK)),
    margin=dict(l=70, r=30, t=60, b=60),
    width=900, height=560,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sheets(tmin: float, tmax: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    xl      = pd.ExcelFile(XLSX)
    df_io   = xl.parse("Inlet_Outlet_Temp_Fluid")
    df_soil = xl.parse("Temp_soil")
    df_io   = df_io[(df_io["Time (h)"] >= tmin) & (df_io["Time (h)"] <= tmax)].reset_index(drop=True)
    df_soil = df_soil[(df_soil["Time (h)"] >= tmin) & (df_soil["Time (h)"] <= tmax)].reset_index(drop=True)
    return df_io, df_soil


# ── E_in ──────────────────────────────────────────────────────────────────────

def compute_ein(df_io: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t_s  = df_io["Time (h)"].to_numpy() * 3600.0
    Tin  = df_io["Fluid Temperature_Inlet"].to_numpy()
    Tout = df_io["Fluid Temperatue_Outlet"].to_numpy()
    Q    = np.clip(RHO_FLUID * V_DOT * CP_FLUID * (Tin - Tout), 0.0, None)
    Ein  = np.concatenate([[0.0], cumulative_trapezoid(Q, t_s)])
    return df_io["Time (h)"].to_numpy(), Ein


# ── V_thaw ────────────────────────────────────────────────────────────────────
# Uses the same 2-D griddata approach as conference_viz.py:
# all sensors are jointly interpolated onto a fine (r, z) grid each timestep,
# then the 0 °C isotherm is extracted per depth row.  This avoids the per-sensor
# noise that causes apparent thaw-radius jumps when individual readings straddle 0.

def _collect_sensors(df_soil: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return sensor positions (n_s, 2) in metres [r, z] and data (n_t, n_s)."""
    pat  = re.compile(r"^H(\d+)D(\d+)$")
    pts, cols = [], []
    for col in df_soil.columns:
        m = pat.match(str(col))
        if m:
            pts.append([int(m.group(2)) / 1000.0, int(m.group(1)) / 1000.0])  # [r, z]
            cols.append(col)
    return np.array(pts), df_soil[cols].to_numpy()


def _thaw_radius_row(T_row: np.ndarray, r_ax: np.ndarray) -> float:
    """Thaw radius [m] from an interpolated temperature row (ascending r)."""
    if T_row.max() <= 0.0:
        return 0.0
    if T_row.min() > 0.0:
        return float(r_ax[-1])
    crossings = np.where(np.diff(np.sign(T_row)) < 0)[0]
    if not crossings.size:
        return 0.0
    ci   = int(crossings[-1])
    dT   = T_row[ci + 1] - T_row[ci]
    frac = (0.0 - T_row[ci]) / dT if dT else 0.0
    return max(0.0, r_ax[ci] + frac * (r_ax[ci + 1] - r_ax[ci]))


def compute_vthaw(df_soil: pd.DataFrame, _unused=None) -> tuple[np.ndarray, np.ndarray]:
    t_h      = df_soil["Time (h)"].to_numpy()
    pts, T_all = _collect_sensors(df_soil)

    # Interpolation grid — same domain as conference_viz
    r_ax = np.linspace(0.0,  pts[:, 0].max(), 80)   # radial axis
    z_ax = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 100)  # depth axis
    rg, zg   = np.meshgrid(r_ax, z_ax)
    query    = np.column_stack([rg.ravel(), zg.ravel()])

    tri = Delaunay(pts)   # triangulation pre-computed once, reused every timestep

    V_thaw = np.zeros(len(t_h))
    for ti, vals in enumerate(T_all):
        valid = ~np.isnan(vals)
        if valid.sum() < 3:
            continue

        # Linear interp on fixed triangulation, nearest-neighbour fill for hull edges
        T_grid = LinearNDInterpolator(tri, vals)(query).reshape(len(z_ax), len(r_ax))
        nan_m  = np.isnan(T_grid.ravel())
        if nan_m.any():
            nn           = NearestNDInterpolator(pts[valid], vals[valid])
            flat         = T_grid.ravel()
            flat[nan_m]  = nn(query[nan_m])
            T_grid       = flat.reshape(len(z_ax), len(r_ax))

        r_thaw = np.array([_thaw_radius_row(T_grid[zi], r_ax) for zi in range(len(z_ax))])
        V_thaw[ti] = trapezoid(np.pi * r_thaw ** 2, z_ax)

    return t_h, V_thaw


# ── Align ─────────────────────────────────────────────────────────────────────

def align(t_io, Ein, t_soil, V_thaw):
    lo   = max(t_io.min(), t_soil.min())
    hi   = min(t_io.max(), t_soil.max())
    mask = (t_io >= lo) & (t_io <= hi)
    t    = t_io[mask]
    return t, Ein[mask], np.interp(t, t_soil, V_thaw)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _save(fig: go.Figure, path: Path, title: str) -> None:
    fig.write_image(str(path), scale=2)
    print(f"  Saved  {path.name}")


def plot_and_save(t, Ein, V_thaw, out_dir: Path, er_tmin: float = 0.0) -> None:
    out_dir.mkdir(exist_ok=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        Er  = np.where(V_thaw > 1e-6, Ein / V_thaw / 1e6, np.nan)
        eta = np.where(Ein > 0, RHO_ICE * ICE_FRAC * V_thaw * LF / Ein * 100, np.nan)

    # clip specific energy to skip early outliers
    er_mask = t >= er_tmin

    def base_fig(title, xlab, ylab, color, x, y) -> go.Figure:
        fig = go.Figure(go.Scatter(x=x, y=y, mode="lines",
                                   line=dict(color=color, width=2)))
        fig.update_layout(**LAYOUT_BASE,
                          title=dict(text=title, font=dict(size=15, color="white")),
                          xaxis_title=xlab, yaxis_title=ylab)
        return fig

    plots = [
        ("E_in vs V_thaw",
         "Thaw volume V_thaw (L)", "Cumulative energy E_in (MJ)",
         COLORS[0], V_thaw * 1000, Ein / 1e6, "ein_vs_vthaw.png"),

        ("Specific energy consumption",
         "Time (h)", "Specific energy E_r (MJ/m³)",
         COLORS[1], t[er_mask], Er[er_mask], "specific_energy.png"),

        ("Thaw front growth",
         "Time (h)", "Thaw volume V_thaw (L)",
         COLORS[2], t, V_thaw * 1000, "vthaw_vs_time.png"),

        ("Melting efficiency",
         "Time (h)", "η_melt (%)",
         COLORS[3], t, eta, "eta_melt.png"),
    ]

    for title, xlab, ylab, color, x, y, fname in plots:
        fig = base_fig(title, xlab, ylab, color, x, y)
        _save(fig, out_dir / fname, title)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(tmin: float, tmax: float, tmin_er: float = 0.0) -> None:
    print(f"Loading {XLSX.name}  (t = {tmin:.1f}–{tmax:.1f} h) ...")
    df_io, df_soil = load_sheets(tmin, tmax)

    t_io,   Ein    = compute_ein(df_io)
    t_soil, V_thaw = compute_vthaw(df_soil)

    print(f"  Max V_thaw = {V_thaw.max()*1000:.2f} L  at t = {t_soil[V_thaw.argmax()]:.2f} h")
    print(f"  Max E_in   = {Ein.max()/1e6:.3f} MJ")

    t, Ein, V_thaw = align(t_io, Ein, t_soil, V_thaw)

    print(f"Saving plots to {OUT_DIR} ...")
    plot_and_save(t, Ein, V_thaw, OUT_DIR, er_tmin=tmin_er)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tmin",    type=float, default=0.0,  metavar="H")
    p.add_argument("--tmax",    type=float, default=4.0,  metavar="H")
    p.add_argument("--er-tmin", type=float, default=0.0,  metavar="H",
                   help="Start time for the specific energy plot (default: 0)")
    args = p.parse_args()
    run(tmin=args.tmin, tmax=args.tmax, tmin_er=args.er_tmin)
