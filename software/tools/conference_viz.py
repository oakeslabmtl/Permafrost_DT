#!/usr/bin/env python3
"""
Conference heat-spreading visualization for the permafrost digital twin.

Shows a mirrored radial cross-section with:
  - Animated temperature field (Jet colormap, same as the viz service)
  - Red 0 °C thaw-front isotherm
  - User-defined safety limit as dashed radial lines
  - Interactive time slider (live window) / smooth video export (--output)

All frames are pre-computed before playback, so the slider and animation
are instantaneous — no computation per frame, no blinking.

Usage
-----
    python conference_viz.py                              # live window + slider
    python conference_viz.py --output heat.mp4           # save video
    python conference_viz.py --limit 80 --fps 24         # custom options (limit in mm)
    python conference_viz.py --tmin 0 --tmax 10          # time window

Options
-------
    --data    PATH  Excel or JSON data file (default: Test_Heat Exchanger__Temp_soil.xlsx)
    --limit   MM    Safety radius in millimetres (default: 100)
    --output  FILE  Save as .mp4 or .gif         (default: show window)
    --fps     N     Frames per second            (default: 30)
    --interp  N     Sub-frames between keyframes (default: 10)
    --nframes N     Keyframes sampled from data  (default: 120)
    --tmin    H     Start time in hours          (default: data start)
    --tmax    H     End   time in hours          (default: data end)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider
from scipy.interpolate import griddata

# ── optional skimage for cleaner isotherm extraction ────────────────────────
try:
    from skimage.measure import find_contours as _ski_contours

    def _extract_isotherm(T: np.ndarray, level: float = 0.0) -> list[np.ndarray]:
        return _ski_contours(T, level=level)

except ImportError:
    def _extract_isotherm(T: np.ndarray, level: float = 0.0) -> list[np.ndarray]:
        """Per-row linear interpolation fallback (no skimage needed)."""
        nz, nr = T.shape
        pts: list[list[float]] = []
        for zi in range(nz):
            row = T[zi]
            sign = np.sign(row - level)
            cross = np.where(np.diff(sign))[0]
            if cross.size:
                ci = int(cross[-1])
                dv = row[ci + 1] - row[ci]
                frac = (level - row[ci]) / dv if dv else 0.0
                pts.append([float(zi), float(ci) + frac])
        return [np.array(pts)] if pts else []


# ── Physical domain ──────────────────────────────────────────────────────────
R_MAX_M  = 0.24
Z_MAX_M  = 0.30
R_PIPE_M = 0.0075
T_THAW_C = 0.0

# ── Interpolation grid ───────────────────────────────────────────────────────
NR = 300
NZ = 375

# ── Visual style ─────────────────────────────────────────────────────────────
_BG          = "#05101e"
_GRID_COLOR  = "#1a2c3e"
_TICK_COLOR  = "#7a98b0"
_LIMIT_COLOR = "#ef4444"
_FRONT_COLOR = "#ff2020"
_PIPE_COLOR  = "#fbbf24"
_SLIDER_COLOR = "#38bdf8"


# ────────────────────────────────────────────────────────────────────────────
# Data loading  (xlsx or json)
# ────────────────────────────────────────────────────────────────────────────

def load_xlsx(path: Path) -> tuple[list[float], list[dict]]:
    import pandas as pd
    df = pd.read_excel(path)
    times: list[float] = df["Time (h)"].tolist()
    pattern = re.compile(r"^H(\d+)D(\d+)$")
    sensors: list[dict] = []
    for col in df.columns:
        m = pattern.match(str(col))
        if not m:
            continue
        sensors.append({
            "x": int(m.group(2)) / 1000.0,
            "z": int(m.group(1)) / 1000.0,
            "T": df[col].tolist(),
        })
    return times, sensors


def load_json(path: Path) -> tuple[list[float], list[dict]]:
    with open(path) as f:
        raw = json.load(f)
    times: list[float] = list(raw.pop("times_hours"))
    sensors = [
        {"x": float(v["x_m"]), "z": float(v["z_m"]), "T": list(v["temperature_series"])}
        for v in raw.values()
    ]
    return times, sensors


def load_data(path: Path) -> tuple[list[float], list[dict]]:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return load_xlsx(path)
    return load_json(path)


def subsample(
    times: list[float],
    sensors: list[dict],
    tmin: float | None,
    tmax: float | None,
    nframes: int,
) -> tuple[list[float], list[dict]]:
    t = np.array(times)
    lo = tmin if tmin is not None else float(t.min())
    hi = tmax if tmax is not None else float(t.max())
    mask = (t >= lo) & (t <= hi)
    idx_all = np.where(mask)[0]
    if len(idx_all) == 0:
        raise ValueError(f"No data in time window [{lo}, {hi}] h")
    pick = np.linspace(0, len(idx_all) - 1, min(nframes, len(idx_all)), dtype=int)
    idx  = idx_all[pick]
    t_sub = [times[i] for i in idx]
    s_sub = [{"x": s["x"], "z": s["z"], "T": [s["T"][i] for i in idx]}
             for s in sensors]
    return t_sub, s_sub


# ────────────────────────────────────────────────────────────────────────────
# Grid & interpolation
# ────────────────────────────────────────────────────────────────────────────

def make_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = np.linspace(0.0, R_MAX_M, NR)
    z = np.linspace(0.0, Z_MAX_M, NZ)
    rg, zg = np.meshgrid(r, z)
    return r, z, rg, zg


def interp_field(
    sensors: list[dict], t_idx: int, rg: np.ndarray, zg: np.ndarray
) -> np.ndarray:
    pts  = np.array([[s["x"], s["z"]] for s in sensors])
    vals = np.array([s["T"][t_idx]    for s in sensors])
    q    = np.column_stack([rg.ravel(), zg.ravel()])
    T    = griddata(pts, vals, q, method="cubic")
    bad  = np.isnan(T)
    if bad.any():
        T[bad] = griddata(pts, vals, q[bad], method="nearest")
    return T.reshape(rg.shape)


def mirror(T: np.ndarray) -> np.ndarray:
    """(NZ, NR) → (NZ, 2·NR) symmetric about r = 0."""
    return np.concatenate([T[:, ::-1], T], axis=1)


# ────────────────────────────────────────────────────────────────────────────
# Thaw-front extraction  (coordinates returned in mm)
# ────────────────────────────────────────────────────────────────────────────

def front_coords_mm(
    T_half: np.ndarray, r_axis: np.ndarray, z_axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_mm, z_mm) of the closed mirrored 0 °C isotherm."""
    contours = _extract_isotherm(T_half, level=T_THAW_C)
    if not contours:
        return np.empty(0), np.empty(0)
    c = max(contours, key=len)
    r_c = np.interp(c[:, 1], np.arange(NR), r_axis)
    z_c = np.interp(c[:, 0], np.arange(NZ), z_axis)
    order = np.argsort(z_c)
    r_s, z_s = r_c[order], z_c[order]
    x_full = np.concatenate([ r_s,  -r_s[::-1], [r_s[0]] ])
    z_full = np.concatenate([ z_s,   z_s[::-1], [z_s[0]] ])
    return x_full * 1000.0, z_full * 1000.0   # metres → mm


# ────────────────────────────────────────────────────────────────────────────
# Pre-computation
# ────────────────────────────────────────────────────────────────────────────

def precompute(
    sensors: list[dict],
    times: list[float],
    r_axis: np.ndarray,
    z_axis: np.ndarray,
    rg: np.ndarray,
    zg: np.ndarray,
    n_sub: int,
) -> list[dict]:
    n = len(times)
    print("  Interpolating keyframes ...", flush=True)
    grids = [interp_field(sensors, i, rg, zg) for i in range(n)]

    frames: list[dict] = []
    print("  Generating sub-frames   ...", flush=True)
    for i in range(n - 1):
        for k in range(n_sub):
            alpha = k / n_sub
            T = (1.0 - alpha) * grids[i] + alpha * grids[i + 1]
            t = (1.0 - alpha) * times[i] + alpha * times[i + 1]
            xf, zf = front_coords_mm(T, r_axis, z_axis)
            frames.append({"t": t, "img": mirror(T), "xf": xf, "zf": zf})

    T = grids[-1]
    xf, zf = front_coords_mm(T, r_axis, z_axis)
    frames.append({"t": times[-1], "img": mirror(T), "xf": xf, "zf": zf})
    return frames


# ────────────────────────────────────────────────────────────────────────────
# Figure construction
# ────────────────────────────────────────────────────────────────────────────

def build_figure(limit_mm: float, vmin: float = -8.0, vmax: float = 30.0):
    plt.rcParams.update({
        "figure.facecolor": _BG,
        "axes.facecolor":   _BG,
        "text.color":       "white",
        "axes.labelcolor":  "#cbd5e1",
        "xtick.color":      _TICK_COLOR,
        "ytick.color":      _TICK_COLOR,
        "font.family":      "DejaVu Sans",
        "font.size":        11,
    })

    fig = plt.figure(figsize=(9, 9), dpi=150)   # 10×10 → 9×9 (−10 %)
    fig.patch.set_facecolor(_BG)

    # ── Main axes + colorbar (top raised since no title) ──────────────────
    gs = fig.add_gridspec(
        1, 2,
        left=0.09, right=0.94, bottom=0.14, top=0.97,
        width_ratios=[20, 1], wspace=0.025,
    )
    ax  = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    for sp in ax.spines.values():
        sp.set_color(_GRID_COLOR)
    cax.set_facecolor(_BG)
    for sp in cax.spines.values():
        sp.set_color(_GRID_COLOR)

    R_MAX_MM = R_MAX_M * 1000
    Z_MAX_MM = Z_MAX_M * 1000

    ax.set_xlim(-R_MAX_MM, R_MAX_MM)
    ax.set_ylim(Z_MAX_MM, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("Radial distance from pipe centre  (mm)", fontsize=11, labelpad=6)
    ax.set_ylabel("Depth  (mm)", fontsize=11, labelpad=6)
    ax.tick_params(labelsize=10, color=_TICK_COLOR, length=4)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.65)

    # ── Temperature field ─────────────────────────────────────────────────
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(
        np.zeros((NZ, 2 * NR)),
        extent=[-R_MAX_MM, R_MAX_MM, Z_MAX_MM, 0],
        origin="upper",
        aspect="auto",
        cmap="jet",
        norm=norm,
        interpolation="bilinear",
        zorder=1,
    )

    # ── Colorbar ──────────────────────────────────────────────────────────
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Temperature  (°C)", fontsize=10, color="#cbd5e1", labelpad=5)
    _ticks = sorted(set([
        round(vmin, 1), round(vmin / 2, 1), 0.0,
        round(vmax / 4, 1), round(vmax / 2, 1), round(vmax * 3 / 4, 1), round(vmax, 1),
    ]))
    _ticks = [t for t in _ticks if vmin <= t <= vmax]
    cb.set_ticks(_ticks)
    _labels = [f"{t:g}" for t in _ticks]
    _labels[-1] += " °C"
    tick_labels = cb.ax.set_yticklabels(_labels, fontsize=8.5, color=_TICK_COLOR)
    zero_idx = _ticks.index(0.0)
    tick_labels[zero_idx].set_color(_FRONT_COLOR)
    tick_labels[zero_idx].set_fontweight("bold")
    tick_labels[zero_idx].set_fontsize(9)

    # ── Thaw front ────────────────────────────────────────────────────────
    (front_line,) = ax.plot(
        [], [], linewidth=1.2, color=_FRONT_COLOR, zorder=4,
        solid_capstyle="round", solid_joinstyle="round",
    )

    # ── Safety limit — dashed lines only, no annotation box ──────────────
    for sign in (+1, -1):
        ax.axvline(sign * limit_mm, color=_LIMIT_COLOR, linewidth=1.8,
                   linestyle="--", alpha=0.90, zorder=3)
    ax.axvspan( limit_mm, R_MAX_MM, color=_LIMIT_COLOR, alpha=0.05, zorder=0)
    ax.axvspan(-R_MAX_MM, -limit_mm, color=_LIMIT_COLOR, alpha=0.05, zorder=0)

    # ── Heating pipe — rectangle only, no text label inside ───────────────
    pw = R_PIPE_M * 1000   # mm
    ax.add_patch(Rectangle(
        (-pw, 0), 2 * pw, Z_MAX_MM,
        lw=1.5, edgecolor=_PIPE_COLOR, facecolor="#0d1b2a", zorder=5,
    ))

    # ── Legend (compact) ─────────────────────────────────────────────────
    ax.legend(
        handles=[
            Line2D([0], [0], color=_FRONT_COLOR, linewidth=1.4,
                   label="0 °C thaw front"),
            Line2D([0], [0], color=_LIMIT_COLOR, linewidth=1.8, linestyle="--",
                   label=f"Safety limit  ({limit_mm:.0f} mm)"),
            Line2D([0], [0], color=_PIPE_COLOR, linewidth=3,
                   label="Heating pipe"),
        ],
        loc="lower right", fontsize=6.5,
        facecolor="#0b1828", edgecolor=_GRID_COLOR,
        labelcolor="white", framealpha=0.90, borderpad=0.3,
        handlelength=1.0, handletextpad=0.3, labelspacing=0.2,
    )

    # ── Time label ────────────────────────────────────────────────────────
    time_text = ax.text(
        0.013, 0.024, "",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=7.7, fontweight="bold", color="white", zorder=10,
        bbox=dict(boxstyle="round,pad=0.2", fc=_BG, ec=_GRID_COLOR, lw=0.7, alpha=0.90),
    )

    return fig, ax, im, front_line, time_text


# ────────────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────────────

def run(
    data_path: Path,
    limit_mm: float,
    output: Path | None,
    fps: int,
    n_sub: int,
    nframes: int,
    tmin: float | None,
    tmax: float | None,
) -> None:
    print(f"Loading  {data_path} ...")
    times, sensors = load_data(data_path)
    print(f"  {len(sensors)} sensors,  {len(times)} raw timesteps "
          f"({times[0]:.2f} to {times[-1]:.2f} h)")

    if tmax is None:
        tmax = 4.0   # default cap: show first 4 hours
    times, sensors = subsample(times, sensors, tmin, tmax, nframes)
    print(f"  Subsampled to {len(times)} keyframes "
          f"({times[0]:.2f} to {times[-1]:.2f} h)")

    r_axis, z_axis, rg, zg = make_grid()

    all_temps = [v for s in sensors for v in s["T"] if v == v]
    vmin = min(float(np.floor(min(all_temps))), -1.0)
    vmax = max(float(np.ceil(max(all_temps))),   1.0)
    print(f"  Temperature range: [{vmin:.1f}, {vmax:.1f}] deg C")

    print("Pre-computing frames ...")
    frames = precompute(sensors, times, r_axis, z_axis, rg, zg, n_sub)
    print(f"  {len(frames)} frames ready")

    fig, ax, im, front_line, time_text = build_figure(limit_mm, vmin=vmin, vmax=vmax)
    t_array = np.array([f["t"] for f in frames])

    def draw_frame(fi: int) -> None:
        f = frames[fi]
        im.set_data(f["img"])
        front_line.set_data(f["xf"], f["zf"])
        time_text.set_text(f"t = {f['t']:.2f} h")

    draw_frame(0)

    if output is not None:
        def _update(fi: int):
            draw_frame(fi)
            return im, front_line, time_text

        def _init():
            draw_frame(0)
            return im, front_line, time_text

        ani = anim.FuncAnimation(
            fig, _update, frames=len(frames),
            init_func=_init, interval=1000 // fps,
            blit=True, repeat=True, repeat_delay=800,
        )
        suffix = output.suffix.lower()
        writer = (anim.PillowWriter(fps=fps) if suffix == ".gif" else
                  anim.FFMpegWriter(fps=fps, bitrate=5000,
                                    extra_args=["-vcodec", "libx264",
                                                "-pix_fmt", "yuv420p",
                                                "-preset", "slow", "-crf", "18"]))
        print(f"Saving to {output} ...")
        ani.save(str(output), writer=writer, dpi=150,
                 savefig_kwargs={"facecolor": fig.get_facecolor()})
        print("Done.")

    else:
        t0, t1 = float(t_array[0]), float(t_array[-1])
        current = [0]

        def draw_frame_tracked(fi: int) -> None:
            current[0] = fi
            draw_frame(fi)

        # ── Layout constants (bottom-left compact control strip) ──────────
        # All in figure-fraction coords. Tweak BTN_L / SL_L / SL_W to reposition.
        BTN_L, BTN_B, BTN_S = 0.01,  0.045, 0.048   # button: left, bottom, size (square)
        SL_L,  SL_B,  SL_W  = 0.060, 0.052, 0.280   # slider: left, bottom, width
        LBL_B                = 0.028                  # y of time labels below slider

        # ── Round play/pause button drawn with a matplotlib patch ─────────
        ax_btn = fig.add_axes(
            [BTN_L, BTN_B, BTN_S, BTN_S * fig.get_size_inches()[0] /
             fig.get_size_inches()[1]],
            facecolor="none",
        )
        ax_btn.set_xlim(-1, 1)
        ax_btn.set_ylim(-1, 1)
        ax_btn.set_aspect("equal")
        ax_btn.axis("off")

        import matplotlib.patches as mpatches
        # Circle background
        circle = mpatches.Circle((0, 0), 0.88, color=_SLIDER_COLOR,
                                  zorder=2, transform=ax_btn.transData)
        ax_btn.add_patch(circle)
        # Play triangle (Unicode ▶ drawn as text)
        icon_text = ax_btn.text(
            0.08, 0, "▶", ha="center", va="center",
            fontsize=13, color=_BG, fontweight="bold", zorder=3,
        )

        # Invisible Button widget layered on top for click detection
        ax_btn_hit = fig.add_axes(
            [BTN_L, BTN_B, BTN_S, BTN_S * fig.get_size_inches()[0] /
             fig.get_size_inches()[1]],
            facecolor="none",
        )
        ax_btn_hit.axis("off")
        btn_play = Button(ax_btn_hit, "", color="none", hovercolor="none")

        # ── Compact slider (bottom-left, next to button) ──────────────────
        ax_sl = fig.add_axes([SL_L, SL_B, SL_W, 0.022], facecolor="#0b1828")
        for sp in ax_sl.spines.values():
            sp.set_color(_GRID_COLOR)

        slider = Slider(
            ax_sl, label="",
            valmin=t0, valmax=t1, valinit=t0,
            color=_SLIDER_COLOR, initcolor="none",
        )
        slider.label.set_visible(False)
        slider.valtext.set_visible(False)

        for t in times:
            ax_sl.axvline(t, color=_GRID_COLOR, linewidth=0.5, alpha=0.6)

        # ── Labels below the slider ───────────────────────────────────────
        fig.text(SL_L, LBL_B, f"{t0:.1f} h",
                 color=_TICK_COLOR, fontsize=7.5, ha="left",  va="center")
        fig.text(SL_L + SL_W, LBL_B, f"{t1:.1f} h",
                 color=_TICK_COLOR, fontsize=7.5, ha="right", va="center")
        _time_label = fig.text(
            SL_L + SL_W / 2, LBL_B, f"t = {t0:.2f} h",
            color="white", fontsize=8, fontweight="bold",
            ha="center", va="center",
        )

        def on_change(val: float) -> None:
            fi = int(np.argmin(np.abs(t_array - val)))
            draw_frame_tracked(fi)
            _time_label.set_text(f"t = {frames[fi]['t']:.2f} h")
            fig.canvas.draw_idle()

        slider.on_changed(on_change)

        playing = [False]
        timer = fig.canvas.new_timer(interval=250)   # 2 keyframe-steps per second

        def _tick(_=None) -> None:
            next_fi = (current[0] + 2) % len(frames)
            slider.eventson = False
            slider.set_val(t_array[next_fi])
            slider.eventson = True
            draw_frame_tracked(next_fi)
            _time_label.set_text(f"t = {frames[next_fi]['t']:.2f} h")
            fig.canvas.draw_idle()

        timer.add_callback(_tick)

        def on_play_pause(_event) -> None:
            if playing[0]:
                playing[0] = False
                timer.stop()
                icon_text.set_text("▶")
                circle.set_color(_SLIDER_COLOR)
            else:
                playing[0] = True
                timer.start()
                icon_text.set_text("⏸")
                circle.set_color("#22d3ee")
            fig.canvas.draw_idle()

        btn_play.on_clicked(on_play_pause)

        plt.show()


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_data = here.parent.parent / "data" / "Test_Heat Exchanger__Temp_soil.xlsx"

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data",    type=Path,  default=default_data, metavar="PATH")
    p.add_argument("--limit",   type=float, default=100.0,        metavar="MM",
                   help="Safety radius in millimetres (default: 100)")
    p.add_argument("--output",  type=Path,  default=None,         metavar="FILE")
    p.add_argument("--fps",     type=int,   default=30)
    p.add_argument("--interp",  type=int,   default=10,           metavar="N")
    p.add_argument("--nframes", type=int,   default=120,          metavar="N")
    p.add_argument("--tmin",    type=float, default=None,         metavar="H")
    p.add_argument("--tmax",    type=float, default=None,         metavar="H",
                   help="End time in hours (default: 4.0)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_path=args.data,
        limit_mm=args.limit,
        output=args.output,
        fps=args.fps,
        n_sub=args.interp,
        nframes=args.nframes,
        tmin=args.tmin,
        tmax=args.tmax,
    )