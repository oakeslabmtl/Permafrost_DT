"""U-shaped pipe (heat-exchanger loop) thermal model, calibration and boundary helper.

This is the clean, importable core extracted from the ``Optimization1.ipynb`` /
``flow temperature_optimization.py`` calibration study. It contains no pandas,
matplotlib or file IO so it can be used both offline and inside a live service.

Physics — a 1D finite-volume model of the fluid travelling through the U-tube,
coupled to an effective wall/sample temperature:

    Fluid:  dTf_i/dt = a (Tf_{i-1} - Tf_i) + kf (Tw_i - Tf_i)      a = v/ds
    Wall :  dTw_i/dt = kw (Tf_i - Tw_i)   - kloss (Tw_i - Tenv)

The measured inlet temperature drives the first cell; the last cell is the
simulated outlet. Six effective parameters are inverted against the measured
outlet temperature:

    v      effective fluid velocity            [m/s]
    kf     fluid <-> wall exchange rate        [1/s]
    kw     wall thermal response rate          [1/s]
    kloss  wall -> environment loss rate       [1/s]
    Tw0    initial wall/sample temperature     [C]
    Tenv   background/environment temperature  [C]

``calibrate_pipe`` fits these with a bounded L-BFGS-B search (logit-encoded
parameters + gradient descent, warm-startable). ``simulate_pipe`` runs the
forward model. ``effective_interface_temperature`` reduces the simulated pipe
state to a single boundary temperature the soil FEM can consume as a Robin BC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# Parameter order used everywhere in this module.
PARAM_NAMES: tuple[str, ...] = ("v", "kf", "kw", "kloss", "Tw0", "Tenv")

# Broad physical search bounds (not p0 +/- x%); copied from the validated study.
DEFAULT_LOWER: dict[str, float] = {
    "v": 1e-5,
    "kf": 1e-5,
    "kw": 1e-7,
    "kloss": 0.0,
    "Tw0": -10.0,
    "Tenv": -10.0,
}
DEFAULT_UPPER: dict[str, float] = {
    "v": 0.05,
    "kf": 0.05,
    "kw": 0.01,
    "kloss": 0.005,
    "Tw0": 40.0,
    "Tenv": 40.0,
}
# Initial guess for the rate parameters (Tw0/Tenv are seeded from the data).
DEFAULT_P0: dict[str, float] = {
    "v": 0.002,
    "kf": 0.004,
    "kw": 0.0002,
    "kloss": 0.00002,
    "Tw0": 20.0,
    "Tenv": 20.0,
}


@dataclass(frozen=True)
class PipeModelConfig:
    """Geometry / discretisation of the U-shaped pipe."""

    h_sample_m: float = 0.30
    r_bend_m: float = 0.0
    n_cells: int = 120
    max_dt_s: float = 5.0
    # Upper bound on time steps per simulation. The scheme is implicit upwind +
    # backward Euler (unconditionally stable), so the advective CFL step is an
    # accuracy choice, not a stability requirement. At high fluid velocity the
    # CFL step becomes tiny and the step count explodes; this floors dt so a
    # sim never exceeds max_steps (transit is then sub-step / quasi-steady,
    # which is physically correct when transport is fast vs the forcing).
    max_steps: int = 3000

    @property
    def length_m(self) -> float:
        return 2.0 * self.h_sample_m + np.pi * self.r_bend_m


class StopOptimization(Exception):
    """Raised from the L-BFGS-B callback to early-stop on small loss change."""


# ----------------------------------------------------------------------------
# Parameter encoding (bounded -> unbounded via logit, so L-BFGS-B stays inside
# the physical box without explicit constraints).
# ----------------------------------------------------------------------------

def _as_array(params: Mapping[str, float] | Sequence[float]) -> np.ndarray:
    if isinstance(params, Mapping):
        return np.array([float(params[name]) for name in PARAM_NAMES], dtype=float)
    arr = np.asarray(list(params), dtype=float)
    if arr.shape[0] != len(PARAM_NAMES):
        raise ValueError(f"Expected {len(PARAM_NAMES)} parameters, got {arr.shape[0]}.")
    return arr


def _to_dict(arr: np.ndarray) -> dict[str, float]:
    return {name: float(arr[idx]) for idx, name in enumerate(PARAM_NAMES)}


def _bounds_arrays(
    lower: Mapping[str, float] | None,
    upper: Mapping[str, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    lo = dict(DEFAULT_LOWER)
    hi = dict(DEFAULT_UPPER)
    if lower:
        lo.update({k: float(v) for k, v in lower.items() if k in lo})
    if upper:
        hi.update({k: float(v) for k, v in upper.items() if k in hi})
    lb = np.array([lo[name] for name in PARAM_NAMES], dtype=float)
    ub = np.array([hi[name] for name in PARAM_NAMES], dtype=float)
    if np.any(lb >= ub):
        raise ValueError("Each lower bound must be strictly below its upper bound.")
    return lb, ub


def normalize_parameters(p: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Map physical parameters to [0, 1] linearly within their bounds.

    Linear (not logit) so ``dp/dx = ub - lb`` is constant: every variable is
    O(1) and equally sensitive to the optimiser's finite-difference step,
    unlike a logit transform which flattens the gradient near the bound edges
    (where the low-valued rate parameters live) and stalls L-BFGS-B.
    """
    return np.clip((p - lb) / (ub - lb), 0.0, 1.0)


def denormalize_parameters(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return lb + np.clip(x, 0.0, 1.0) * (ub - lb)


# ----------------------------------------------------------------------------
# Forward model
# ----------------------------------------------------------------------------

def simulate_pipe(
    params: Mapping[str, float] | Sequence[float],
    times_s: Sequence[float],
    tin_c: Sequence[float],
    config: PipeModelConfig | None = None,
    tf_initial_c: float | None = None,
) -> dict[str, np.ndarray]:
    """Integrate the coupled fluid/wall FVM model over ``times_s``.

    ``times_s`` are seconds (monotonic, starting near zero); ``tin_c`` is the
    measured inlet temperature at those times. Returns the simulated outlet
    series plus the final along-pipe fluid/wall profiles.
    """

    cfg = config or PipeModelConfig()
    p = _as_array(params)
    v, kf, kw, kloss, Tw0, Tenv = (float(x) for x in p)

    times_s = np.asarray(times_s, dtype=float)
    tin_c = np.asarray(tin_c, dtype=float)
    if times_s.ndim != 1 or times_s.shape[0] < 2:
        raise ValueError("times_s must contain at least two samples.")

    n = int(cfg.n_cells)
    length = cfg.length_m
    ds = length / n
    s = np.linspace(ds / 2.0, length - ds / 2.0, n)

    t_end = float(times_s[-1])
    if tf_initial_c is None:
        tf_initial_c = float(tin_c[0])

    # Stability-limited step (advective + reactive), capped.
    dt_adv = 0.5 * ds / max(v, 1e-12)
    dt_react = 0.2 / max(kf + kw + kloss, 1e-12)
    dt = min(cfg.max_dt_s, dt_adv, dt_react)
    # Floor dt so the step count stays bounded at high velocity (stable scheme).
    if t_end > 0.0 and cfg.max_steps > 0:
        dt = max(dt, t_end / float(cfg.max_steps))

    t = np.arange(0.0, t_end + dt, dt)
    if t[-1] > t_end:
        t[-1] = t_end
    nt = len(t)

    Tf = tf_initial_c * np.ones(n)
    Tw = Tw0 * np.ones(n)
    tout_sim = np.empty(nt)
    tout_sim[0] = Tf[-1]

    tin_interp = interp1d(times_s, tin_c, kind="linear", fill_value="extrapolate")

    for step in range(1, nt):
        dt_local = t[step] - t[step - 1]
        tin_now = float(tin_interp(t[step]))
        a = v / ds

        Tf_old = Tf
        Tw_old = Tw
        Tf_new = np.empty(n)

        # First cell: inlet boundary condition.
        Tf_new[0] = (Tf_old[0] + dt_local * (a * tin_now + kf * Tw_old[0])) / (
            1.0 + dt_local * (a + kf)
        )
        # Internal cells (sequential upwind recurrence).
        denom = 1.0 + dt_local * (a + kf)
        for i in range(1, n):
            Tf_new[i] = (Tf_old[i] + dt_local * (a * Tf_new[i - 1] + kf * Tw_old[i])) / denom

        # Wall/sample update (elementwise).
        Tw = (Tw_old + dt_local * (kw * Tf_new + kloss * Tenv)) / (
            1.0 + dt_local * (kw + kloss)
        )
        Tf = Tf_new
        tout_sim[step] = Tf[-1]

    return {
        "t_sim_s": t,
        "tout_sim_c": tout_sim,
        "tf_final_c": Tf,
        "tw_final_c": Tw,
        "s_m": s,
    }


def _outlet_at_data(sim: dict[str, np.ndarray], times_s: Sequence[float]) -> np.ndarray:
    return np.interp(np.asarray(times_s, dtype=float), sim["t_sim_s"], sim["tout_sim_c"])


def evaluate_pipe(
    params: Mapping[str, float] | Sequence[float],
    times_s: Sequence[float],
    tin_c: Sequence[float],
    tout_c: Sequence[float],
    config: PipeModelConfig | None = None,
) -> dict[str, float | np.ndarray]:
    """Run the model and compute fit metrics against measured outlet."""

    tout_c = np.asarray(tout_c, dtype=float)
    sim = simulate_pipe(params, times_s, tin_c, config, tf_initial_c=float(tout_c[0]))
    pred = _outlet_at_data(sim, times_s)
    err = pred - tout_c
    mse = float(np.mean(err**2))
    return {
        "pred_outlet_c": pred,
        "error_c": err,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "max_abs_error": float(np.max(np.abs(err))) if err.size else float("nan"),
    }


# ----------------------------------------------------------------------------
# Calibration (bounded L-BFGS-B with logit encoding + loss-change early stop)
# ----------------------------------------------------------------------------

def calibrate_pipe(
    times_s: Sequence[float],
    tin_c: Sequence[float],
    tout_c: Sequence[float],
    config: PipeModelConfig | None = None,
    *,
    p0: Mapping[str, float] | None = None,
    lower: Mapping[str, float] | None = None,
    upper: Mapping[str, float] | None = None,
    warm_start: Mapping[str, float] | None = None,
    fixed: Mapping[str, float] | None = None,
    maxiter: int = 40,
    maxfun: int = 80,
    loss_tol: float = 1e-4,
    fd_step: float = 1e-2,
) -> dict[str, object]:
    """Invert the effective pipe parameters from measured inlet/outlet.

    ``warm_start`` (e.g. the previous calibration result) is clipped into the
    bounds and used as the starting point so live re-fits converge in a handful
    of iterations.

    ``fixed`` holds named parameters at exactly the given value: they are
    removed from the optimisation entirely (not just tightly bounded), so only
    the remaining free parameters are fitted. Useful for imposing a measured
    quantity (e.g. a known flow rate ``v``) while testing.
    """

    cfg = config or PipeModelConfig()
    times_s = np.asarray(times_s, dtype=float)
    tin_c = np.asarray(tin_c, dtype=float)
    tout_c = np.asarray(tout_c, dtype=float)
    if times_s.shape[0] < 2:
        raise ValueError("Need at least two samples to calibrate.")

    lb, ub = _bounds_arrays(lower, upper)

    # Seed initial guess; default Tw0 from first outlet reading.
    guess = dict(DEFAULT_P0)
    guess["Tw0"] = float(tout_c[0])
    if p0:
        guess.update({k: float(v) for k, v in p0.items() if k in guess})
    start = warm_start if warm_start else guess
    p_start = np.clip(_as_array({**guess, **{k: float(v) for k, v in dict(start).items() if k in guess}}), lb, ub)

    # Parameters held at an exact imposed value are removed from the search.
    fixed = {k: float(v) for k, v in (fixed or {}).items() if k in PARAM_NAMES}
    base = p_start.copy()
    for idx, name in enumerate(PARAM_NAMES):
        if name in fixed:
            base[idx] = fixed[name]  # exact, not routed through the bounds
    free_idx = np.array([i for i, name in enumerate(PARAM_NAMES) if name not in fixed], dtype=int)
    free_lb = lb[free_idx]
    free_ub = ub[free_idx]

    def build_params(x_free: np.ndarray) -> np.ndarray:
        p = base.copy()
        if free_idx.size:
            p[free_idx] = denormalize_parameters(x_free, free_lb, free_ub)
        return p

    x0 = normalize_parameters(p_start[free_idx], free_lb, free_ub) if free_idx.size else np.array([])

    history: list[float] = []
    eval_count = {"n": 0}
    last: dict[str, object] = {"x": None, "loss": None}

    def objective(x: np.ndarray) -> float:
        eval_count["n"] += 1
        loss = float(evaluate_pipe(build_params(x), times_s, tin_c, tout_c, cfg)["mse"])
        last["x"], last["loss"] = np.array(x, dtype=float), loss
        return loss

    def callback(x: np.ndarray) -> None:
        # Reuse the loss already computed at this point to avoid a second sim.
        if last["loss"] is not None and last["x"] is not None and np.allclose(last["x"], x):
            loss = float(last["loss"])  # type: ignore[arg-type]
        else:
            loss = objective(x)
        history.append(loss)
        if len(history) >= 2 and abs(history[-1] - history[-2]) < loss_tol:
            raise StopOptimization

    x_fit = x0
    stopped_early = False
    message = ""
    success = False
    if free_idx.size == 0:
        # All parameters fixed: nothing to optimise, just evaluate.
        message = "all parameters fixed"
        success = True
    else:
        try:
            # ``eps`` is the finite-difference step in the normalised [0, 1] space:
            # a fraction of each parameter's full range, so the numerical gradient
            # is meaningful for every parameter regardless of physical magnitude.
            result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=[(0.0, 1.0)] * free_idx.size,
                callback=callback,
                options={"maxiter": maxiter, "maxfun": maxfun, "ftol": 1e-12, "gtol": 1e-9, "eps": float(fd_step)},
            )
            x_fit = result.x
            success = bool(result.success)
            message = str(result.message)
        except StopOptimization:
            stopped_early = True
            message = "stopped on loss-change tolerance"
            success = True

    p_fit = build_params(x_fit)
    final = evaluate_pipe(p_fit, times_s, tin_c, tout_c, cfg)

    return {
        "parameters": _to_dict(p_fit),
        "rmse": float(final["rmse"]),
        "mae": float(final["mae"]),
        "mse": float(final["mse"]),
        "max_abs_error": float(final["max_abs_error"]),
        "success": success,
        "stopped_early": stopped_early,
        "message": message,
        "eval_count": int(eval_count["n"]),
        "n_samples": int(times_s.shape[0]),
        "travel_time_min": float(cfg.length_m / max(p_fit[0], 1e-12) / 60.0),
        "fixed": dict(fixed),
    }


# ----------------------------------------------------------------------------
# Boundary-condition reduction
# ----------------------------------------------------------------------------

def effective_interface_temperature(
    params: Mapping[str, float] | Sequence[float],
    times_s: Sequence[float],
    tin_c: Sequence[float],
    config: PipeModelConfig | None = None,
    *,
    source: str = "fluid",
    reduction: str = "mean",
    tf_initial_c: float | None = None,
) -> float:
    """Reduce the simulated along-pipe profile to a single interface temperature.

    This is the temperature the soil FEM sees at the pipe boundary. ``source``
    selects the fluid ("fluid", the Robin reference) or effective wall ("wall")
    profile; ``reduction`` is "mean", "outlet"/"min"/"max".
    """

    sim = simulate_pipe(params, times_s, tin_c, config, tf_initial_c=tf_initial_c)
    profile = sim["tf_final_c"] if source == "fluid" else sim["tw_final_c"]
    if reduction == "mean":
        return float(np.mean(profile))
    if reduction == "outlet":
        return float(profile[-1])
    if reduction == "min":
        return float(np.min(profile))
    if reduction == "max":
        return float(np.max(profile))
    raise ValueError(f"Unknown reduction '{reduction}'.")


def interface_temperature_profile(
    params: Mapping[str, float] | Sequence[float],
    times_s: Sequence[float],
    tin_c: Sequence[float],
    config: PipeModelConfig | None = None,
    *,
    depths_m: Sequence[float],
    source: str = "fluid",
    tf_initial_c: float | None = None,
) -> list[tuple[float, float]]:
    """Depth-resolved interface temperature along the pipe axis.

    The U-tube inlet and outlet are both at the surface (depth 0); the bend is
    at the deep end. The pipe coordinate ``s`` therefore maps to depth twice:
    the descending leg (``s = z``) and the ascending leg (``s = L - z``). The
    soil at the axis at depth ``z`` sees both legs, so the interface temperature
    is their average:

        T_axis(z) = 0.5 * (Tf(s=z) + Tf(s=L-z))

    Returns ``[(z_m, temperature_c), ...]`` for each requested depth. This is
    what makes the FEM pipe boundary hot near the surface (inlet) and coolest at
    the deep bend, instead of a single depth-uniform value.
    """

    cfg = config or PipeModelConfig()
    sim = simulate_pipe(params, times_s, tin_c, cfg, tf_initial_c=tf_initial_c)
    values = sim["tf_final_c"] if source == "fluid" else sim["tw_final_c"]
    s = sim["s_m"]
    length = cfg.length_m

    profile: list[tuple[float, float]] = []
    for z in depths_m:
        z = float(z)
        s_down = min(max(z, s[0]), s[-1])
        s_up = min(max(length - z, s[0]), s[-1])
        t_down = float(np.interp(s_down, s, values))
        t_up = float(np.interp(s_up, s, values))
        profile.append((z, 0.5 * (t_down + t_up)))
    return profile


def _self_test() -> None:
    """Synthetic smoke test: calibration must materially improve the fit.

    Uses fast oscillating forcing and strong wall coupling so the effective
    parameters are actually identifiable from the outlet (unlike slowly-forced
    data, where transport/exchange are invisible and the loss is flat).
    """

    cfg = PipeModelConfig(n_cells=25)
    rng = np.random.default_rng(0)
    # 20-minute window with a 5-minute inlet oscillation so transport/exchange
    # are exercised and the effective parameters are identifiable.
    times_s = np.linspace(0.0, 20.0 * 60.0, 120)
    tin = 25.0 + 10.0 * np.sin(2.0 * np.pi * times_s / 300.0)

    truth = {"v": 0.02, "kf": 0.01, "kw": 0.002, "kloss": 0.001, "Tw0": 15.0, "Tenv": 22.0}
    sim = simulate_pipe(truth, times_s, tin, cfg, tf_initial_c=15.0)
    tout = _outlet_at_data(sim, times_s) + rng.normal(0, 0.02, times_s.size)

    # Starting-guess fit quality, for a before/after comparison.
    guess = {**DEFAULT_P0, "Tw0": float(tout[0])}
    rmse_initial = float(evaluate_pipe(guess, times_s, tin, tout, cfg)["rmse"])

    result = calibrate_pipe(times_s, tin, tout, cfg, maxiter=60, maxfun=150)
    eff = effective_interface_temperature(result["parameters"], times_s, tin, cfg)

    print("initial rmse:", round(rmse_initial, 4))
    print("calibration:", {k: result[k] for k in ("rmse", "mae", "success", "eval_count", "travel_time_min")})
    print("parameters :", {k: round(v, 6) for k, v in result["parameters"].items()})
    print("effective interface temperature (C):", round(eff, 4))

    assert np.isfinite(result["rmse"]), "non-finite RMSE"
    assert np.isfinite(eff), "non-finite effective interface temperature"
    assert result["rmse"] < rmse_initial * 0.75, (
        f"calibration did not improve fit ({result['rmse']:.4f} vs initial {rmse_initial:.4f})"
    )
    print("SELF-TEST OK")


if __name__ == "__main__":
    _self_test()
