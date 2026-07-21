import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d


# ============================================================
# FVM simulation of U-shaped heat exchanger
# Using measured inlet temperature as boundary condition
# Calibrating outlet temperature response
# ============================================================


# ------------------------------------------------------------
# 1. Read experimental data
# ------------------------------------------------------------

filename = "heat_data.txt"

data = []

with open(filename, "r") as file:
    lines = file.readlines()

for line in lines:
    parts = line.strip().split()

    if len(parts) >= 3:
        try:
            values = [float(parts[0]), float(parts[1]), float(parts[2])]
            data.append(values)
        except ValueError:
            pass

data = np.array(data)

time_h = data[:, 0]       # time [h]
Tin_exp = data[:, 1]      # inlet temperature [C]
Tout_exp = data[:, 2]     # measured outlet temperature [C]

time_s = time_h * 3600.0  # convert time to seconds


# ------------------------------------------------------------
# Optional: remove NaN values and average repeated time values
# ------------------------------------------------------------

# valid = np.isfinite(time_s) & np.isfinite(Tin_exp) & np.isfinite(Tout_exp)
# time_s = time_s[valid]
# Tin_exp = Tin_exp[valid]
# Tout_exp = Tout_exp[valid]
#
# unique_times = np.unique(time_s)
# Tin_unique = np.zeros_like(unique_times)
# Tout_unique = np.zeros_like(unique_times)
#
# for i, t in enumerate(unique_times):
#     mask = time_s == t
#     Tin_unique[i] = np.mean(Tin_exp[mask])
#     Tout_unique[i] = np.mean(Tout_exp[mask])
#
# time_s = unique_times
# Tin_exp = Tin_unique
# Tout_exp = Tout_unique


# ------------------------------------------------------------
# 2. Geometry of U-shaped pipe
# ------------------------------------------------------------

H_sample = 0.30      # sample height [m]
Rbend = 0.00         # bend radius [m], use actual value if known

L_pipe = 2.0 * H_sample + np.pi * Rbend

N = 120
ds = L_pipe / N

print(f"Pipe length = {L_pipe:.3f} m")
print(f"Number of FVM cells = {N}")
print(f"Cell size = {ds:.5f} m")


# ------------------------------------------------------------
# 3. Calibrated effective model parameters
# ------------------------------------------------------------
# p[0] = v       : effective fluid velocity [m/s]
# p[1] = kf      : fluid-wall heat exchange coefficient [1/s]
# p[2] = kw      : wall thermal response coefficient [1/s]
# p[3] = kloss   : wall heat-loss coefficient to environment [1/s]
# p[4] = Tw0     : initial wall/sample temperature [C]
# p[5] = Tenv    : external/background temperature [C]

p0 = np.array([
    0.002,        # v [m/s]
    0.004,        # kf [1/s]
    0.0002,       # kw [1/s]
    0.00002,      # kloss [1/s]
    Tout_exp[0],  # initial wall/sample temperature [C]
    20.0          # environmental temperature [C]
])

lb = np.array([
    1e-5,    # minimum v
    1e-5,    # minimum kf
    1e-7,    # minimum kw
    0.0,     # minimum kloss
    -10.0,   # minimum Tw0
    -10.0    # minimum Tenv
])

ub = np.array([
    0.05,    # maximum v
    0.05,    # maximum kf
    0.01,    # maximum kw
    0.005,   # maximum kloss
    40.0,    # maximum Tw0
    40.0     # maximum Tenv
])


# ------------------------------------------------------------
# Local functions
# ------------------------------------------------------------

def encode_parameters(p, lb, ub):
    """
    Convert bounded physical parameters p into unbounded variables theta.
    This is equivalent to the MATLAB encodeParameters function.
    """

    r = (p - lb) / (ub - lb)
    r = np.clip(r, 1e-8, 1.0 - 1e-8)

    theta = np.log(r / (1.0 - r))

    return theta


def decode_parameters(theta, lb, ub):
    """
    Convert unbounded variables theta back to bounded physical parameters p.
    This is equivalent to the MATLAB decodeParameters function.
    """

    r = 1.0 / (1.0 + np.exp(-theta))
    p = lb + (ub - lb) * r

    return p


def simulate_fvm(p, time_data, Tin_data, L_pipe, N, Tf_initial):
    """
    FVM simulation of transient fluid temperature in U-shaped pipe.

    Fluid equation:
        dTf_i/dt = a (Tf_{i-1} - Tf_i) + kf (Tw_i - Tf_i)

    Wall/sample effective equation:
        dTw_i/dt = kw (Tf_i - Tw_i) - kloss (Tw_i - Tenv)
    """

    v = p[0]
    kf = p[1]
    kw = p[2]
    kloss = p[3]
    Tw0 = p[4]
    Tenv = p[5]

    ds = L_pipe / N

    s = np.linspace(ds / 2.0, L_pipe - ds / 2.0, N)

    t_end = time_data[-1]

    # Stable time step
    dt_adv = 0.5 * ds / max(v, 1e-12)
    dt_react = 0.2 / max(kf + kw + kloss, 1e-12)

    dt = min(5.0, dt_adv, dt_react)

    t = np.arange(0.0, t_end + dt, dt)

    if t[-1] > t_end:
        t[-1] = t_end

    nt = len(t)

    Tf = Tf_initial * np.ones(N)
    Tw = Tw0 * np.ones(N)

    Tf_hist = np.zeros((N, nt))
    Tw_hist = np.zeros((N, nt))
    Tout_sim = np.zeros(nt)

    Tf_hist[:, 0] = Tf
    Tw_hist[:, 0] = Tw
    Tout_sim[0] = Tf[-1]

    # Interpolator for inlet temperature
    Tin_interpolator = interp1d(
        time_data,
        Tin_data,
        kind="linear",
        fill_value="extrapolate"
    )

    for n in range(1, nt):

        dt_local = t[n] - t[n - 1]

        Tin_now = float(Tin_interpolator(t[n]))

        Tf_old = Tf.copy()
        Tw_old = Tw.copy()

        a = v / ds

        # ----------------------------------------------------
        # Fluid FVM equation:
        #
        # dTf_i/dt = a (Tf_{i-1} - Tf_i)
        #            + kf (Tw_i - Tf_i)
        #
        # Upwind advection + heat exchange
        # ----------------------------------------------------

        Tf_new = np.zeros(N)

        # First cell: inlet boundary condition
        Tf_new[0] = (
            Tf_old[0] + dt_local * (a * Tin_now + kf * Tw_old[0])
        ) / (
            1.0 + dt_local * (a + kf)
        )

        # Internal cells
        for i in range(1, N):
            Tf_new[i] = (
                Tf_old[i] + dt_local * (a * Tf_new[i - 1] + kf * Tw_old[i])
            ) / (
                1.0 + dt_local * (a + kf)
            )

        # ----------------------------------------------------
        # Wall/sample effective thermal storage:
        #
        # dTw_i/dt = kw (Tf_i - Tw_i)
        #            - kloss (Tw_i - Tenv)
        # ----------------------------------------------------

        Tw_new = (
            Tw_old + dt_local * (kw * Tf_new + kloss * Tenv)
        ) / (
            1.0 + dt_local * (kw + kloss)
        )

        Tf = Tf_new.copy()
        Tw = Tw_new.copy()

        Tf_hist[:, n] = Tf
        Tw_hist[:, n] = Tw
        Tout_sim[n] = Tf[-1]

    return Tout_sim, t, Tf_hist, Tw_hist, s


def objective_function(theta, lb, ub, time_s, Tin_exp, Tout_exp, L_pipe, N):
    """
    Least-squares objective function.
    """

    p = decode_parameters(theta, lb, ub)

    Tout_sim, t_sim, _, _, _ = simulate_fvm(
        p,
        time_s,
        Tin_exp,
        L_pipe,
        N,
        Tout_exp[0]
    )

    Tout_interp = np.interp(time_s, t_sim, Tout_sim)

    err = Tout_interp - Tout_exp

    J = np.mean(err ** 2)

    return J


# ------------------------------------------------------------
# 4. Optimization
# ------------------------------------------------------------

theta0 = encode_parameters(p0, lb, ub)

eval_counter = {"count": 0}

def objective_function_with_print(theta):
    eval_counter["count"] += 1

    J = objective_function(
        theta,
        lb,
        ub,
        time_s,
        Tin_exp,
        Tout_exp,
        L_pipe,
        N
    )

    if eval_counter["count"] % 10 == 0:
        p_now = decode_parameters(theta, lb, ub)
        print(
            f"Eval {eval_counter['count']:4d} | "
            f"J = {J:.6e} | "
            f"v = {p_now[0]:.4e}, "
            f"kf = {p_now[1]:.4e}, "
            f"kw = {p_now[2]:.4e}, "
            f"kloss = {p_now[3]:.4e}"
        )

    return J


result = minimize(
    objective_function_with_print,
    theta0,
    method="Nelder-Mead",
    options={
        "maxiter": 60,
        "maxfev": 150,
        "xatol": 1e-4,
        "fatol": 1e-4,
        "disp": True
    }
)

theta_fit = result.x
p_fit = decode_parameters(theta_fit, lb, ub)


# ------------------------------------------------------------
# 5. Run final simulation with calibrated parameters
# ------------------------------------------------------------

Tout_sim, t_sim, Tf_hist, Tw_hist, s = simulate_fvm(
    p_fit,
    time_s,
    Tin_exp,
    L_pipe,
    N,
    Tout_exp[0]
)

Tout_at_data = np.interp(time_s, t_sim, Tout_sim)

RMSE = np.sqrt(np.mean((Tout_at_data - Tout_exp) ** 2))

print("\n===== Calibrated parameters =====")
print(f"v      = {p_fit[0]:.6e} m/s")
print(f"kf     = {p_fit[1]:.6e} 1/s")
print(f"kw     = {p_fit[2]:.6e} 1/s")
print(f"kloss  = {p_fit[3]:.6e} 1/s")
print(f"Tw0    = {p_fit[4]:.4f} C")
print(f"Tenv   = {p_fit[5]:.4f} C")
print(f"Travel time L/v = {L_pipe / p_fit[0] / 60.0:.2f} min")
print(f"RMSE   = {RMSE:.4f} C")


# ------------------------------------------------------------
# 6. Plot inlet, measured outlet, and simulated outlet
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))
plt.plot(time_s / 3600.0, Tin_exp, linewidth=1.5, label="Measured inlet")
plt.plot(time_s / 3600.0, Tout_exp, "o", markersize=4, label="Measured outlet")
plt.plot(t_sim / 3600.0, Tout_sim, linewidth=2, label="Simulated outlet")

plt.xlabel("Time [h]")
plt.ylabel("Temperature [°C]")
plt.legend(loc="best")
plt.grid(True)
plt.title("FVM simulation of U-shaped heat exchanger")
plt.tight_layout()


# ------------------------------------------------------------
# 7. Plot final temperature distribution along pipe
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))
plt.plot(s, Tf_hist[:, -1], linewidth=2, label="Fluid temperature")
plt.plot(s, Tw_hist[:, -1], linewidth=2, label="Wall/sample temperature")

plt.xlabel("Pipe coordinate s [m]")
plt.ylabel("Temperature [°C]")
plt.legend(loc="best")
plt.grid(True)
plt.title("Final temperature distribution along U-shaped pipe")
plt.tight_layout()


# ------------------------------------------------------------
# 8. Plot residual error
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))
plt.plot(time_s / 3600.0, Tout_at_data - Tout_exp, linewidth=1.5)

plt.xlabel("Time [h]")
plt.ylabel("Simulation error [°C]")
plt.grid(True)
plt.title("Outlet temperature error")
plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 9. Save calibrated results
# ------------------------------------------------------------

calibrated_parameters = np.array([
    ["v_m_per_s", p_fit[0]],
    ["kf_1_per_s", p_fit[1]],
    ["kw_1_per_s", p_fit[2]],
    ["kloss_1_per_s", p_fit[3]],
    ["Tw0_C", p_fit[4]],
    ["Tenv_C", p_fit[5]],
    ["RMSE_C", RMSE]
], dtype=object)

with open("calibrated_parameters.csv", "w") as f:
    f.write("parameter,value\n")
    for row in calibrated_parameters:
        f.write(f"{row[0]},{row[1]}\n")

output_table = np.column_stack((
    t_sim,
    t_sim / 3600.0,
    np.interp(t_sim, time_s, Tin_exp),
    Tout_sim
))

np.savetxt(
    "simulated_outlet_temperature.csv",
    output_table,
    delimiter=",",
    header="time_s,time_h,Tin_interp_C,Tout_sim_C",
    comments="",
    fmt="%.8f"
)

print("\nSaved files:")
print("calibrated_parameters.csv")
print("simulated_outlet_temperature.csv")