import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



#milenium bridge simulation with crowd ramp and tuned parameters to see the effect clearly. See comments in code for details.

rng = np.random.default_rng(0)


# Bridge parameters
M = 1.0
K = 1.0
B = 0.40                      # increased damping -> increases Nc and stabilizes growth
Omega = np.sqrt(K / M)


# Coupling / forcing parameters
G = 0.02                      # smaller forcing per pedestrian -> increases Nc
C = 0.60                      # moderate coupling
alpha = np.pi / 2

# Pedestrian frequency distribution (Normal centered at Omega)
sigma = 2.0                   # larger spread -> P(Omega) smaller -> Nc larger

def sample_Omegas(n: int) -> np.ndarray:
    return rng.normal(loc=Omega, scale=sigma, size=n)


# Crowd ramp settings (staircase N(t))
N0 = 20
Nmax = 200
dN = 10
dT = 10.0

T_end = ((Nmax - N0) // dN) * dT + 30.0

# Integration settings
rtol = 1e-6
atol = 1e-8
max_step = 0.05



# Theory: critical crowd size Nc and tc
P_at_Omega = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)          # Normal pdf at its mean
Nc = (2.0 * B * Omega) / (np.pi * G * C * P_at_Omega)

if Nc <= N0:
    tc = 0.0
else:
    kcrit = int(np.ceil((Nc - N0) / dN))
    tc = kcrit * dT

print(f"Omega={Omega:.4f}")
print(f"sigma={sigma:.4f}, P(Omega)={P_at_Omega:.6f}")
print(f"Nc(theory)={Nc:.2f}, tc={tc:.2f} s")



# Helper: A(t), Psi(t) from X, V

def bridge_amp_phase(X: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.sqrt(X**2 + (V / Omega)**2)
    Psi = np.arctan2(X, V / Omega)
    return A, Psi


# RHS for fixed N within a segment
# y = [X, V, theta_1,...,theta_N]
def make_rhs(Omegas: np.ndarray):
    def rhs(t, y):
        X = y[0]
        V = y[1]
        thetas = y[2:]

        # Bridge amplitude/phase (scalar)
        A, Psi = bridge_amp_phase(np.array([X]), np.array([V]))
        A = A[0]
        Psi = Psi[0]

        # Pedestrian forcing on bridge
        forcing = G * np.sum(np.sin(thetas))

        dX = V
        dV = (forcing - B * V - K * X) / M

        # Pedestrian phase dynamics
        dtheta = Omegas + C * A * np.sin(Psi - thetas + alpha)

        return np.concatenate(([dX, dV], dtheta))

    return rhs


# Simulation with crowd ramp (piecewise segments)
X0 = 1e-3
V0 = 0.0
thetas0 = rng.uniform(0.0, 2.0 * np.pi, size=N0)
Omegas = sample_Omegas(N0)

y0 = np.concatenate(([X0, V0], thetas0))

t_all, X_all, V_all, N_all, R_all = [], [], [], [], []

t_seg_start = 0.0
t_seg_end = dT

while t_seg_start < T_end - 1e-12:
    t_seg_end = min(t_seg_end, T_end)

    sol = solve_ivp(
        make_rhs(Omegas),
        t_span=(t_seg_start, t_seg_end),
        y0=y0,
        method="RK45",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    # Store only what is consistent across varying N
    t_all.append(sol.t)
    X_all.append(sol.y[0, :])
    V_all.append(sol.y[1, :])
    N_all.append(np.full_like(sol.t, len(Omegas), dtype=float))

    # Order parameter for this segment
    thetas_seg = sol.y[2:, :]
    R_seg = np.abs(np.mean(np.exp(1j * thetas_seg), axis=0))
    R_all.append(R_seg)

    # Advance
    t_seg_start = sol.t[-1]
    y0 = sol.y[:, -1].copy()

    # At each ramp boundary, add pedestrians (unless at Nmax)
    if abs(t_seg_start / dT - round(t_seg_start / dT)) < 1e-9:
        if len(Omegas) < Nmax:
            add = min(dN, Nmax - len(Omegas))
            new_thetas = rng.uniform(0.0, 2.0 * np.pi, size=add)
            new_Omegas = sample_Omegas(add)

            y0 = np.concatenate((y0, new_thetas))
            Omegas = np.concatenate((Omegas, new_Omegas))

    t_seg_end = t_seg_start + dT


# Concatenate 1D arrays safely
t = np.concatenate(t_all)
X = np.concatenate(X_all)
V = np.concatenate(V_all)
N_t = np.concatenate(N_all)
R = np.concatenate(R_all)

A, Psi = bridge_amp_phase(X, V)


# Optional: a numerical onset time (when A exceeds a small threshold)
A_thresh = 0.05
idx = np.where(A > A_thresh)[0]
t_onset = t[idx[0]] if len(idx) else np.nan
print(f"t_onset (A>{A_thresh}) ≈ {t_onset:.2f} s")


# Plot: N(t), A(t), R(t)
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(t, N_t)
axes[0].axvline(tc, linestyle="--")
axes[0].set_ylabel("N(t)")
axes[0].set_title("Millennium Bridge — crowd ramp experiment (tuned parameters)")

axes[1].plot(t, A)
axes[1].axvline(tc, linestyle="--")
axes[1].set_ylabel("A(t)")

axes[2].plot(t, R)
axes[2].axvline(tc, linestyle="--")
axes[2].set_ylabel("R(t)")
axes[2].set_xlabel("time")

plt.tight_layout()
plt.show()

def simulate_bridge_for_sigma(
    sigma: float,
    *,
    seed: int = 0,
    # Bridge
    M: float = 1.0,
    K: float = 1.0,
    B: float = 0.40,
    # Coupling
    G: float = 0.02,
    C: float = 0.60,
    alpha: float = np.pi / 2,
    # Ramp
    N0: int = 20,
    Nmax: int = 200,
    dN: int = 10,
    dT: float = 10.0,
    T_extra: float = 30.0,
    # Integrator
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_step: float = 0.05,
    # Empirical threshold
    A_thresh: float = 0.5,
):
    """
    Runs the crowd-ramp bridge simulation for a given sigma and returns:
      Nc_theory, N_empirical, tc_theory, t_onset, (t, N_t, A, R)
    N_empirical is defined as N(t_onset) where A(t) first exceeds A_thresh.
    If onset never happens, N_empirical = np.nan.
    """
    rng = np.random.default_rng(seed)

    Omega = np.sqrt(K / M)

    def sample_Omegas(n: int) -> np.ndarray:
        return rng.normal(loc=Omega, scale=sigma, size=n)

    # Theory
    P_at_Omega = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    Nc_theory = (2.0 * B * Omega) / (np.pi * G * C * P_at_Omega)

    if Nc_theory <= N0:
        tc_theory = 0.0
    else:
        kcrit = int(np.ceil((Nc_theory - N0) / dN))
        tc_theory = kcrit * dT

    def bridge_amp_phase(X: np.ndarray, V: np.ndarray):
        A = np.sqrt(X**2 + (V / Omega) ** 2)
        Psi = np.arctan2(X, V / Omega)
        return A, Psi

    def make_rhs(Omegas: np.ndarray):
        def rhs(t, y):
            X = y[0]
            V = y[1]
            thetas = y[2:]

            A, Psi = bridge_amp_phase(np.array([X]), np.array([V]))
            A = A[0]
            Psi = Psi[0]

            forcing = G * np.sum(np.sin(thetas))

            dX = V
            dV = (forcing - B * V - K * X) / M
            dtheta = Omegas + C * A * np.sin(Psi - thetas + alpha)

            return np.concatenate(([dX, dV], dtheta))

        return rhs

    # Ramp horizon
    T_end = ((Nmax - N0) // dN) * dT + T_extra

    # Initial conditions
    X0 = 1e-3
    V0 = 0.0
    thetas0 = rng.uniform(0.0, 2.0 * np.pi, size=N0)
    Omegas = sample_Omegas(N0)
    y0 = np.concatenate(([X0, V0], thetas0))

    # Storage (safe across changing N)
    t_all, X_all, V_all, N_all, R_all, A_all = [], [], [], [], [], []

    t_seg_start = 0.0
    t_seg_end = dT

    while t_seg_start < T_end - 1e-12:
        t_seg_end = min(t_seg_end, T_end)

        sol = solve_ivp(
            make_rhs(Omegas),
            t_span=(t_seg_start, t_seg_end),
            y0=y0,
            method="RK45",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

        # store time + bridge
        t_all.append(sol.t)
        X_seg = sol.y[0, :]
        V_seg = sol.y[1, :]
        X_all.append(X_seg)
        V_all.append(V_seg)

        N_now = len(Omegas)
        N_all.append(np.full_like(sol.t, N_now, dtype=float))

        # compute A(t) for segment
        A_seg, _ = bridge_amp_phase(X_seg, V_seg)
        A_all.append(A_seg)

        # order parameter segment
        thetas_seg = sol.y[2:, :]
        R_seg = np.abs(np.mean(np.exp(1j * thetas_seg), axis=0))
        R_all.append(R_seg)

        # advance
        t_seg_start = sol.t[-1]
        y0 = sol.y[:, -1].copy()

        # add pedestrians at ramp boundaries
        if abs(t_seg_start / dT - round(t_seg_start / dT)) < 1e-9:
            if len(Omegas) < Nmax:
                add = min(dN, Nmax - len(Omegas))
                new_thetas = rng.uniform(0.0, 2.0 * np.pi, size=add)
                new_Omegas = sample_Omegas(add)
                y0 = np.concatenate((y0, new_thetas))
                Omegas = np.concatenate((Omegas, new_Omegas))

        t_seg_end = t_seg_start + dT

    # Concatenate
    t = np.concatenate(t_all)
    N_t = np.concatenate(N_all)
    A = np.concatenate(A_all)
    R = np.concatenate(R_all)

    # Empirical onset from A_thresh
    idx = np.where(A > A_thresh)[0]
    if len(idx) == 0:
        t_onset = np.nan
        N_emp = np.nan
    else:
        i0 = idx[0]
        t_onset = t[i0]
        N_emp = N_t[i0]

    return Nc_theory, N_emp, tc_theory, t_onset, (t, N_t, A, R)


# Run a sigma sweep
sigma_values = np.linspace(0.5, 5.0, 10)   
A_thresh = 0.5                             # empirical onset threshold

Nc_theory_list = []
N_emp_list = []
tc_list = []
t_onset_list = []

# Optional: multiple seeds to show variability
seeds = [0, 1, 2]   # set to [0] for faster; more seeds = smoother averages

for sig in sigma_values:
    emp_vals = []
    onset_vals = []
    tc_vals = []
    for sd in seeds:
        Nc_th, N_emp, tc, t_onset, _ = simulate_bridge_for_sigma(
            sig,
            seed=sd,
            A_thresh=A_thresh,
        )
        emp_vals.append(N_emp)
        onset_vals.append(t_onset)
        tc_vals.append(tc)

    Nc_theory_list.append(Nc_th)  # same for all seeds
    N_emp_list.append(np.nanmean(emp_vals))
    t_onset_list.append(np.nanmean(onset_vals))
    tc_list.append(np.nanmean(tc_vals))

Nc_theory_arr = np.array(Nc_theory_list, dtype=float)
N_emp_arr = np.array(N_emp_list, dtype=float)


# Plot: empirical vs theory Nc as a function of sigma
plt.figure(figsize=(8, 5))
plt.plot(sigma_values, Nc_theory_arr, marker="o", label="Theory $N_c$")
plt.plot(sigma_values, N_emp_arr, marker="o", label=f"Empirical $N_c$ (A>{A_thresh})")
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$N_c$")
plt.title("Critical crowd size vs frequency spread")
plt.legend()
plt.tight_layout()
plt.show()


# (Optional) Show onset time comparison
plt.figure(figsize=(8, 5))
plt.plot(sigma_values, tc_list, marker="o", label="Theory $t_c$")
plt.plot(sigma_values, t_onset_list, marker="o", label=f"Empirical onset time (A>{A_thresh})")
plt.xlabel(r"$\sigma$")
plt.ylabel("time")
plt.title("Onset time vs frequency spread")
plt.legend()
plt.tight_layout()
plt.show()

