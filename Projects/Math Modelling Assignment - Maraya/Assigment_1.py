
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Literal, Optional

# Optional animation imports (only used if animate=True)
from matplotlib import animation


InitMode = Literal["disperse", "concentrated"]


@dataclass
class KuramotoParams:
    N: int = 100
    K: float = 7.0
    dt: float = 0.01
    T: float = 50.0
    omega_mean: float = 0.0
    omega_std: float = 1.0
    seed: Optional[int] = 1


def sample_frequencies(N: int, mean: float, std: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=mean, scale=std, size=N)


def initial_phases(
    N: int,
    mode: InitMode,
    rng: np.random.Generator,
    frac_of_circle: float = 0.1,
) -> np.ndarray:
    """
    disperse: uniform on [0, 2π)
    concentrated: uniform on a small arc of length 2π*frac_of_circle
    """
    if mode == "disperse":
        return rng.uniform(0.0, 2 * np.pi, size=N)

    if mode == "concentrated":
        arc = 2 * np.pi * frac_of_circle
        center = rng.uniform(0.0, 2 * np.pi)
        thetas = rng.uniform(center - arc / 2, center + arc / 2, size=N)
        return np.mod(thetas, 2 * np.pi)

    raise ValueError(f"Unknown init mode: {mode}")


def order_parameter(theta: np.ndarray) -> tuple[float, float]:
    """
    Returns (r, Psi) where:
      r e^{i Psi} = (1/N) sum exp(i theta_j)
    """
    z = np.mean(np.exp(1j * theta))
    r = np.abs(z)
    psi = np.angle(z)
    return r, psi


def kuramoto_rhs_mean_field(theta: np.ndarray, omega: np.ndarray, K: float) -> np.ndarray:
    """
    Exact mean-field form for the all-to-all Kuramoto model:
      dθ_i/dt = ω_i + K r sin(Ψ - θ_i)
    """
    r, psi = order_parameter(theta)
    return omega + K * r * np.sin(psi - theta)


def integrate_euler(
    theta0: np.ndarray,
    omega: np.ndarray,
    K: float,
    dt: float,
    T: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Explicit Euler integration.
    Returns:
      t: shape (M,)
      theta_hist: shape (M, N)
      r_hist: shape (M,)
    """
    N = theta0.size
    M = int(np.floor(T / dt)) + 1
    t = np.linspace(0.0, dt * (M - 1), M)

    theta_hist = np.zeros((M, N), dtype=float)
    r_hist = np.zeros(M, dtype=float)

    theta = theta0.copy()
    theta_hist[0] = theta
    r_hist[0] = order_parameter(theta)[0]

    for k in range(1, M):
        dtheta = kuramoto_rhs_mean_field(theta, omega, K)
        theta = theta + dt * dtheta
        theta = np.mod(theta, 2 * np.pi)  # keep in [0, 2π)
        theta_hist[k] = theta
        r_hist[k] = order_parameter(theta)[0]

    return t, theta_hist, r_hist

    print("r min:", r_hist.min())
    print("r max:", r_hist.max())
    print("r final:", r_hist[-1])

def plot_r_vs_time(t, r, title="Kuramoto order parameter r(t)"):
    plt.figure(figsize=(7,4))
    plt.plot(t, r)
    plt.xlabel("time t")
    plt.ylabel("r(t)")
    plt.title(title)
    plt.ylim(0, 1.02)          # give headroom above 1
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # zoom into first few seconds
    plt.figure(figsize=(7,4))
    plt.plot(t, r)
    plt.xlim(0, 5)
    plt.ylim(0, 1.02)
    plt.xlabel("time t")
    plt.ylabel("r(t)")
    plt.title(title + " (zoom: t in [0,5])")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def animate_circle(
    theta_hist: np.ndarray,
    interval_ms: int = 30,
    show_centroid: bool = True,
) -> None:
    """
    Simple unit-circle animation of oscillator phases.
    """
    M, N = theta_hist.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Kuramoto oscillators on the unit circle")

    # unit circle
    circle = plt.Circle((0, 0), 1.0, fill=False, color="gray", alpha=0.6)
    ax.add_artist(circle)

    scat = ax.scatter([], [], s=40, alpha=0.7)
    (centroid_line,) = ax.plot([], [], lw=2)

    def init():
        scat.set_offsets(np.zeros((N, 2)))
        centroid_line.set_data([], [])
        return scat, centroid_line

    def update(frame):
        theta = theta_hist[frame]
        x = np.cos(theta)
        y = np.sin(theta)
        scat.set_offsets(np.column_stack([x, y]))

        if show_centroid:
            r, psi = order_parameter(theta)
            centroid_line.set_data([0, r * np.cos(psi)], [0, r * np.sin(psi)])
        else:
            centroid_line.set_data([], [])

        return scat, centroid_line

    ani = animation.FuncAnimation(
        fig, update, frames=M, init_func=init, blit=True, interval=interval_ms
    )
    plt.show()

def plot_circle_snapshot(theta, title="Kuramoto snapshot"):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.cos(theta)
    y = np.sin(theta)

    r, psi = order_parameter(theta)

    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=50, alpha=0.8)

    # unit circle
    circle = plt.Circle((0,0), 1, fill=False, color='gray', alpha=0.4)
    plt.gca().add_artist(circle)

    # order parameter vector
    plt.plot([0, r*np.cos(psi)],
             [0, r*np.sin(psi)],
             linewidth=3)

    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.gca().set_aspect('equal', 'box')

    plt.xlabel("cos(theta)")
    plt.ylabel("sin(theta)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()
def animate_circle(theta_hist, interval_ms=30, trail=False, save_path=None):
    """
    Animate oscillators on the unit circle.
    theta_hist: array of shape (M, N)
    interval_ms: time between frames in ms
    trail: if True, leaves faint trails (optional)
    save_path: if not None, saves animation (e.g. "kuramoto.mp4" or "kuramoto.gif")
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import animation

    M, N = theta_hist.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("cos(theta)")
    ax.set_ylabel("sin(theta)")
    ax.set_title("Kuramoto oscillators on the unit circle")

    # unit circle
    circle = plt.Circle((0, 0), 1.0, fill=False, color="gray", alpha=0.4)
    ax.add_artist(circle)

    # initial positions
    theta0 = theta_hist[0]
    x0, y0 = np.cos(theta0), np.sin(theta0)

    scat = ax.scatter(x0, y0, s=50, alpha=0.8)

    # order parameter vector (red arrow)
    (vec_line,) = ax.plot([0, 0], [0, 0], linewidth=3)

    # optional trail
    if trail:
        (trail_line,) = ax.plot([], [], linewidth=1, alpha=0.2)
        trail_x, trail_y = [], []
    else:
        trail_line = None

    def update(frame):
        theta = theta_hist[frame]
        x, y = np.cos(theta), np.sin(theta)
        scat.set_offsets(np.column_stack([x, y]))

        r, psi = order_parameter(theta)
        vec_line.set_data([0, r * np.cos(psi)], [0, r * np.sin(psi)])

        if trail:
            trail_x.append(r * np.cos(psi))
            trail_y.append(r * np.sin(psi))
            trail_line.set_data(trail_x, trail_y)
            return scat, vec_line, trail_line

        return scat, vec_line

    ani = animation.FuncAnimation(fig, update, frames=M, interval=interval_ms, blit=True)

    # Save if requested
    if save_path is not None:
        if save_path.endswith(".gif"):
            ani.save(save_path, writer="pillow", fps=max(1, int(1000 / interval_ms)))
        else:
            ani.save(save_path, fps=max(1, int(1000 / interval_ms)))

    plt.show()

def run_part1(
    params: KuramotoParams,
    init_mode: InitMode = "disperse",
    concentrated_frac: float = 0.1,
    animate: bool = False,
) -> None:
    rng = np.random.default_rng(params.seed)

    omega = sample_frequencies(params.N, params.omega_mean, params.omega_std, rng)
    theta0 = initial_phases(params.N, init_mode, rng, frac_of_circle=concentrated_frac)

    t, theta_hist, r_hist = integrate_euler(theta0, omega, params.K, params.dt, params.T)
    plot_circle_snapshot(theta_hist[-1],
                     title=f"Snapshot — N={params.N}, K={params.K}")

    plot_r_vs_time(
        t,
        r_hist,
        title=f"Kuramoto r(t) — N={params.N}, K={params.K}, init={init_mode}",
    )
    animate_circle(theta_hist, interval_ms=30)
    if animate:
        animate_circle(theta_hist, interval_ms=30, show_centroid=True)



# ===================== PART 2 ===============================


def compute_r_infty_vs_K():
    """
    Part 2: compute empirical r_infty(K) with error bars and plot theoretical curve.
    Paste this function above the `if __name__ == "__main__":` block and call it from there.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from math import pi
    import time

    t0 = time.time()
    # ----------------- USER PARAMETERS (tune if slow) -----------------
    N = 100               # number of oscillators (increase to 5000 for better agreement)
    seed = 1
    dt = 0.02
    T = 120.0              # total simulation time (seconds of sim) -- reduce if too slow
    avg_steps = 150        # average over last avg_steps to compute r_infty
    K_samples = 15         # number of empirical K samples (20 suggested; 15 is faster)
    ntheta = 400           # integration points for theoretical integral
    rgrid_samples = 401    # samples to scan r in [0,1) for bracketing
    # ----------------------------------------------------------------

    rng = np.random.default_rng(seed)

    # Gaussian g(omega)
    g0 = 1.0 / np.sqrt(2.0 * pi)
    Kc = 2.0 / (pi * g0)
    print(f"[Part2] Gaussian analytical Kc = {Kc:.4f}")

    Kmin = Kc / 3.0
    Kmax = 3.0 * Kc

    # time grid length (number of integration steps)
    M = int(np.floor(T / dt))
    if avg_steps >= M:
        raise ValueError("avg_steps must be smaller than number of simulation steps; increase T or reduce avg_steps")

    # sample omegas and initial thetas once (same realization for all K to reduce variance)
    omega = rng.normal(loc=0.0, scale=1.0, size=N)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=N)

    def order_parameter(theta):
        z = np.mean(np.exp(1j * theta))
        return np.abs(z), np.angle(z)

    def kuramoto_rhs(theta, omega, K):
        r, psi = order_parameter(theta)
        return omega + K * r * np.sin(psi - theta)

    def integrate_euler_rhistory(K):
        """Integrate using explicit Euler, return r_hist (length M)."""
        theta = theta0.copy()
        r_hist = np.empty(M, dtype=float)
        for k in range(M):
            dtheta = kuramoto_rhs(theta, omega, K)
            theta += dt * dtheta
            theta %= (2.0 * np.pi)
            r_hist[k] = order_parameter(theta)[0]
        return r_hist

    # --------- Empirical loop over K samples ----------
    Ks = np.linspace(Kmin, Kmax, K_samples)
    r_means = np.zeros_like(Ks)
    r_stds  = np.zeros_like(Ks)

    print("[Part2] Running empirical simulations for Ks ...")
    for i, K in enumerate(Ks):
        r_hist = integrate_euler_rhistory(K)
        last = r_hist[-avg_steps:]
        r_means[i] = np.mean(last)
        r_stds[i]  = np.std(last, ddof=0)
        print(f"  K={K:.4f}  r_inf_mean={r_means[i]:.4f}  std={r_stds[i]:.4f}")

    # --------- Theoretical self-consistency (robust solver) ----------
    def self_consistency_F(r, K, ntheta_local=ntheta):
        """F(r) = 1 - K * integral_{-pi/2}^{pi/2} cos^2(theta) g(K r sin theta) dtheta."""
        if r <= 0.0:
            # small r limit
            return 1.0 - K * g0 * (np.pi / 2.0)
        thetas = np.linspace(-np.pi / 2.0, np.pi / 2.0, ntheta_local)
        cos2 = np.cos(thetas) ** 2
        arg = K * r * np.sin(thetas)
        gvals = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * arg * arg)
        integral = np.trapz(cos2 * gvals, thetas)
        return 1.0 - K * integral

    def solve_r_bracket(Kval, rgrid=None, tol=1e-7, maxiter=60):
        """Robust solver: sample F(r) on rgrid to find bracket then bisection refine."""
        # quick check: if F(0) > 0 => no nonzero solution
        if self_consistency_F(0.0, Kval) > 0.0:
            return 0.0

        if rgrid is None:
            rgrid = np.linspace(1e-8, 0.9999, rgrid_samples)

        Fvals = np.array([self_consistency_F(rr, Kval) for rr in rgrid])
        # find sign changes
        idxs = np.where(np.sign(Fvals[:-1]) * np.sign(Fvals[1:]) < 0)[0]
        if len(idxs) == 0:
            # fallback: sometimes numerical noise; try to find minimum absolute value
            idx = np.argmin(np.abs(Fvals))
            if abs(Fvals[idx]) < 1e-4:
                return rgrid[idx]
            return 0.0

        # pick first bracket (smallest positive root)
        idx = idxs[0]
        lo = rgrid[idx]
        hi = rgrid[idx + 1]
        Flo = self_consistency_F(lo, Kval)
        Fhi = self_consistency_F(hi, Kval)
        # bisection refine
        for _ in range(maxiter):
            mid = 0.5 * (lo + hi)
            Fm = self_consistency_F(mid, Kval)
            if abs(Fm) < tol:
                return mid
            if Flo * Fm <= 0:
                hi = mid
                Fhi = Fm
            else:
                lo = mid
                Flo = Fm
        return 0.5 * (lo + hi)

    # compute theory curve on fine K grid
    K_fine = np.linspace(Kmin, Kmax, 300)
    r_theory = np.zeros_like(K_fine)
    print("[Part2] Solving theoretical self-consistency curve (this may take a moment)...")
    for j, Kval in enumerate(K_fine):
        r_theory[j] = solve_r_bracket(Kval)

    # --------- Plot empirical + theoretical -----------
    plt.figure(figsize=(8,5.2))
    plt.plot(K_fine, r_theory, '-', lw=2, color='tab:blue', label='Theoretical')
    plt.errorbar(Ks, r_means, yerr=r_stds, fmt='o', color='tab:red',
                 ecolor='tab:red', elinewidth=1.5, capsize=3, markersize=6, label='Empirical (mean ± std)')
    plt.axvline(Kc, color='k', linestyle='--', label=f'Kc ≈ {Kc:.3f}')
    plt.xlabel('Coupling strength K')
    plt.ylabel(r'Order parameter $r_\infty$')
    plt.title(r'$r_\infty(K)$ : empirical vs theoretical')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    outname = "r_infty_vs_K.png"
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    plt.show()

    dt_sec = time.time() - t0
    print(f"[Part2] Done. Figure saved as '{outname}'. Elapsed ~{dt_sec:.1f}s")


# ===================== PART 3 ===============================


def millennium_bridge_simulation():


    import numpy as np
    import matplotlib.pyplot as plt

    print("\nRunning Part 3: Millennium Bridge simulation\n")

    # ---------------- PARAMETERS (tuned so N0 < Nc < Nmax) ----------------
    M = 1.0
    B = 0.05
    K_bridge = 1.0

    # IMPORTANT: choose G small enough so Nc is not below N0
    G = 0.004       # lowered (helps make Nc larger, so tc is not at 0)
    C = 1.2
    alpha = 0.0

    Omega_bridge = np.sqrt(K_bridge / M)

    dt = 0.01
    T_total = 300.0
    steps = int(T_total / dt)

    # Crowd ramp parameters
    N0 = 10
    dN = 5
    deltaT = 30.0
    Nmax = 200

    # Frequency distribution P(Ω): Normal(mean=Ω, std=σ)
    sigma = 0.2
    P_Omega_at_mean = 1.0 / (np.sqrt(2*np.pi) * sigma)

    # Theoretical critical crowd size Nc
    # Nc = 2 B Ω / (π G C P(Ω))  and for Normal P(Ω)=1/(sqrt(2π)σ)
    Nc = (2 * B * Omega_bridge) / (np.pi * G * C * P_Omega_at_mean)
    print(f"Theoretical Nc ≈ {Nc:.2f}  (should satisfy N0 < Nc < Nmax)")

    # ---------------- INITIAL CONDITIONS ----------------
    # small initial bridge motion
    X = 1e-2
    Xdot = 0.0

    # initial crowd
    N = N0
    theta = np.random.uniform(0, 2*np.pi, N)
    Omega_i = np.random.normal(Omega_bridge, sigma, N)

    # ---------------- STORAGE ----------------
    time = np.empty(steps)
    A_hist = np.empty(steps)
    R_hist = np.empty(steps)
    N_hist = np.empty(steps, dtype=int)

    tc = None

    # precompute ramp step interval
    ramp_every = int(deltaT / dt)

    # ---------------- SIMULATION LOOP ----------------
    for step in range(steps):

        t = step * dt

        # Crowd ramp: add dN new pedestrians every ΔT until reaching Nmax
        if step % ramp_every == 0 and N < Nmax:
            add = min(dN, Nmax - N)
            new_theta = np.random.uniform(0, 2*np.pi, add)
            new_Omega = np.random.normal(Omega_bridge, sigma, add)
            theta = np.concatenate([theta, new_theta])
            Omega_i = np.concatenate([Omega_i, new_Omega])
            N += add

        # Bridge amplitude A(t)
        A = np.sqrt(X**2 + (Xdot / Omega_bridge)**2)

        # Kuramoto order parameter R(t)
        R = np.abs(np.mean(np.exp(1j * theta)))

        # Save
        time[step] = t
        A_hist[step] = A
        R_hist[step] = R
        N_hist[step] = N

        # time when N first reaches/exceeds Nc
        if tc is None and N >= Nc:
            tc = t

        # ---------------- Bridge dynamics (Euler) ----------------
        forcing = np.sum(G * np.sin(theta))
        Xddot = (forcing - B * Xdot - K_bridge * X) / M

        Xdot = Xdot + dt * Xddot
        X = X + dt * Xdot

        # ---------------- Pedestrian phase dynamics ----------------
        # IMPORTANT FIX: Psi(t) must follow the assignment definition
        # Psi(t) = atan2( X(t), Xdot(t)/Omega )
        Psi = np.arctan2(X, Xdot / Omega_bridge)

        theta_dot = Omega_i + C * A * np.sin(Psi - theta + alpha)
        theta = theta + dt * theta_dot

        # keep theta bounded (optional but nice)
        theta = np.mod(theta, 2*np.pi)

    # ---------------- PLOTS ----------------
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)

    axes[0].plot(time, N_hist)
    axes[0].set_ylabel("N(t)")
    axes[0].set_title("Crowd ramp experiment")

    axes[1].plot(time, A_hist)
    axes[1].set_ylabel("A(t)")

    axes[2].plot(time, R_hist)
    axes[2].set_ylabel("R(t)")
    axes[2].set_xlabel("time")

    if tc is not None:
        for ax in axes:
            ax.axvline(tc, linestyle="--", color="black")
        axes[0].text(tc + 1, axes[0].get_ylim()[1]*0.95, "N = Nc", va="top")

    plt.tight_layout()
    plt.savefig("millennium_bridge_results.png", dpi=300)
    plt.show()

    print("Saved figure: millennium_bridge_results.png")
    if tc is None:
        print("Note: tc was not found (N never reached Nc). Try increasing Nmax or adjusting parameters.")
    else:
        print(f"tc ≈ {tc:.2f} s  (first time N(t) ≥ Nc)")



if __name__ == "__main__":
    # --- Default values requested in the assignment for Part 1 ---
    p = KuramotoParams(N=100, K=1.0, dt=0.01, T=20.0, seed=1)
    run_part1(p, init_mode="disperse", animate=False)
    compute_r_infty_vs_K()
    millennium_bridge_simulation()

    