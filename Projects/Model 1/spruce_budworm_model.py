import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def spruce_budworm(t: float, x: float, r: float = 0.5, k: float = 10.0) -> float:
    """Spruce budworm model: dx/dt = r*x*(1-x/k) - x^2/(1+x^2)."""
    return r * x * (1 - x / k) - (x**2) / (1 + x**2)


def plot_spruce_budworm_rate(x_t, r=0.5, k=10.0, n=400):
    x = np.linspace(0, k, n)
    dxdt = spruce_budworm(0.0, x, r=r, k=k)

    s = np.sign(dxdt)
    s[s == 0] = 1e-12
    crossing_idx = np.where(np.diff(s) != 0)[0]

    eq_points, stability = [], []
    for i in crossing_idx:
        x0, x1 = x[i], x[i + 1]
        y0, y1 = dxdt[i], dxdt[i + 1]

        x_star = 0.5 * (x0 + x1) if y1 == y0 else x0 - y0 * (x1 - x0) / (y1 - y0)

        if y0 > 0 and y1 < 0:
            stab = "stable"
        elif y0 < 0 and y1 > 0:
            stab = "unstable"
        else:
            stab = "stable" if (y1 - y0) < 0 else "unstable"

        eq_points.append(x_star)
        stability.append(stab)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, dxdt)
    ax.axhline(0, linewidth=2)
    ax.axvline(x_t, linestyle="--")

    eq_points = np.array(eq_points, dtype=float)
    if len(eq_points) > 0:
        stable_mask = np.array([s == "stable" for s in stability])
        unstable_mask = ~stable_mask
        ax.plot(eq_points[stable_mask], np.zeros(np.sum(stable_mask)), "bo", label="Stable")
        ax.plot(eq_points[unstable_mask], np.zeros(np.sum(unstable_mask)), "ro", label="Unstable")
        ax.legend()

    ax.set_xlabel("Budworm Population")
    ax.set_ylabel("Rate of Change")
    ax.set_title("Spruce budworm Rate of Change")
    ax.grid(True)

    return fig, ax


def evolve_spruce_budworm(t: np.ndarray, x: np.ndarray, r=0.5, k=10.0, t_eval=10, n_eval=200):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    t_span = (t[-1], t[-1] + t_eval)
    t_points = np.linspace(t_span[0], t_span[1], n_eval)

    sol = solve_ivp(
        fun=spruce_budworm,
        t_span=t_span,
        y0=[x[-1]],
        t_eval=t_points,
        args=(r, k),
        method="RK45",
    )

    t_new = sol.t[1:]
    x_new = sol.y[0][1:]

    t_out = np.concatenate([t, t_new])
    x_out = np.concatenate([x, x_new])
    x_out = np.clip(x_out, 0.0, None)

    return t_out, x_out


def plot_spruce_budworm(t: np.ndarray, x: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, x, color="green")
    ax.set_xlabel("Time")
    ax.set_ylabel("Budworm Population")
    ax.set_title("Spruce budworm Population Dynamics")
    ax.set_ylim(bottom=0)
    ax.grid(True)
    return fig, ax
