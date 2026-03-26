from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# importing func from analytical part:
from gierer_1d_analytical import gm_equilibrium, gm_jacobian, turing_test, neumann_eigenvalue_1d, analyze_modes_1d, leading_mode

# Numerical part

# for figure saving:
ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figs"
ANIM_DIR = ROOT / "animations"
FIG_DIR.mkdir(exist_ok=True)
ANIM_DIR.mkdir(exist_ok=True)


def save_anim_path(filename: str):
    return ANIM_DIR / filename

# Numerical needed funcs - model build up

def laplacian_1d(u: np.ndarray, dx: float):
    return (np.roll(u, 1) - 2 * u + np.roll(u, -1)) / dx**2


def apply_neumann(u: np.ndarray):
    u[0] = u[1]
    u[-1] = u[-2]
    return u


def gm_rhs(u, v, a, b, d, gamma, dx):
    eps = 1e-12
    lu = laplacian_1d(u, dx)
    lv = laplacian_1d(v, dx)

    f = a - b * u + (u**2) / np.maximum(v, eps)
    g = u**2 - v

    dudt = lu + gamma * f
    dvdt = d * lv + gamma * g
    return dudt, dvdt

# Time stepping, simulation and animation

def simulate_gm_1d(a=0.4, b=1.0, d=30.0, gamma=1.0,
                   L=40.0, N=40, dx=1.0, dt=0.01,
                   num_steps=50000, save_every=500,
                   noise_amp=0.01, seed=0):

    rng = np.random.default_rng(seed)
    x = np.linspace(0, L, N)

    u_star, v_star = gm_equilibrium(a, b)

    u = u_star * (1 + noise_amp * rng.standard_normal(N))
    v = v_star * (1 + noise_amp * rng.standard_normal(N))

    apply_neumann(u)
    apply_neumann(v)

    frames = []

    for step in range(num_steps):
        dudt, dvdt = gm_rhs(u, v, a, b, d, gamma, dx)
        u += dt * dudt
        v += dt * dvdt

        apply_neumann(u)
        apply_neumann(v)
        v = np.maximum(v, 1e-10)

        if step % save_every == 0:
            frames.append(v.copy())

    return x, np.array(frames)

def animate_solution(x, frames, filename="gm_1d.gif"):
    fig, ax = plt.subplots()
    line, = ax.plot(x, frames[0])
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(np.min(frames), np.max(frames))

    def update(i):
        line.set_ydata(frames[i])
        ax.set_title(f"timestep {i}")
        return (line,)

    ani = FuncAnimation(fig, update, frames=len(frames), blit=True)
    ani.save(save_anim_path(filename), writer=PillowWriter(fps=10))
    plt.close()

# Main exercise - Parameters given for numerical part: 
def run_case(d_value):
    a, b, gamma, L = 0.4, 1.0, 1.0, 40.0

    # Analytical prediction (imported!)
    rows = analyze_modes_1d(a, b, d_value, gamma, L, n_max=20)
    lead = leading_mode(rows)

    print(f"\nCase d={d_value}")
    if lead:
        print(f"Predicted leading mode n* = {lead['n']}")
    else:
        print("No instability predicted")

    # Numerical simulation
    x, frames = simulate_gm_1d(a=a, b=b, d=d_value, gamma=gamma,
                              L=L, N=40, dx=1.0, dt=0.01,
                              num_steps=50000, save_every=500, seed=1)

    animate_solution(x, frames, filename=f"gm_1d_d{int(d_value)}.gif")


if __name__ == "__main__":
    print(f"Saving animations to: {ANIM_DIR}")

    run_case(30.0)
    run_case(20.0)

