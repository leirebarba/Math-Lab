from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# GRAY-SCOTT 2D
# Assignment coverage:
#   - 2D Gray-Scott model on a square grid
#   - periodic boundary conditions via np.roll()
#   - initial condition exactly as in the assignment:
#       * u = 1, v = 0 everywhere
#       * centered 20x20 square with (u,v) = (0.5, 0.5)
#       * 10% random perturbation inside the square
#   - assignment cases A-E with D1 = 0.1 and D2 = 0.05


# For figure saving:
ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figs"
ANIM_DIR = ROOT / "animations"
FIG_DIR.mkdir(exist_ok=True)
ANIM_DIR.mkdir(exist_ok=True)

def save_fig(filename: str):
    out = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out}")

def animation_path(filename: str):
    out = ANIM_DIR / filename
    print(f"Saving animation: {out}")
    return out

# Assignment paramenters: 
D1 = 0.1
D2 = 0.05

GRAY_SCOTT_CASES = {
    "A": {"F": 0.040, "k": 0.060},
    "B": {"F": 0.014, "k": 0.047},
    "C": {"F": 0.062, "k": 0.065},
    "D": {"F": 0.078, "k": 0.061},
    "E": {"F": 0.082, "k": 0.059},
}

# Model build up, model functions: 
def laplacian_periodic(U: np.ndarray, dx: float):
    return (
        np.roll(U, 1, axis=0)
        + np.roll(U, -1, axis=0)
        + np.roll(U, 1, axis=1)
        + np.roll(U, -1, axis=1)
        - 4.0 * U
    ) / dx**2



def gray_scott_rhs(u: np.ndarray, v: np.ndarray, D1: float, D2: float,
                   F: float, k: float, dx: float):
    lu = laplacian_periodic(u, dx)
    lv = laplacian_periodic(v, dx)
    uv2 = u * v * v

    dudt = D1 * lu - uv2 + F * (1.0 - u)
    dvdt = D2 * lv + uv2 - (F + k) * v
    return dudt, dvdt

# Initial condition:
def initial_condition(N: int = 250, patch_size: int = 20, seed: int = 0):
    """ Assignment-style initial condition.

    Background: u = 1, v = 0
    Center square: u = 0.5, v = 0.5 with 10% random perturbation
    """
    rng = np.random.default_rng(seed)

    u = np.ones((N, N), dtype=float)
    v = np.zeros((N, N), dtype=float)

    c = N // 2
    r = patch_size // 2
    sl = np.s_[c - r:c + r, c - r:c + r]

    u[sl] = 0.5
    v[sl] = 0.5

    # 10% perturbation around 0.5 inside the square
    u[sl] += 0.1 * 0.5 * rng.standard_normal((patch_size, patch_size))
    v[sl] += 0.1 * 0.5 * rng.standard_normal((patch_size, patch_size))

    u = np.clip(u, 0.0, 2.0)
    v = np.clip(v, 0.0, 2.0)
    return u, v

# Simulation function:
def simulate_gray_scott(F: float, k: float,
                        D1: float = D1, D2: float = D2,
                        N: int = 250, dx: float = 1.0, dt: float = 1.0,
                        patch_size: int = 20,
                        num_steps: int = 50000, save_every: int = 200,
                        seed: int = 0):
    u, v = initial_condition(N=N, patch_size=patch_size, seed=seed)
    frames_u = []
    frames_v = []

    for step in range(num_steps):
        dudt, dvdt = gray_scott_rhs(u, v, D1, D2, F, k, dx)
        u += dt * dudt
        v += dt * dvdt

        if step % save_every == 0:
            frames_u.append(u.copy())
            frames_v.append(v.copy())

    return np.array(frames_u), np.array(frames_v)

# Plots and Animations: 
def plot_initial_condition(u0: np.ndarray, v0: np.ndarray, case_label: str):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im0 = axes[0].imshow(u0, origin="lower", cmap="viridis")
    axes[0].set_title(f"Initial u - case {case_label}")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(v0, origin="lower", cmap="viridis")
    axes[1].set_title(f"Initial v - case {case_label}")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    save_fig(f"07_gray_scott_initial_{case_label}.png")

def plot_final_frame(frame: np.ndarray, case_label: str, species: str = "v"):
    plt.figure(figsize=(6, 6))
    plt.imshow(frame, origin="lower", cmap="plasma")
    plt.title(f"Final {species} pattern - case {case_label}")
    plt.xticks([])
    plt.yticks([])
    save_fig(f"08_gray_scott_final_{species}_{case_label}.png")



def animate_frames(frames: np.ndarray, filename: str, title_prefix: str = "Gray-Scott v"):
    fig, ax = plt.subplots(figsize=(6, 6))

    vmin = np.percentile(frames, 1)
    vmax = np.percentile(frames, 99)
    im = ax.imshow(frames[0], origin="lower", cmap="plasma", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

    def update(i):
        im.set_array(frames[i])
        ax.set_title(f"{title_prefix} - frame {i+1}/{len(frames)}")
        return (im,)

    ani = FuncAnimation(fig, update, frames=len(frames), blit=True)
    ani.save(animation_path(filename), writer=PillowWriter(fps=10))
    plt.close(fig)

# Assignment specific cases 

def run_case(case_label: str,
             N: int = 250,
             dx: float = 1.0,
             dt: float = 1.0,
             patch_size: int = 20,
             num_steps: int = 50000,
             save_every: int = 200,
             seed: int = 1):
    pars = GRAY_SCOTT_CASES[case_label]
    F = pars["F"]
    k = pars["k"]

    print("\n" + "=" * 72)
    print(f"GRAY-SCOTT CASE {case_label}")
    print(f"D1 = {D1}, D2 = {D2}, F = {F}, k = {k}")
    print("=" * 72)

    u0, v0 = initial_condition(N=N, patch_size=patch_size, seed=seed)
    plot_initial_condition(u0, v0, case_label)

    frames_u, frames_v = simulate_gray_scott(
        F=F,
        k=k,
        D1=D1,
        D2=D2,
        N=N,
        dx=dx,
        dt=dt,
        patch_size=patch_size,
        num_steps=num_steps,
        save_every=save_every,
        seed=seed,
    )

    plot_final_frame(frames_v[-1], case_label, species="v")
    animate_frames(frames_v, filename=f"gray_scott_{case_label}.gif", title_prefix=f"Gray-Scott v ({case_label})")

def main():
    print(f"Figures will be saved in: {FIG_DIR}")
    print(f"Animations will be saved in: {ANIM_DIR}")

    for case_label in ["A", "B", "C", "D", "E"]:
        run_case(
            case_label,
            N=250,
            dx=1.0,
            dt=0.2, # based on the indications in the assignment
            patch_size=20,
            num_steps=250000,
            save_every=1000,
            seed=1,
        )


if __name__ == "__main__":
    main()
