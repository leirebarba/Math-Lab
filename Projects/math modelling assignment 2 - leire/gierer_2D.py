from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# import part from previous scripts: 
from gierer_1d_analytical import gm_equilibrium, gm_jacobian, turing_test

# for figure saving:
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

#1 - studding turing instability on the omega space
#2 - Compute unstable and leading modes for the given values of a,b,d and gamma
#3- test numerical prediction 


# 2D needed funcs (analytical part) - model build up
def neumann_eigenvalue_2d(nx: int, ny: int, L1: float, L2: float):
    """Neumann Laplacian eigenvalue on (0,L1)x(0,L2).

    phi_{nx,ny}(x,y) = cos(nx*pi*x/L1) cos(ny*pi*y/L2)
    lambda_{nx,ny} = -[(nx*pi/L1)^2 + (ny*pi/L2)^2]
    """
    return -((nx * np.pi / L1) ** 2 + (ny * np.pi / L2) ** 2)



def mode_matrix_2d(a: float, b: float, d: float, gamma: float,
                   L1: float, L2: float, nx: int, ny: int):
    fu, fv, gu, gv = gm_jacobian(a, b)
    lam = neumann_eigenvalue_2d(nx, ny, L1, L2)
    A = np.array([
        [gamma * fu + lam * 1.0, gamma * fv],
        [gamma * gu, gamma * gv + lam * d],
    ], dtype=float)
    return A, lam


def analyze_modes_2d(a: float, b: float, d: float, gamma: float,
                     L1: float, L2: float, nx_max: int = 12, ny_max: int = 12):
    rows = []
    for nx in range(nx_max + 1):
        for ny in range(ny_max + 1):
            A, lam = mode_matrix_2d(a, b, d, gamma, L1, L2, nx, ny)
            eigvals = np.linalg.eigvals(A)
            growth = np.max(eigvals.real)
            rows.append({
                "mode": (nx, ny),
                "lambda": lam,
                "sigma_1": eigvals[0],
                "sigma_2": eigvals[1],
                "max_real_part": growth,
                "unstable": growth > 0,
            })
    return rows

def unstable_modes_2d(rows):
    vals = [row for row in rows if row["unstable"]]
    vals.sort(key=lambda r: r["max_real_part"], reverse=True)
    return vals

def leading_mode_2d(rows):
    unstable = unstable_modes_2d(rows)
    return unstable[0] if unstable else None

# 2D numerical simulation
def laplacian_2d(U: np.ndarray, dx: float):
    U_pad = np.pad(U, ((1, 1), (1, 1)), mode="edge")
    return (
        U_pad[2:, 1:-1] +
        U_pad[:-2, 1:-1] +
        U_pad[1:-1, 2:] +
        U_pad[1:-1, :-2] -
        4.0 * U
    ) / dx**2

# did not use np.roll as it gave issues with the corners and the way we apply neumann BCs, so I implemented a more explicit version of the laplacian with padding and edge mode to handle the boundaries correctly.
# This is in order for the simulations to work properly and correctly 


def apply_neumann_2d(U: np.ndarray):
    U[0, :] = U[1, :]
    U[-1, :] = U[-2, :]
    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]
    return U

def gm_rhs_2d(u: np.ndarray, v: np.ndarray, a: float, b: float,
              d: float, gamma: float, dx: float):
    eps = 1e-12
    lu = laplacian_2d(u, dx)
    lv = laplacian_2d(v, dx)

    f = a - b * u + (u**2) / np.maximum(v, eps)
    g = u**2 - v

    dudt = lu + gamma * f
    dvdt = d * lv + gamma * g
    return dudt, dvdt

def simulate_gm_2d(a=0.4, b=1.0, d=30.0, gamma=1.0,
                   L1=20.0, L2=50.0, dx=1.0, dt=0.01,
                   num_steps=50000, save_every=500,
                   noise_amp=0.01, seed=0):
    rng = np.random.default_rng(seed)

    N1 = int(L1 / dx)
    N2 = int(L2 / dx)

    u_star, v_star = gm_equilibrium(a, b)

    u = u_star * (1 + noise_amp * rng.standard_normal((N1, N2)))
    v = v_star * (1 + noise_amp * rng.standard_normal((N1, N2)))
    apply_neumann_2d(u)
    apply_neumann_2d(v)

    frames = []

    for step in range(num_steps):
        dudt, dvdt = gm_rhs_2d(u, v, a, b, d, gamma, dx)
        u += dt * dudt
        v += dt * dvdt

        apply_neumann_2d(u)
        apply_neumann_2d(v)
        v = np.maximum(v, 1e-10)

        if step % save_every == 0:
            frames.append(v.copy())

    return np.array(frames)

# Plots and animations 
def plot_growth_heatmap(rows, nx_max: int, ny_max: int, filename: str, title: str):
    growth = np.zeros((nx_max + 1, ny_max + 1))
    for row in rows:
        nx, ny = row["mode"]
        growth[nx, ny] = row["max_real_part"]

    plt.figure(figsize=(7, 5))
    plt.imshow(growth.T, origin="lower", aspect="auto")
    plt.colorbar(label="max Re(sigma)")
    plt.xlabel("nx")
    plt.ylabel("ny")
    plt.title(title)
    save_fig(filename)



def plot_mode_spectrum(rows, filename: str, title: str):
    values = [r["max_real_part"] for r in rows]

    plt.figure(figsize=(10, 4.5))
    plt.plot(range(len(values)), values, marker="o", markersize=3, linewidth=1)
    plt.axhline(0.0, linestyle="--")

    # show only a few labels to avoid overlap - based on issues with previous spectrum plot:
    step = max(1, len(rows) // 10)
    tick_positions = list(range(0, len(rows), step))
    tick_labels = [f"{rows[i]['mode']}" for i in tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=45, ha="right")

    plt.ylabel("max Re(sigma)")
    plt.title(title)
    save_fig(filename)

def animate_2d(frames: np.ndarray, filename: str, cmap: str = "viridis"):
    fig, ax = plt.subplots(figsize=(6, 6))
    vmin = np.percentile(frames, 1)
    vmax = np.percentile(frames, 99)
    im = ax.imshow(frames[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

    def update(i):
        im.set_array(frames[i])
        ax.set_title(f"Frame {i+1}/{len(frames)}")
        return (im,)

    ani = FuncAnimation(fig, update, frames=len(frames), blit=True)
    ani.save(animation_path(filename), writer=PillowWriter(fps=10))
    plt.close(fig)

# Main exercise - parameters given for numerical part:
def print_case_summary(a: float, b: float, d: float, gamma: float,
                       L1: float, L2: float, nx_max: int = 12, ny_max: int = 12):
    print("\n" + "=" * 78)
    print(f"2D CASE: a={a}, b={b}, d={d}, gamma={gamma}, L1={L1}, L2={L2}")
    print("=" * 78)

    print(f"Necessary Turing conditions satisfied? {bool(turing_test(a, b, d))}")

    rows = analyze_modes_2d(a, b, d, gamma, L1, L2, nx_max=nx_max, ny_max=ny_max)
    unstable = unstable_modes_2d(rows)
    lead = leading_mode_2d(rows)

    print(f"Number of unstable modes in scanned range: {len(unstable)}")
    if unstable:
        print("Unstable modes:")
        for row in unstable:
            nx, ny = row["mode"]
            print(
                f"  mode=({nx},{ny}), lambda={row['lambda']:.6f}, "
                f"max Re(sigma)={row['max_real_part']:.6f}"
            )
    else:
        print("Unstable modes: none")

    if lead is None:
        print("Leading mode: none")
    else:
        print(
            f"Leading mode: ({lead['mode'][0]},{lead['mode'][1]}) with growth rate "
            f"{lead['max_real_part']:.6f}"
        )

def run_case(d_value: float, nx_max: int = 12, ny_max: int = 12):
    a = 0.4
    b = 1.0
    gamma = 1.0
    L1 = 20.0
    L2 = 50.0

    rows = analyze_modes_2d(a, b, d_value, gamma, L1, L2, nx_max=nx_max, ny_max=ny_max)
    lead = leading_mode_2d(rows)

    print_case_summary(a, b, d_value, gamma, L1, L2, nx_max=nx_max, ny_max=ny_max)

    plot_growth_heatmap(
        rows,
        nx_max,
        ny_max,
        filename=f"05_gm2d_growth_heatmap_d{int(d_value)}.png",
        title=f"2D mode growth heatmap for d={int(d_value)}",
    )

    plot_mode_spectrum(
        rows,
        filename=f"06_gm2d_mode_spectrum_d{int(d_value)}.png",
        title=f"2D spectrum by mode for d={int(d_value)}",
    )

    if lead is not None:
        print(f"Predicted leading mode = {lead['mode']}")
    else:
        print("No instability predicted")

    frames = simulate_gm_2d(
        a=a,
        b=b,
        d=d_value,
        gamma=gamma,
        L1=L1,
        L2=L2,
        dx=1.0,
        dt=0.005,
        num_steps=100000,
        save_every=1000,
        noise_amp=0.001,
        seed=1,
    )
    animate_2d(frames, filename=f"gm_2d_d{int(d_value)}.gif")



def main():
    print(f"Figures will be saved in: {FIG_DIR}")
    print(f"Animations will be saved in: {ANIM_DIR}")

    run_case(30.0)
    run_case(20.0)


if __name__ == "__main__":
    main()

