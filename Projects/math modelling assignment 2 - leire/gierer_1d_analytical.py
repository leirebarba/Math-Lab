from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Analytical part 

#1 - Plotting Turing Space 
#2 - Sturm-Liouville on (0,L) with Neumann BCs (boundary conditions) 
#3- Unstable spatial modes and leading mode for given values 


#For figure saving: 
ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(exist_ok=True)


def save_fig(filename: str):
    out = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out}")


# Model essential funcs - model build up 

def gm_equilibrium(a: float, b: float):
    """Homogeneous steady state for Gierer-Meinhardt.

    f(u,v) = a - b u + u^2/v
    g(u,v) = u^2 - v
    """
    u_star = (a + 1.0) / b
    v_star = u_star ** 2
    return u_star, v_star


def gm_jacobian(a: float, b: float):
    """Jacobian entries evaluated at the homogeneous steady state."""
    # Simplified closed forms from the webpage/theory
    fu = 2.0 * b / (a + 1.0) - b
    fv = - (b / (a + 1.0)) ** 2
    gu = 2.0 * (a + 1.0) / b
    gv = -1.0
    return fu, fv, gu, gv

def turing_test(a, b, d):
    """Necessary Turing conditions for D1=1, D2=d.

    Works with scalars or NumPy arrays.
    """
    fu, fv, gu, gv = gm_jacobian(a, b)
    Delta = fu * gv - fv * gu

    cond1 = (fu + gv) < 0
    cond2 = Delta > 0
    cond3 = (gv + d * fu) > 2.0 * np.sqrt(d * np.maximum(Delta, 0.0))

    return cond1 & cond2 & cond3


def describe_local_stability(a: float, b: float, d: float):
    fu, fv, gu, gv = gm_jacobian(a, b)
    trJ = fu + gv
    detJ = fu * gv - fv * gu
    lhs = gv + d * fu
    rhs = 2.0 * np.sqrt(max(d * detJ, 0.0))

    return {
        "fu": fu,
        "fv": fv,
        "gu": gu,
        "gv": gv,
        "trJ": trJ,
        "detJ": detJ,
        "lhs_diffusion": lhs,
        "rhs_diffusion": rhs,
        "turing_necessary": bool(turing_test(a, b, d)),
    }


# Sturm-Liouville

def neumann_eigenvalue_1d(n: int, L: float):
    """Laplacian eigenvalue on (0,L) with Neumann BCs.

    phi_n(x) = cos(n*pi*x/L), lambda_n = -(n*pi/L)^2
    """
    return - (n * np.pi / L) ** 2


def mode_matrix(a: float, b: float, d: float, gamma: float, L: float, n: int):
    fu, fv, gu, gv = gm_jacobian(a, b)
    lam = neumann_eigenvalue_1d(n, L)
    A_n = np.array([
        [gamma * fu + lam * 1.0, gamma * fv],
        [gamma * gu, gamma * gv + lam * d],
    ], dtype=float)
    return A_n, lam

def analyze_modes_1d(a: float, b: float, d: float, gamma: float, L: float, n_max: int = 30):
    """Compute temporal eigenvalues sigma^(n) for each spatial mode n."""
    rows = []
    for n in range(n_max + 1):
        A_n, lam = mode_matrix(a, b, d, gamma, L, n)
        eigvals = np.linalg.eigvals(A_n)
        growth = np.max(eigvals.real)
        rows.append({
            "n": n,
            "lambda_n": lam,
            "sigma_1": eigvals[0],
            "sigma_2": eigvals[1],
            "max_real_part": growth,
            "unstable": growth > 0,
        })
    return rows


def unstable_modes(rows):
    vals = [row for row in rows if row["unstable"]]
    vals.sort(key=lambda r: r["max_real_part"], reverse=True)
    return vals


def leading_mode(rows):
    vals = unstable_modes(rows)
    return vals[0] if vals else None

# Plots
def plot_turing_space_section(b: float = 1.0, a_min: float = 0.0, a_max: float = 1.0,
                              d_min: float = 0.0, d_max: float = 100.0,
                              na: int = 500, nd: int = 500,
                              filename: str = "01_gm1d_turing_space_b1.png"):
    arr_a = np.linspace(a_min, a_max, na)
    arr_d = np.linspace(d_min, d_max, nd)
    mesh_a, mesh_d = np.meshgrid(arr_a, arr_d)
    mask = turing_test(mesh_a, b, mesh_d)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.contourf(mesh_a, mesh_d, mask.astype(float), levels=[-0.5, 0.5, 1.5])
    ax.scatter([0.4, 0.4], [20, 30], marker="o")
    ax.text(0.41, 20.5, "(0.4, 20)", fontsize=9)
    ax.text(0.41, 30.5, "(0.4, 30)", fontsize=9)
    ax.set_xlabel("a")
    ax.set_ylabel("d")
    ax.set_title("Gierer-Meinhardt Turing space section (b = 1)")
    save_fig(filename)

def plot_growth_rates_for_assignment_cases(a: float = 0.4, b: float = 1.0, gamma: float = 1.0,
                                           L: float = 40.0, n_max: int = 20,
                                           filename: str = "02_gm1d_growth_rates_assignment_cases.png"):
    rows_d30 = analyze_modes_1d(a, b, d=30.0, gamma=gamma, L=L, n_max=n_max)
    rows_d20 = analyze_modes_1d(a, b, d=20.0, gamma=gamma, L=L, n_max=n_max)

    n_vals = [r["n"] for r in rows_d30]
    g30 = [r["max_real_part"] for r in rows_d30]
    g20 = [r["max_real_part"] for r in rows_d20]

    plt.figure(figsize=(8, 4.5))
    plt.plot(n_vals, g30, marker="o", label="d = 30")
    plt.plot(n_vals, g20, marker="s", label="d = 20")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Spatial mode n")
    plt.ylabel("max Re(sigma^(n))")
    plt.title("Growth rate of each 1D Neumann mode")
    plt.legend()
    save_fig(filename)

def plot_assignment_case_spectra(a: float = 0.4, b: float = 1.0, gamma: float = 1.0,
                                 L: float = 40.0, d: float = 30.0, n_max: int = 20,
                                 filename: str = "03_gm1d_growth_rates_d30.png"):
    rows = analyze_modes_1d(a, b, d=d, gamma=gamma, L=L, n_max=n_max)
    n_vals = [r["n"] for r in rows]
    g_vals = [r["max_real_part"] for r in rows]

    plt.figure(figsize=(8, 4.5))
    plt.plot(n_vals, g_vals, marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Spatial mode n")
    plt.ylabel("max Re(sigma^(n))")
    plt.title(f"Mode growth rates for a=0.4, b=1, d={d}, gamma=1, L=40")
    save_fig(filename)

# Required results: 

def print_case_summary(a: float, b: float, d: float, gamma: float, L: float, n_max: int = 20):
    print("\n" + "=" * 72)
    print(f"CASE: a={a}, b={b}, d={d}, gamma={gamma}, L={L}")
    print("=" * 72)

    u_star, v_star = gm_equilibrium(a, b)
    print(f"Equilibrium: u* = {u_star:.6f}, v* = {v_star:.6f}")

    info = describe_local_stability(a, b, d)
    print(f"fu = {info['fu']:.6f}, fv = {info['fv']:.6f}, gu = {info['gu']:.6f}, gv = {info['gv']:.6f}")
    print(f"tr(J) = {info['trJ']:.6f}")
    print(f"det(J) = {info['detJ']:.6f}")
    print(f"gv + d fu = {info['lhs_diffusion']:.6f}")
    print(f"2 sqrt(d detJ) = {info['rhs_diffusion']:.6f}")
    print(f"Necessary Turing conditions satisfied? {info['turing_necessary']}")

    rows = analyze_modes_1d(a, b, d, gamma, L, n_max=n_max)
    unstable = unstable_modes(rows)

    print("\nSpatial spectrum (Neumann BCs): lambda_n = -(n pi / L)^2")
    print("Unstable modes:")
    if not unstable:
        print("  None")
    else:
        for row in unstable:
            print(
                f"  n={row['n']:2d}, lambda_n={row['lambda_n']:.6f}, "
                f"max Re(sigma^(n))={row['max_real_part']:.6f}"
            )

    lead = leading_mode(rows)
    if lead is None:
        print("Leading mode: none (no instability)")
    else:
        print(
            f"Leading mode: n={lead['n']} with growth rate "
            f"{lead['max_real_part']:.6f}"
        )

# Main exercise - Parameters given for analytical part 

def main():
    # Assignment values
    gamma = 1.0
    a = 0.4
    b = 1.0
    L = 40.0

    print(f"Figures will be saved in: {FIG_DIR}")

    # Graph 1: Turing space section for b = 1, with assignment points marked
    plot_turing_space_section(b=1.0)

    # Graph 2: compare mode growth rates for the two assignment cases
    plot_growth_rates_for_assignment_cases(a=a, b=b, gamma=gamma, L=L, n_max=20)

    # Graph 3 and 4: one separate graph for each assignment case
    plot_assignment_case_spectra(a=a, b=b, gamma=gamma, L=L, d=30.0,
                                 n_max=20, filename="03_gm1d_growth_rates_d30.png")
    plot_assignment_case_spectra(a=a, b=b, gamma=gamma, L=L, d=20.0,
                                 n_max=20, filename="04_gm1d_growth_rates_d20.png")

    # Printed analytic conclusions for the exact assignment cases
    print_case_summary(a=a, b=b, d=30.0, gamma=gamma, L=L, n_max=20)
    print_case_summary(a=a, b=b, d=20.0, gamma=gamma, L=L, n_max=20)


if __name__ == "__main__":
    main()


