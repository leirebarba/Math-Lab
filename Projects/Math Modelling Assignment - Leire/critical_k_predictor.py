import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
from kuramoto import simulate_kuramoto_scipy, initial_phases


# PART 2

def normal_pdf(x, mu=0.0, sigma=1.0):
    return np.exp(-0.5*((x-mu)/sigma)**2) / (np.sqrt(2*np.pi)*sigma)

def Kc_theory_for_g0(g0):
    # Kuramoto mean-field result: Kc = 2 / (pi * g(0)) - saw the derivaiton in class
    return 2.0 / (np.pi * g0)

# tail_step=300 taken as the example in the assignment pdf 
def r_infty_from_rt(r_t, tail_steps=300):
    tail = r_t[-tail_steps:] if len(r_t) >= tail_steps else r_t
    return float(np.mean(tail)), float(np.std(tail, ddof=1))

def strogatz_integral_equation_residual(r, K, g):
    """
    Residual of:
      1 = K * ∫_{-pi/2}^{pi/2} cos^2(theta) * g(K r sin(theta)) dtheta
    so residual(r) = 1 - RHS.
    """
    integrand = lambda th: (np.cos(th)**2) * g(K * r * np.sin(th))
    val, _ = quad(integrand, -np.pi/2, np.pi/2, limit=200)
    return 1.0 - K * val

def strogatz_integral_equation_residual(r, K, g):
    """
    Residual of:
      1 = K * ∫_{-pi/2}^{pi/2} cos^2(theta) * g(K r sin(theta)) dtheta
    so residual(r) = 1 - RHS.
    """
    integrand = lambda th: (np.cos(th)**2) * g(K * r * np.sin(th))
    val, _ = quad(integrand, -np.pi/2, np.pi/2, limit=200)
    return 1.0 - K * val

def r_theory_from_strogatz(K, g, Kc, r_max=0.999):
    """
    For K <= Kc => r=0.
    For K > Kc => solve the implicit equation for r in (0,1).
    """
    if K <= Kc:
        return 0.0

    # We want a bracket [a,b] where residual(a) and residual(b) have opposite signs.
    a = 1e-8
    fa = strogatz_integral_equation_residual(a, K, g)

    # Try increasing b until sign changes (or give up)
    b = 0.05
    fb = strogatz_integral_equation_residual(b, K, g)
    tries = 0
    while fa * fb > 0 and b < r_max and tries < 30:
        b *= 1.5
        fb = strogatz_integral_equation_residual(b, K, g)
        tries += 1

    if fa * fb > 0:
        # Fallback: if root-bracketing fails, return NaN (won't plot)
        return np.nan

    return float(brentq(lambda r: strogatz_integral_equation_residual(r, K, g), a, b, maxiter=200))

# ---------- Part 2 experiment ----------

def part2_rinf_vs_K(
    N=5000,              
    dt=0.02,
    T=80.0,
    tail_steps=300,
    seed_omega=0,
    seed_theta=1
):
    rng = np.random.default_rng(seed_omega)
    omega = rng.normal(0.0, 1.0, N)

    # Use the SAME omega and theta0 for all K so the curve is smoother
    theta0 = initial_phases(N, mode="disperse", seed=seed_theta)

    # Theoretical Kc for standard normal g(ω): g(0)=1/sqrt(2pi)
    g0 = normal_pdf(0.0, 0.0, 1.0)
    Kc = Kc_theory_for_g0(g0)
    print(f"Theoretical Kc (standard normal): {Kc:.4f}")

    Kmin = Kc / 3.0
    Kmax = 3.0 * Kc
    K_values = np.linspace(Kmin, Kmax, 20)

    r_means = []
    r_stds = []

    for K in K_values:
        t, theta_t, r_t = simulate_kuramoto_scipy(theta0, omega, K, dt=dt, T=T)
        r_mean, r_std = r_infty_from_rt(r_t, tail_steps=tail_steps)
        r_means.append(r_mean)
        r_stds.append(r_std)
        print(f"K={K:.3f}  r_inf≈{r_mean:.3f} ± {r_std:.3f}")

    r_means = np.array(r_means)
    r_stds  = np.array(r_stds)

    # --- theoretical curve from Strogatz implicit equation ---
    g = lambda x: normal_pdf(x, 0.0, 1.0)
    r_theory = np.array([r_theory_from_strogatz(K, g=g, Kc=Kc) for K in K_values])

    # --- plot ---
    plt.figure(figsize=(7,5))
    plt.errorbar(K_values, r_means, yerr=r_stds, fmt='o', capsize=3, label='Numerical (mean ± std)')
    plt.plot(K_values, r_theory, '-', label='Theory (implicit eq.)')
    plt.axvline(Kc, linestyle='--', label=f'Theory Kc ≈ {Kc:.3f}')
    plt.xlabel("Coupling strength K")
    plt.ylabel(r"Asymptotic order parameter $r_\infty$")
    plt.title(r"$r_\infty$ vs $K$ (Kuramoto)")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return K_values, r_means, r_stds, r_theory, Kc


# Run Part 2
K_values, r_means, r_stds, r_theory, Kc = part2_rinf_vs_K()


# Computing plot and theoretical Kc for different variances of the normal distribution
def run_part2_for_sigma(
    sigma,
    N=5000,
    dt=0.02,
    T=80.0,
    tail_steps=300,
    seed_omega=0,
    seed_theta=1,
    num_K=20
):
    """
    Runs Part 2 sweep for ONE sigma (Normal(0, sigma^2)):
      - computes Kc from g(0)
      - sweeps 20 K values in [Kc/3, 3Kc]
      - for each K runs simulate_kuramoto_scipy(...)
      - estimates r_infty as mean(last 300 steps) and std(last 300 steps)
    Returns: K_values, r_means, r_stds, Kc
    """
    rng = np.random.default_rng(seed_omega)

    # sample omega ~ N(0, sigma^2)
    omega = rng.normal(0.0, sigma, N)

    # fixed initial condition (same for all K, so curves are smooth)
    theta0 = initial_phases(N, mode="disperse", seed=seed_theta)

    # theory Kc = 2 / (pi * g(0)), and for Normal(0, sigma^2): g(0)=1/(sqrt(2pi)*sigma)
    g0 = 1.0 / (np.sqrt(2*np.pi) * sigma)
    Kc = Kc_theory_for_g0(g0)

    # 20 K values in [Kc/3, 3Kc]
    K_values = np.linspace(Kc/3.0, 3.0*Kc, 20)

    r_means = []
    r_stds = []

    for K in K_values:
        t, theta_t, r_t = simulate_kuramoto_scipy(theta0, omega, K, dt=dt, T=T)
        r_mean, r_std = r_infty_from_rt(r_t, tail_steps=tail_steps)
        r_means.append(r_mean)
        r_stds.append(r_std)

    return K_values, np.array(r_means), np.array(r_stds), Kc


def main():
    # Choose the sigma values you want to compare
    sigmas = [0.5, 1.0, 2.0]

    # Simulation parameters (tune if needed)
    N = 200
    dt = 0.02
    T = 80.0
    tail_steps = 300

    plt.figure(figsize=(7, 5))

    for sigma in sigmas:
        K_values, r_means, r_stds, Kc = run_part2_for_sigma(
            sigma=sigma,
            N=N,
            dt=dt,
            T=T,
            tail_steps=tail_steps,
            seed_omega=0,   # keep fixed for fair comparison
            seed_theta=1
        )

        # Numerical results with error bars
        plt.errorbar(
            K_values, r_means, yerr=r_stds,
            fmt='o', capsize=3,
            label=f"Numerical (sigma={sigma}, Kc≈{Kc:.2f})"
        )

        # Optional: overlay theory curve using your Strogatz solver
        # (uses the same g(x) = Normal(0, sigma^2) as the simulation)
        g_pdf = lambda x, s=sigma: np.exp(-0.5*(x/s)**2) / (np.sqrt(2*np.pi)*s)
        r_theory = np.array([r_theory_from_strogatz(K, g=g_pdf, Kc=Kc) for K in K_values])
        plt.plot(K_values, r_theory, '-', label=f"Theory (sigma={sigma})")

        # Optional: mark Kc for each sigma (can clutter a bit with 3 lines)
        plt.axvline(Kc, linestyle='--', alpha=0.4)

    plt.xlabel("Coupling strength K")
    plt.ylabel(r"Asymptotic order parameter $r_\infty$")
    plt.title(r"$r_\infty$ vs $K$ for different $g(\omega)=\mathcal{N}(0,\sigma^2)$")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

