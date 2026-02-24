import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

# PART 1

def initial_phases(N, mode="disperse", frac=0.1, seed=None):
    rng = np.random.default_rng(seed)

    if mode == "disperse":
        return rng.uniform(0.0, 2*np.pi, size=N)

    elif mode == "concentrated":
        return rng.uniform(0.0, 2*np.pi*frac, size=N)

    else:
        raise ValueError("mode must be 'disperse' or 'concentrated'")


def order_parameter(theta):
    z = np.mean(np.exp(1j * theta))
    r = np.abs(z)
    psi = np.angle(z)
    return r, psi

def kuramoto_rhs(t, theta, omega, K):
    theta = np.mod(theta, 2*np.pi)
    r, psi = order_parameter(theta)
    return omega + K * r * np.sin(psi - theta)


def simulate_kuramoto_scipy(theta0, omega, K, dt=0.01, T=40.0):
    t_eval = np.arange(0.0, T + dt, dt)

    sol = solve_ivp(
        kuramoto_rhs,
        (0.0, T),
        theta0,
        t_eval=t_eval,
        args=(omega, K),
    )

    theta_t = sol.y
    r_t = np.abs(np.mean(np.exp(1j * theta_t), axis=0))

    return sol.t, theta_t, r_t

# Parameters given by the assignment
N = 100
dt = 0.01
T = 40.0

rng = np.random.default_rng(0)
omega = rng.normal(0.0, 1.0, N)
theta0 = initial_phases(N, mode="disperse", seed=1)

if __name__ == "__main__":
    # subcritical and supercritical K
    K_sub = 1.0 
    K_sup = 7.0   

    t1, theta1, r1 = simulate_kuramoto_scipy(theta0, omega, K_sub, dt, T)
    t2, theta2, r2 = simulate_kuramoto_scipy(theta0, omega, K_sup, dt, T)

    plt.figure()
    plt.plot(t1, r1, label=f"K={K_sub} (subcritical)")
    plt.plot(t2, r2, label=f"K={K_sup} (supercritical)")
    plt.xlabel("t")
    plt.ylabel("r(t)")
    plt.title("Order parameter r(t) for K < Kc and K > Kc")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

    # ANIMATION 

    # Parameters for animation
    N = 100
    dt = 0.02
    coupling_strength = 7.0   
    frames = 300
    interval = 50  # ms between frames

    rng = np.random.default_rng(0)
    omega = rng.normal(0.0, 1.0, N)

    theta = initial_phases(N, mode="disperse", seed=1)  # global state updated per frame

    # Figure set up for phase plane animation
    fig, ax_phase = plt.subplots(figsize=(6, 6))

    ax_phase.set_title("Kuramoto Model")
    ax_phase.set_xlabel("Cos(theta)")
    ax_phase.set_ylabel("Sin(theta)")
    ax_phase.set_xlim(-1.1, 1.1)
    ax_phase.set_ylim(-1.1, 1.1)
    ax_phase.set_aspect("equal")
    ax_phase.grid(True)

    circle = plt.Circle((0, 0), 1, color="lightgray", fill=False)
    ax_phase.add_artist(circle)

    scatter = ax_phase.scatter([], [], s=50, color="blue", alpha=0.25)

    centroid_line, = ax_phase.plot([], [], color="red", linewidth=2)
    centroid_point, = ax_phase.plot([], [], "ro", markersize=8)

    # Optional: show r(t) as text
    r_text = ax_phase.text(-1.05, 1.05, "", fontsize=12, color="red")

    def init():
        scatter.set_offsets(np.zeros((N, 2)))
        centroid_line.set_data([], [])
        centroid_point.set_data([], [])
        r_text.set_text("")
        return [scatter, centroid_line, centroid_point, r_text]

    def update_function(frame: int):
        global theta  # match your style

        # integrate from 0 to dt (one step)
        sol = solve_ivp(
            kuramoto_rhs,          # must be kuramoto_rhs(t, theta, omega, K)
            (0.0, dt),
            theta,
            args=(omega, coupling_strength),
            rtol=1e-6,
            atol=1e-9,
        )
        theta = sol.y[:, -1]
        theta = np.mod(theta, 2*np.pi)

        # update oscillator points
        x = np.cos(theta)
        y = np.sin(theta)
        scatter.set_offsets(np.c_[x, y])

        # centroid (order parameter)
        z = np.mean(np.exp(1j * theta))
        r = np.abs(z)
        cx = np.real(z)
        cy = np.imag(z)

        centroid_line.set_data([0.0, cx], [0.0, cy])
        centroid_point.set_data([cx], [cy])
        r_text.set_text(f"r = {r:.3f}")

        return [scatter, centroid_line, centroid_point, r_text]

    ani = FuncAnimation(
        fig,
        update_function,
        init_func=init,
        frames=frames,
        interval=interval,
        blit=True,
    )

    plt.show()
    display(HTML(ani.to_jshtml()))


