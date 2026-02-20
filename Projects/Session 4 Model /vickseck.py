import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import scipy.spatial.distance
import matplotlib.animation as animation
from matplotlib.axes import Axes
from IPython.display import HTML, display 
# make sure Python can find your package/module
sys.path.append(".")

# change limit so animation is shown
import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 50  # in MB

def init_particles_plot(ax_plane, xy, box_size, tail_len):

    # ---- build tail tensor (2, N, TAIL_LEN) ----
    xy_tail = np.repeat(xy[:, :, np.newaxis], tail_len, axis=2)

    # ---- tail (small grey dots) ----
    (plt_particles,) = ax_plane.plot(
        xy_tail[0].flatten(),
        xy_tail[1].flatten(),
        linestyle="",
        marker=".",
        markersize=3,
        color="grey"
    )

    # ---- current positions (bigger black circles) ----
    (plt_current,) = ax_plane.plot(
        xy[0],
        xy[1],
        linestyle="",
        marker="o",
        markersize=4,
        color="black"
    )

    ax_plane.set_xlim(0, box_size)
    ax_plane.set_ylim(0, box_size)
    ax_plane.set_aspect("equal")

    return xy_tail, plt_particles, plt_current

def initialize_particles(num_boids, box_size):
    theta = np.random.uniform(0, 2 * np.pi, num_boids)
    xy = np.random.uniform(0, box_size, (2, num_boids))
    return xy, theta


num_boids = 100
box_size = 10.0
radius_interaction = 1.0
noise = 0.2

# Initialize system
xy, theta = initialize_particles(num_boids, box_size)

# Prepare new heading array
theta_new = np.zeros_like(theta)

# Loop over each particle
for i in range(num_boids):
    # Initialize list to store neighbor headings
    ls_theta_neighbors = []
    # Loop over all other particles to find neighbors
    for j in range(num_boids):
        # Skip self comparison
        if i == j:
            continue
        # Compute distance between particle i and j
        dist = np.linalg.norm(xy[:, i] - xy[:, j])
        # If distance is less than radius,
        # add the heading of particle j to the list of neighbors
        if dist <= radius_interaction:
            ls_theta_neighbors.append(theta[j])
    # After finding neighbors
    # Compute the average heading...
    if len(ls_theta_neighbors) > 0:
     theta_new[i] = np.mean(ls_theta_neighbors)
    else:
     theta_new[i] = theta[i]   # keep previous direction
    # ...and add noise
    theta_new[i] += noise * np.pi * np.random.uniform(-1, 1)

    def vicsek_equations(
    xy: np.ndarray,
    theta: np.ndarray,
    noise: float = 0.1,
    box_size: float = 25,
    dt: float = 1,
    radius_interaction: float = 1,
    v0: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
        # ---- 1. Distance matrix ----
        # xy is shape (2, N), so transpose to (N, 2)
        d_matrix = scipy.spatial.distance.pdist(xy.T)
        d_matrix = scipy.spatial.distance.squareform(d_matrix)

        # Boolean neighbor matrix (includes itself)
        neighbors = d_matrix <= radius_interaction

        # ---- 2. Compute average direction (vectorized) ----
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Sum over neighbors
        sum_sin = neighbors @ sin_theta
        sum_cos = neighbors @ cos_theta

        # Number of neighbors for each particle
        count = neighbors.sum(axis=1)

        # Average direction
        mean_sin = sum_sin / count
        mean_cos = sum_cos / count

        theta_avg = np.arctan2(mean_sin, mean_cos)

        # ---- 3. Add noise ----
        num_boids = theta.shape[0]
        noise_arr = noise * (np.random.uniform(size=num_boids) - 0.5)
        theta_new = theta_avg + noise_arr

        # Keep angles in [0, 2π]
        theta_new = np.mod(theta_new, 2 * np.pi)

        # ---- 4. Update positions ----
        v = v0 * np.array([np.cos(theta_new), np.sin(theta_new)])
        xy_new = xy + dt * v

        # ---- 5. Periodic boundary conditions ----
        xy_new = np.mod(xy_new, box_size)

        return xy_new, theta_new
    
def vicsek_order_parameter(theta: np.ndarray, v0: float = 0.03) -> float:
    """
    Normalized order parameter φ(t).
    φ = |mean velocity| / v0
    """
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    avg_vx = np.mean(vx)
    avg_vy = np.mean(vy)

    phi = np.sqrt(avg_vx**2 + avg_vy**2) / v0
    return float(phi)

# First visualization
num_boids = 100
box_size = 10.0
eta = 0.2
radius_interaction = 1.0
dt = 1
v0 = 0.03

xy, theta = initialize_particles(num_boids, box_size)

T = 300
phis = []

for t in range(T):
    xy, theta = vicsek_equations(
        xy, theta,
        noise=eta,
        box_size=box_size,
        dt=dt,
        radius_interaction=radius_interaction,
        v0=v0
    )
    phis.append(vicsek_order_parameter(theta, v0=v0))

plt.plot(phis)
plt.xlabel("time step")
plt.ylabel("order parameter φ(t)")
plt.title(f"Vicsek order parameter, eta={eta}")
plt.show()

# mean phi values 
num_boids = 100
box_size = 10.0
radius_interaction = 1.0
dt = 1
v0 = 0.03
T = 300

def run_phi_mean(eta, T=400, burn_in=100):
    xy, theta = initialize_particles(num_boids, box_size)
    phis = []
    for t in range(T):
        xy, theta = vicsek_equations(xy, theta, noise=eta, box_size=box_size,
                                     dt=dt, radius_interaction=radius_interaction, v0=v0)
        if t >= burn_in:
            phis.append(vicsek_order_parameter(theta, v0=v0))
    return float(np.mean(phis))

print("mean phi low noise (eta=0.05):", run_phi_mean(0.05))
print("mean phi high noise (eta=2.0):", run_phi_mean(2.0))

# Order parameter vs noise Plot 
def simulate_mean_phi(N, rho=4.0, eta=1.0, T=400, burn_in=100, radius_interaction=1.0, dt=1, v0=0.03):
    L = np.sqrt(N / rho)
    xy, theta = initialize_particles(N, L)

    phis = []
    for t in range(T):
        xy, theta = vicsek_equations(xy, theta, noise=eta, box_size=L,
                                     dt=dt, radius_interaction=radius_interaction, v0=v0)
        if t >= burn_in:
            phis.append(vicsek_order_parameter(theta, v0=v0))
    return float(np.mean(phis)), L

etas = np.linspace(0, 5, 10)

for N in [40, 100]:
    rho = 4.0
    meanphis = []
    L_used = None
    for eta in etas:
        mphi, L_used = simulate_mean_phi(N, rho=rho, eta=eta)
        meanphis.append(mphi)
    plt.plot(etas, meanphis, marker='o', linestyle='-', label=f"N={N}, L={L_used:.1f}")

plt.xlabel("Noise (η)")
plt.ylabel("Order parameter φ")
plt.title("Order parameter vs Noise (density=4.0)")
plt.legend()
plt.show()

def mean_phi_for_density(
    rho: float,
    L: float = 10.0,
    eta: float = 2.0,
    T: int = 400,
    burn_in: int = 100,
    radius_interaction: float = 1.0,
    dt: float = 1.0,
    v0: float = 0.03,
):
    # N = rho * L^2  (must be an integer)
    N = int(round(rho * L**2))
    N = max(N, 2)  # safety

    xy, theta = initialize_particles(N, L)

    phis = []
    for t in range(T):
        xy, theta = vicsek_equations(
            xy, theta,
            noise=eta,
            box_size=L,
            dt=dt,
            radius_interaction=radius_interaction,
            v0=v0
        )
        if t >= burn_in:
            phis.append(vicsek_order_parameter(theta, v0=v0))

    return float(np.mean(phis)), N

# plot order parameter vs density
eta = 2.0      # fixed noise (disordered side)
L = 10.0       # fixed box size (use 20.0 if your laptop can handle it)
rhos = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]

phis = []
Ns = []

for rho in rhos:
    mphi, N = mean_phi_for_density(rho, L=L, eta=eta, T=400, burn_in=100)
    phis.append(mphi)
    Ns.append(N)
    print(f"rho={rho:>4}, N={N:>4}, mean_phi={mphi:.3f}")

plt.figure()
plt.plot(rhos, phis, marker="o")
plt.xlabel("Density ρ = N/L²")
plt.ylabel("Order parameter φ (time-avg)")
plt.title(f"Order parameter vs Density (eta={eta}, L={L})")
plt.show()

def run_simulation(dt: float = 1.0):

    num_boids = 100
    noise_eta = 1
    box_size = 10.0
    radius_interaction = 1.0
    v0 = 0.09

    TAIL_LEN = 20

    xy, theta = initialize_particles(num_boids, box_size)

    fig, ax = plt.subplots(figsize=(6, 6))

    # ---- initialize tail + artists ----
    xy_tail, plt_particles, plt_current = init_particles_plot(
        ax, xy, box_size, TAIL_LEN
    )

    def update(frame):
        nonlocal xy, theta, xy_tail

        # ---- evolve system ----
        xy, theta = vicsek_equations(
            xy,
            theta,
            noise=noise_eta,
            box_size=box_size,
            dt=dt,
            radius_interaction=radius_interaction,
            v0=v0,
        )

        # ---- shift tail history ----
        xy_tail = np.roll(xy_tail, shift=-1, axis=2)

        # write newest positions in last slice
        xy_tail[:, :, -1] = xy

        # update tail artist
        plt_particles.set_data(
            xy_tail[0].flatten(),
            xy_tail[1].flatten()
        )

        # update head positions
        plt_current.set_data(xy[0], xy[1])

        ax.set_title(f"Vicsek Model | t={frame}")

        return plt_particles, plt_current

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=400,
        interval=50,
        blit=True
    )

    plt.close(fig)   # prevents the “static first frame”
    return HTML(ani.to_jshtml())

display(run_simulation())



