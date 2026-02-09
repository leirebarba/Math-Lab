import numpy as np
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import available_events
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.pyplot as plt

def fit_nag(t, state, i_app, gamma, alpha, epsilon):
    # state represents the current values of the variables (e.g., x, y)

    v,w = state
    # params are any additional parameters needed to compute the derivatives
    # each of them is separated by commas

    f = v * (1-v)*(v-alpha)

    dvdt = (f-w+i_app)/epsilon
    dwdt = v-gamma*w

    dstate_dt = (dvdt,dwdt)

    return dstate_dt  # this should be a sequence of the same length as `state`

# Define parameters
i_app, gamma, alpha, epsilon= 0.5, 0.5, 0.1, 0.01
# Define initial condition
v0, w0 = 0, 3
# Define time span and evaluation points
t_span = (0, 20) # Start at t=0 and end at t=20
t_eval = np.linspace(t_span[0], t_span[1], 2000)  # Return 2000 points between t=0 and t=20
# Solve the IVP
sol = solve_ivp(fit_nag, t_span, [v0, w0], args=(i_app, gamma, alpha, epsilon), t_eval=t_eval)
# sol.y[0] will give you x(t) and sol.y[1] will give you y(t)

print(sol)

# basic plot 

fig, av = plt.subplots(figsize=(8, 4))
for v0, w0 in [(0, 3), (0, 1.5), (2, 0)]:
    sol = solve_ivp(fit_nag, [0, 20], [v0, w0], args=(i_app, gamma, alpha, epsilon), t_eval=np.linspace(0, 20, 2000))
    av.plot(sol.y[0], sol.y[1], label=f"IC: ({v0}, {w0})")  # Plot y vs. x
    av.plot(v0, w0, 'o', color='black')  # Mark the initial condition
plt.xlabel("v")
plt.ylabel("w")
plt.legend()
plt.show()

i_app, gamma, alpha, epsilon= 0.5, 0.5, 0.1, 0.01

# We need to define a function that represents the system of
# equations for the nullclines
def fit_nag_fixed(vw: np.ndarray) -> np.ndarray:
    return fit_nag(0, vw, i_app, gamma, alpha, epsilon)

equilibrium_point = fsolve(fit_nag_fixed, [0.2, 0.2])  # Initial guess for the equilibrium point
print(f"Equilibrium point: v={equilibrium_point[0]:.2f}, w={equilibrium_point[1]:.2f}")

# parameters (same as your solve)
i_app, gamma, alpha, epsilon = 0.5, 0.5, 0.1, 0.01

v_star, w_star = equilibrium_point  # equilibrium from fsolve

def fprime(v, alpha):
    return -3*v**2 + 2*(1+alpha)*v - alpha

J = np.array([
    [fprime(v_star, alpha)/epsilon,   -1/epsilon],
    [1.0,                            -gamma]
], dtype=float)

eigvals = np.linalg.eigvals(J)

print("Equilibrium:", (v_star, w_star))
print("Jacobian:\n", J)
print("Eigenvalues:", eigvals)

# if fixed point is stabke or unstable 

re = np.real(eigvals)

if np.all(re < 0):
    print("Stable (attracting)")
elif np.any(re > 0) and np.any(re < 0):
    print("Saddle (unstable)")
elif np.any(re > 0):
    print("Unstable (repelling)")
else:
    print("Inconclusive (near zero eigenvalue)")

# Plotting the nullclines and the trajectory
# Define the limits for the grid
v_min, v_max = -1.0, 2.5
w_min, w_max = -1, 4
num_points = 400  # Number of points in the grid

# Create a grid of points
v_values = np.linspace(v_min, v_max, num_points)
w_values = np.linspace(w_min, w_max, num_points)
v_grid, w_grid = np.meshgrid(v_values, w_values)
# meshgrid creates two 2D arrays:
# x_grid and y_grid, where each element (i,j) corresponds to the
# coordinates (x_values[i], y_values[j]) in the phase plane.

# Evaluate the derivatives at each point
dv_dt: np.ndarray
dw_dt: np.ndarray
dv_dt, dw_dt = fit_nag(0, [v_grid, w_grid], i_app, gamma, alpha, epsilon)

# Extract nullcline data:
# Find where dx_dt changes sign (zero crossings)
dvdt_zero_crossings = np.where(np.diff(np.sign(dv_dt), axis=0))
dvdt_nullcline_v = v_grid[dvdt_zero_crossings]
dvdt_nullcline_w = w_grid[dvdt_zero_crossings]

# Extract nullcline data - Find where dy_dt changes sign (zero crossings)
dwdt_zero_crossings = np.where(np.diff(np.sign(dw_dt), axis=1))
dwdt_nullcline_v = v_grid[dwdt_zero_crossings]
dwdt_nullcline_w = w_grid[dwdt_zero_crossings]

plt.figure(figsize=(8, 5))
plt.plot(sol.y[0], sol.y[1], label="trajectory")     # your phase curve
plt.scatter(dvdt_nullcline_v, dvdt_nullcline_w, s=1, label="dv/dt = 0 (v-nullcline)", color = "red")
plt.scatter(dwdt_nullcline_v, dwdt_nullcline_w, s=1, label="dw/dt = 0 (w-nullcline)", color = "orange" )

plt.plot(equilibrium_point[0],equilibrium_point[1], "k*", markersize=12, label="equilibrium")

plt.xlabel("v")
plt.ylabel("w")
plt.legend()
plt.grid(True)
plt.show()

# Animation of the trajectory in the phase plane

fig, ax = plt.subplots(figsize=(8, 5))

# background: nullclines
ax.scatter(dvdt_nullcline_v, dvdt_nullcline_w, s=1, color="red", label="dv/dt = 0")
ax.scatter(dwdt_nullcline_v, dwdt_nullcline_w, s=1, color="orange", label="dw/dt = 0")

# equilibrium
ax.plot(equilibrium_point[0], equilibrium_point[1], "k*", markersize=12, label="equilibrium")

# animated objects
traj_line, = ax.plot([], [], lw=2, label="trajectory")
point, = ax.plot([], [], "o", markersize=6, label="current state")

ax.set_xlim(v_min, v_max)
ax.set_ylim(w_min, w_max)
ax.set_xlabel("v")
ax.set_ylabel("w")
ax.grid(True)
ax.legend(loc="best")

step = 5
frames = range(0, len(sol.t), step)

def init():
    traj_line.set_data([], [])
    point.set_data([], [])
    return traj_line, point

def update(frame):
    traj_line.set_data(sol.y[0, :frame+1], sol.y[1, :frame+1])

    # IMPORTANT FIX: point needs sequences (lists), not scalars
    point.set_data([sol.y[0, frame]], [sol.y[1, frame]])
    return traj_line, point

anim = FuncAnimation(fig, update, frames=frames, init_func=init, interval=20, blit=True)

plt.show()

