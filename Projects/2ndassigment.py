import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def laplacian_1d(values, dx):
    lap = -2 * values
    lap += np.roll(values, shift=1)
    lap += np.roll(values, shift=-1)
    return lap / dx**2

def gierer_meinhardt_pde(t, uv, gamma=1, a=0.40, b=1.00, d=20, dx=0.5):
    u, v = uv
    lu = laplacian_1d(u, dx)
    lv = laplacian_1d(v, dx)

    f = a - b * u + (u**2) / v
    g = u**2 - v

    du_dt = lu + gamma * f
    dv_dt = d * lv + gamma * g

    return np.array([du_dt, dv_dt])

np.random.seed(3)

length = 40
dx = 1.0
num_points = int(length / dx)

gamma = 1.0
a = 0.40
b = 1.00
d = 30.0

u_star = (a + 1) / b
v_star = ((a + 1) / b) ** 2

uv = np.ones((2, num_points))
uv[0] *= u_star
uv[1] *= v_star
uv += 0.01 * np.random.randn(2, num_points)

num_iter = 50000
dt = 0.01

print("Running with d =", d)
frames = []
save_every = 500
for step in range(num_iter):
    dudt, dvdt = gierer_meinhardt_pde(0, uv, gamma=gamma, a=a, b=b, d=d, dx=dx)

    uv[0] = uv[0] + dt * dudt
    uv[1] = uv[1] + dt * dvdt

    uv[0, 0] = uv[0, 1]
    uv[0, -1] = uv[0, -2]
    uv[1, 0] = uv[1, 1]
    uv[1, -1] = uv[1, -2]
    if step % save_every == 0:
        frames.append(uv[1].copy())

x = np.linspace(0, length, num_points)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, uv[1])
ax.set_xlabel("x")
ax.set_ylabel("v(x)")
ax.set_title("Gierer-Meinhardt in 1D")
plt.show()
plt.close()


fig, ax = plt.subplots(figsize=(7, 4))
line, = ax.plot(x, frames[0])

ax.set_xlabel("x")
ax.set_ylabel("v(x)")
ax.set_title(f"Gierer-Meinhardt 1D animation, d = {d}")

all_min = min(frame.min() for frame in frames)
all_max = max(frame.max() for frame in frames)
ax.set_ylim(all_min - 0.01, all_max + 0.01)

def update(frame_index):
    line.set_ydata(frames[frame_index])
    return line,

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=100,
    blit=True
)

ani.save(f"gierer_1d_d_{int(d)}.gif", writer="pillow", fps=10)

plt.show()
plt.close()

def laplacian_2d(U, dx):
    lap = -4 * U
    lap += np.roll(U, 1, axis=0)
    lap += np.roll(U, -1, axis=0)
    lap += np.roll(U, 1, axis=1)
    lap += np.roll(U, -1, axis=1)
    return lap / dx**2

def gierer_meinhardt_2d(uv, gamma, a, b, d, dx):
    u, v = uv

    lu = laplacian_2d(u, dx)
    lv = laplacian_2d(v, dx)

    v_safe = np.maximum(v, 1e-8)
    f = a - b*u + (u**2)/v_safe
    g = u**2 - v

    du_dt = lu + gamma*f
    dv_dt = d*lv + gamma*g

    return np.array([du_dt, dv_dt])

N1 = 20
N2 = 50
dx = 1.0

u_star = (a+1)/b
v_star = ((a+1)/b)**2

uv = np.ones((2, N1, N2))
uv[0] *= u_star
uv[1] *= v_star

uv += 0.01 * np.random.randn(2, N1, N2)

num_iter = 100000
dt = 0.001

def apply_neumann_2d(U):
    U[0, :] = U[1, :]
    U[-1, :] = U[-2, :]
    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]

for step in range(num_iter):
    dudt, dvdt = gierer_meinhardt_2d(uv, gamma, a, b, d, dx)

    uv[0] += dt * dudt
    uv[1] += dt * dvdt

    apply_neumann_2d(uv[0])
    apply_neumann_2d(uv[1])

V = uv[1] - v_star

plt.figure(figsize=(8,4))
plt.imshow(V, cmap="viridis", aspect="auto", vmin=-0.02, vmax=0.02)
plt.colorbar()
plt.title(f"Gierer-Meinhardt 2D, d={d}")
plt.tight_layout()
plt.show()