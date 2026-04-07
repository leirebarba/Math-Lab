import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def laplacian_2d_periodic(U, dx):
    lap = -4 * U
    lap += np.roll(U, 1, axis=0)
    lap += np.roll(U, -1, axis=0)
    lap += np.roll(U, 1, axis=1)
    lap += np.roll(U, -1, axis=1)
    return lap / dx**2

def gray_scott_rhs(uv, D1, D2, F, k, dx):
    u, v = uv

    lu = laplacian_2d_periodic(u, dx)
    lv = laplacian_2d_periodic(v, dx)

    reaction = u * v**2

    du_dt = D1 * lu - reaction + F * (1 - u)
    dv_dt = D2 * lv + reaction - (F + k) * v

    return np.array([du_dt, dv_dt])

def initialize_gray_scott(N=250, seed=3):
    np.random.seed(seed)

    u = np.ones((N, N))
    v = np.zeros((N, N))

    c = N // 2
    half = 10

    u[c-half:c+half, c-half:c+half] = 0.5
    v[c-half:c+half, c-half:c+half] = 0.5

    noise_u = 0.05 * np.random.randn(20, 20)
    noise_v = 0.05 * np.random.randn(20, 20)

    u[c-half:c+half, c-half:c+half] += noise_u
    v[c-half:c+half, c-half:c+half] += noise_v

    return np.array([u, v])

def run_gray_scott(F, k, label, seed=3):
    N = 250
    dx = 1.0
    dt = 2.0
    num_iter = 25000          # because t = 5e4 and dt = 2
    save_every = 100          # every 100 steps means every t = 200

    D1 = 0.1
    D2 = 0.05

    uv = initialize_gray_scott(N=N, seed=seed)
    frames = []

    print(f"Running Gray-Scott {label}: F={F}, k={k}")

    for step in range(num_iter):
        dudt, dvdt = gray_scott_rhs(uv, D1, D2, F, k, dx)

        uv[0] += dt * dudt
        uv[1] += dt * dvdt

        # keep values in a reasonable range
        uv[0] = np.clip(uv[0], 0, 1.5)
        uv[1] = np.clip(uv[1], 0, 1.5)

        if step % save_every == 0:
            frames.append(uv[1].copy())

    # final figure
    plt.figure(figsize=(6, 6))
    plt.imshow(uv[1], cmap="viridis")
    plt.colorbar()
    plt.title(f"Gray-Scott {label}: F={F}, k={k}")
    plt.tight_layout()
    plt.savefig(f"gray_scott_{label}.png", dpi=300)
    plt.close()

    # animation
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frames[0], cmap="viridis", animated=True)
    ax.set_title(f"Gray-Scott {label}: F={F}, k={k}")

    def update(frame_index):
        im.set_array(frames[frame_index])
        return [im]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=80,
        blit=True
    )

    ani.save(f"gray_scott_{label}.gif", writer="pillow", fps=10)
    plt.close()

    return uv, frames

run_gray_scott(0.040, 0.060, "A")
run_gray_scott(0.014, 0.047, "B")
run_gray_scott(0.062, 0.065, "C")
run_gray_scott(0.078, 0.061, "D")
run_gray_scott(0.082, 0.059, "E")