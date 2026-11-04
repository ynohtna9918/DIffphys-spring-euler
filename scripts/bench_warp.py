"""Time one semi-implicit step: NumPy vs Warp (CPU, and CUDA if present).

Run from the repo root:  python -m scripts.bench_warp
"""
import time

import warp as wp

from src import make_chain, step_semi_implicit_euler
from src.warp_sim import WarpModel, step_semi_implicit_warp

DT = 0.001
G = [0.0, -9.81, 0.0]


def time_numpy(n, steps):
    sys = make_chain(n, gravity=G)
    t0 = time.perf_counter()
    for _ in range(steps):
        step_semi_implicit_euler(sys, DT)
    return (time.perf_counter() - t0) / steps


def time_warp(n, steps, device):
    sys = make_chain(n, gravity=G)
    model = WarpModel(sys, device)
    x = wp.array(sys.x, dtype=wp.vec3d, device=device)
    v = wp.array(sys.v, dtype=wp.vec3d, device=device)
    x_out = wp.zeros_like(x)
    v_out = wp.zeros_like(v)
    F = wp.zeros_like(x)
    step_semi_implicit_warp(model, x, v, F, DT, x_out, v_out)   # warm-up: compile + first launch
    wp.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(steps):
        step_semi_implicit_warp(model, x, v, F, DT, x_out, v_out)
        x, x_out = x_out, x                                       # ping-pong the two buffers
        v, v_out = v_out, v
    wp.synchronize_device(device)                                 # wait for the GPU to finish
    return (time.perf_counter() - t0) / steps


def main():
    devices = ["cpu"] + (["cuda:0"] if wp.get_cuda_device_count() > 0 else [])
    print(f"{'particles':>10} {'numpy':>12}" + "".join(f" {'warp ' + d:>14}" for d in devices))
    for n, steps in [(10, 2000), (1_000, 2000), (100_000, 200), (1_000_000, 20)]:
        row = f"{n:>10,} {time_numpy(n, steps) * 1e6:>9.1f} us"
        for d in devices:
            row += f" {time_warp(n, steps, d) * 1e6:>11.1f} us"
        print(row)


if __name__ == "__main__":
    main()
