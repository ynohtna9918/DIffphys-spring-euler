"""Check that Warp is installed and can compile and run a kernel on the GPU.

Run from the repo root:  python -m scripts.warp_smoke
"""
import numpy as np
import warp as wp

wp.init()
device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
print(f"running on: {device}")


@wp.kernel
def add_one(a: wp.array(dtype=wp.float64)):
    i = wp.tid()
    a[i] = a[i] + wp.float64(1.0)


@wp.kernel
def scatter_count(idx: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.float64)):
    t = wp.tid()
    wp.atomic_add(counts, idx[t], wp.float64(1.0))


n = 1_000_000
a = wp.zeros(n, dtype=wp.float64, device=device)
wp.launch(add_one, dim=n, inputs=[a], device=device)
print("add_one: sum =", a.numpy().sum(), "(expect 1000000.0)")

idx = wp.array(np.array([0, 0, 2], dtype=np.int32), dtype=wp.int32, device=device)
counts = wp.zeros(3, dtype=wp.float64, device=device)
wp.launch(scatter_count, dim=3, inputs=[idx, counts], device=device)
print("atomic_add:", counts.numpy(), "(expect [2. 0. 1.])")
