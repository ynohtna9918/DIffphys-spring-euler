# diffphys — Warp Guide, Phase 1: the forward simulator on the GPU

Same rules as `GUIDE.md`: you write the code; each step says **what to build,
how to think about it, and what you should see**. Don't start a step until the
previous step's "Expect" check passes.

**Goal of Phase 1:** port forces and semi-implicit Euler to Warp kernels, in
float64, for a single 3-D system, and prove they give the same answers as your
NumPy core. Nothing new physically happens here. The NumPy code is the
reference answer, and every Warp number is checked against it.

**Done when:**
- `tests/test_warp.py` passes on the Mac (CPU) **and** on Windows (CPU + `cuda:0`)
- `python -m scripts.check_cases --warp` prints ALL PASS with the same numbers as NumPy
- `scripts/bench_warp.py` shows where the GPU starts to win

Every "Expect" number below was produced by a reference version of this code
run on your Mac with Warp 1.18.0 (CPU). Numbers marked *(Windows, not
measured)* are estimates. You'll measure them yourself on the RTX 5070.

---

## The plan in one table

| NumPy (`src/mass_spring.py`) | Warp (`src/warp_sim.py`, new) |
|---|---|
| `SpringSystem` holds everything | `WarpModel` holds what never changes (masses, springs, constants); positions and velocities are separate arrays |
| `compute_forces` | two kernels, `external_forces` + `spring_forces`, called by `compute_forces_warp` |
| `np.add.at(F, i, f)` | `wp.atomic_add(F, i, f)` |
| `step_semi_implicit_euler` (changes `sys` in place) | kernel `integrate_semi_implicit` that writes **new** arrays `x_out`, `v_out` |
| `simulate(sys, dt, steps, stepper)` | `simulate_warp(sys, dt, steps, device)`, same return shapes |

**Do not** import `warp` in `src/__init__.py` or `src/mass_spring.py`. The
NumPy core must keep working on a machine without Warp. Import the Warp code
explicitly: `from src.warp_sim import ...`.

---

### Step 1.0 — Same Warp on both machines (~20 min)

**Do (Mac)**
- Your Mac `.venv` is still Python 3.9.6 with Warp 1.12.1, while Windows has
  Warp 1.18.0. Rebuild the Mac venv with Python 3.13 (from the repo root):
  ```bash
  /opt/anaconda3/bin/python3.13 -m venv --clear .venv
  source .venv/bin/activate
  ```
- Change the last line of `requirements.txt` to `warp-lang==1.18.0`, then
  `python -m pip install -r requirements.txt`.
- In VS Code: "Python: Select Interpreter" → `.venv/bin/python`.

**Do (Windows):** after you push, `git pull`, then
`python -m pip install -r requirements.txt` (it's already 1.18.0, so nothing
should change).

**Expect** on both: `python -m pytest -q` → `55 passed`;
`python -c "import warp; print(warp.__version__)"` → `1.18.0`.

(On Warp 1.12.1 you'd see `Warp DeprecationWarning: Support for Python 3.9 is
deprecated and will be removed in Warp 1.13`, which is one more reason to switch.)

---

### Step 1.1 — Warp basics you need (REPL, ~30 min, nothing to keep)

Run these in `python` on the Mac and check each result.

**Types.** float64 is spelled `wp.float64`; a 3-vector of float64 is
`wp.vec3d` (the `d` means double; `wp.vec3` is float32, so don't mix them).

```python
import numpy as np, warp as wp
a = wp.array(np.array([[1., 2, 3], [4, 5, 6]]), dtype=wp.vec3d)
a.shape            # (2,)      <- 2 vectors, not a 2x3 table
a.numpy().shape    # (2, 3)    <- back to NumPy you get the table again
a.device           # cpu  (on Windows: cuda:0, the default when a GPU exists)
```

**History arrays and views.** A 2-D array of vectors, one row per time step:

```python
xs = wp.zeros((4, 2), dtype=wp.vec3d)   # 4 time steps, 2 particles
xs[1].shape                             # (2,)  a VIEW of row 1, no copy
xs.numpy().shape                        # (4, 2, 3)
```

Writing into `xs[1]` from a kernel writes into `xs` itself. `simulate_warp`
depends on this.

**Kernels are strictly typed.**
- Every argument needs a type in the signature.
- A bare float literal like `1.0` is float32. Inside a float64 kernel, write
  `wp.float64(1.0)`. If you forget, the error is:
  `Input types must be the same, got ['float64', 'float32']`.
- Useful built-ins: `wp.tid()`, `wp.length(d)`, `wp.dot(a, b)`,
  `wp.atomic_add(arr, index, value)`, `wp.atomic_sub(...)`. `if` and `return`
  work like in Python.

**Launches.**
`wp.launch(kernel, dim=n, inputs=[...], outputs=[...], device=...)`. The
arguments in `inputs` + `outputs`, in that order, must match the kernel
signature **in order**. Python floats are fine for `wp.float64` arguments.

**On a GPU, launches are asynchronous.** `wp.launch` returns before the kernel
finishes. `.numpy()` waits for you automatically. A timer doesn't (see Step 1.7).

**The first launch compiles.** You'll see
`Module src.warp_sim ... (compiled)` once, then `(cached)` after that. That's
normal.

---

### Step 1.2 — `WarpModel` (~30 min)

**Do:** a small class in `src/warp_sim.py`, built from a `SpringSystem` and a
device:

```python
class WarpModel:
    def __init__(self, sys: SpringSystem, device=None): ...
```

Fields to set:
- `device = wp.get_device(device)`. `None` means Warp's default, so the same
  code runs on `cuda:0` on Windows and on the CPU on the Mac.
- `N`, `M`
- `inv_m`: `wp.array(..., dtype=wp.float64)`
- `springs`: `wp.array(sys.springs.astype(np.int32), dtype=wp.int32)`, shape `(M, 2)`
- `rest`: float64 array
- `k_s`, `k_d`, `drag`: plain floats
- `gravity`: `wp.vec3d(*sys.gravity)`

Raise `ValueError` if `sys.x.shape[1] != 3`, because Warp's vectors are fixed
3-D.

**Think about:** why are `x` and `v` *not* in the model? They change every
step. In Phase 3 (gradients), every step needs its **own** `x` and `v` arrays
so the tape can replay the history. The model is the part that stays fixed.

**Expect** for `WarpModel(make_chain(5))` on the Mac:
```
N 5  M 4  springs.shape (4, 2)  inv_m [0. 1. 1. 1. 1.]  device cpu  gravity [0.0, 0.0, 0.0]
```
`WarpModel(make_chain(5, dim=2))` → `ValueError: the Warp backend is 3-D only, got dim = 2`.

---

### Step 1.3 — The two force kernels (~1.5 h, the most important step)

**Do:** the two kernels, plus the function that launches them.

```python
@wp.kernel
def external_forces(v: wp.array(dtype=wp.vec3d), inv_m: wp.array(dtype=wp.float64),
                    gravity: wp.vec3d, drag: wp.float64,
                    F: wp.array(dtype=wp.vec3d)):
    i = wp.tid()          # one thread per PARTICLE
    ...

@wp.kernel
def spring_forces(x: wp.array(dtype=wp.vec3d), v: wp.array(dtype=wp.vec3d),
                  springs: wp.array2d(dtype=wp.int32), rest: wp.array(dtype=wp.float64),
                  k_s: wp.float64, k_d: wp.float64,
                  F: wp.array(dtype=wp.vec3d)):
    s = wp.tid()          # one thread per SPRING
    ...

def compute_forces_warp(model, x, v, F):
    # launch external_forces with dim=model.N, THEN spring_forces with dim=model.M
```

`external_forces` recipe:
1. Mass from inverse mass, with no division by zero: start with
   `m = wp.float64(0.0)`; `if inv_m[i] > wp.float64(0.0):` set
   `m = wp.float64(1.0) / inv_m[i]`.
2. `F[i] = m * gravity - drag * v[i]`. Use **`=`, not `+=`**.

`spring_forces` recipe: exactly TASKS 2.4 (a)–(g), one spring per thread.
1. `i = springs[s, 0]`, `j = springs[s, 1]`
2. `d = x[j] - x[i]`, `L = wp.length(d)`
3. `if L < wp.float64(1.0e-12): return`
4. `n = d / L`, `rel_v = wp.dot(v[j] - v[i], n)`
5. `f = (k_s * (L - rest[s]) + k_d * rel_v) * n`
6. `wp.atomic_add(F, i, f)` and `wp.atomic_sub(F, j, f)`

**Think about**
- **Why `=` in `external_forces`?** It writes every `F[i]`, so it also *resets*
  the accumulator. That's TASKS 2.1 for free, with no separate zeroing launch.
  It only works because `external_forces` is launched **first**: kernels on the
  same device run in launch order.
- **Why atomics?** Many springs touch the same particle at the same moment.
  `F[i] = F[i] + f` is "read, add, write", and two threads doing that together
  lose one of the updates. This is the GPU version of your `np.add.at` lesson.
- **Trap:** Warp on the **CPU runs the threads one at a time**, so a race bug
  is *invisible* on your Mac. I checked: 100,000 threads doing a non-atomic
  `F[0] = F[0] + 1.0` on the CPU gave exactly `100000.0`. On the GPU it won't.
  That's why the Warp tests must also run on Windows on `cuda:0`. Optional
  experiment: try it there and see how far below 100000 it lands.
- `M = 0` (no springs) means `launch(dim=0)`, which does nothing. No special case
  is needed.

**Expect** (`warp_forces(sys, device)` = build a model, upload `x` and `v` with
`wp.array(..., dtype=wp.vec3d)`, `F = wp.zeros(N, dtype=wp.vec3d)`, call
`compute_forces_warp`, return `F.numpy()`):

```
Case A start (particle 1 at x = 1.1):     [[10.000000000000009, 0, 0], [-10.000000000000009, 0, 0]]
  same, gravity on:                        [[10.000000000000009, 0, 0], [-10.000000000000009, -9.81, 0]]
  particle 1 on top of particle 0:         all zeros, no NaN
no springs, gravity + drag 0.5, v1 = (1,0,0): [[0, -9.81, 0], [-0.5, -9.81, 0]]
random_system(seed), seeds 0..4:          max |F_warp - F_numpy| between 1.1e-13 and 4.6e-13
```

The `...009` is not a bug. `1.1 - 1.0` is `0.10000000000000009` in float64,
and NumPy prints the same thing.

Why isn't the random-system difference exactly 0? Atomics add a particle's
forces in a different order than `np.add.at`, and float addition depends on
order. The forces there are 800–2200 N, so 4.6e-13 is a relative error of
about 2e-16, which is float64's machine precision. That's why the test in
Step 1.6 compares **relative** to `max |F|`.

---

### Step 1.4 — The integrate kernel and one step (~30 min)

**Do**

```python
@wp.kernel
def integrate_semi_implicit(x: wp.array(dtype=wp.vec3d), v: wp.array(dtype=wp.vec3d),
                            F: wp.array(dtype=wp.vec3d), inv_m: wp.array(dtype=wp.float64),
                            dt: wp.float64,
                            x_out: wp.array(dtype=wp.vec3d), v_out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    # v_new = v + F * inv_m * dt   (velocity FIRST)
    # x_out = x + v_new * dt       (position uses the NEW velocity)

def step_semi_implicit_warp(model, x, v, F, dt, x_out, v_out):
    # compute_forces_warp, then launch integrate_semi_implicit with dim=model.N
```

Write `F[i] * inv_m[i] * dt` in that order. It's the same order NumPy uses
(`F * inv_masses`, then `* dt`), so the rounding is identical.

**Think about**
- **Why `x_out` / `v_out` instead of overwriting `x`, `v`?** For gradients, the
  tape needs every step's input to still exist afterwards. If step 5
  overwrites step 4's positions, the backward pass has nothing to differentiate
  through. Writing new arrays now means Phase 3 needs no rewrite.
- **The pinned particle, traced exactly:** `inv_m = 0` gives
  `v_new = 0 + F·0·dt = 0`, then `x_out = x + 0·dt = x`. Not "close to" x but
  equal to x, bit for bit.

**Expect:** one step from the Case A start (`dt = 0.001`) gives values
**identical** to your NumPy `test_one_step` semi-implicit row:
```
x1 = 1.09999    v1 = -0.010000000000000009    E = 0.49995000500000025
```

---

### Step 1.5 — `simulate_warp(sys, dt, steps, device=None)` (~45 min)

**Contract:** same as `simulate(sys, dt, steps, step_semi_implicit_euler)`.
It returns NumPy arrays `(xs, vs, energies)` of shapes `(steps+1, N, 3)`,
`(steps+1, N, 3)` and `(steps+1,)`, with index 0 = the initial state. One
difference: it does **not** modify `sys`. Say so in the docstring.

**Do**
1. `model = WarpModel(sys, device)`
2. Build the initial history in NumPy:
   `xs0 = np.zeros((steps + 1, N, 3))`, set `xs0[0] = sys.x`; same for `vs0`.
   Upload both with `wp.array(..., dtype=wp.vec3d, device=model.device)`. The
   shape on the device is `(steps + 1, N)`.
3. `F = wp.zeros(N, dtype=wp.vec3d, device=...)`: one force buffer, reused
   every step.
4. Loop `t` over `range(steps)`: call `step_semi_implicit_warp` with
   `xs[t], vs[t]` as input and `xs[t + 1], vs[t + 1]` as output (views, from
   Step 1.1).
5. **One** copy back at the end: `xs.numpy()`, `vs.numpy()`.
6. Energies with your existing NumPy function, so there's one source of truth:
   `total_energy(dataclasses.replace(sys, x=xs_np[t], v=vs_np[t]))` for each `t`.

**Think about:** why one big history array instead of copying back each step?
40,000 small GPU→CPU copies would cost more than the physics. One copy at the
end is cheap. Memory is `(steps+1) · N · 48` bytes per array, which is about
9.6 MB for Case C. That's fine here. For a million particles, don't keep a
history; use the ping-pong loop from Step 1.7.

**Expect** (Mac CPU, Warp vs NumPy `simulate`, same system, same steps):

| system | steps | max abs Δx | max abs Δv | max abs ΔE |
|---|---|---|---|---|
| Case A (oscillator at 1.1) | 4000 | 0.0 | 0.0 | 0.0 |
| Case C chain (k_d 5, drag 0.5, gravity) | 2000 | 0.0 | 4.4e-16 | 7.1e-15 |
| `random_system(3)` | 1000 | 2.3e-15 | 1.2e-14 | 9.1e-13 |

Shapes: Case A → `(4001, 2, 3) (4001, 2, 3) (4001,)`. The pinned particle is
bit-identical at the end.

*(Windows `cuda:0`, not measured.)* Expect small nonzero differences, still far
below 1e-10. GPUs use fused multiply-add (`a*b + c` rounded once instead of
twice) and a different atomic order, so the last digits can change.

---

### Step 1.6 — `tests/test_warp.py` (~1 h)

**Do:** top of the file:

```python
import numpy as np
import pytest

wp = pytest.importorskip("warp")      # skip this whole file if Warp isn't installed

from src.warp_sim import WarpModel, compute_forces_warp, simulate_warp
from tests.test_vectorized import random_system   # reuse your random test systems

DEVICES = ["cpu"] + (["cuda:0"] if wp.get_cuda_device_count() > 0 else [])
```

Put `@pytest.mark.parametrize("device", DEVICES)` on every test, and write a
`warp_forces(sys, device)` helper (described in Step 1.3). Five tests:

| test | what it checks | tolerance | bug it catches (I mutation-tested each) |
|---|---|---|---|
| `test_forces_match_numpy` (seeds 0–4) | `random_system(seed)` forces vs `compute_forces` | `max abs diff < 1e-13 · max abs F_np` | damping with absolute velocity; third-law sign (`atomic_add` on j); missing zero-length guard (NaN) |
| `test_zero_length_spring_and_no_springs` | the two edge cases from Step 1.3 | exact zeros / `allclose` | missing guard; `dim=0` launch problems |
| `test_one_step_case_a` | Step 1.4 numbers, and that `sys.x[1, 0]` is still `1.1` afterwards | `abs=1e-12` | explicit-Euler order (x with the old v) |
| `test_trajectory_matches_numpy` | Case C chain, 2000 steps, Warp vs NumPy | `max abs Δx, max abs Δv < 1e-10` | `external_forces` using `+=` (forces pile up across steps); explicit-Euler order |
| `test_pinned_particle_bit_identical` | `make_chain(5, gravity=G)`, 1000 steps | `np.array_equal` | anything that moves a pin |

One bug none of these catch **on the Mac**: a non-atomic scatter. Only the
`cuda:0` runs can catch it (Step 1.3, "Trap").

**Expect**
- Mac: `python -m pytest -q` → `64 passed` (55 old + 9 new).
- Windows: `73 passed` (55 + 9 on `cpu` + 9 on `cuda:0`).

---

### Step 1.7 — `check_cases --warp` (~30 min)

**Do:** in `scripts/check_cases.py`:
- Add a helper `run(sys, steps)` that returns `simulate(sys, DT, steps,
  step_semi_implicit_euler)`, or `simulate_warp(sys, DT, steps, device=DEVICE)`
  when a module-level `BACKEND == "warp"`. Import `simulate_warp` *inside* the
  `if`, so the NumPy run never needs Warp.
- Use `run(...)` in cases A, B, C. **A2 stays NumPy-only**: it compares three
  integrators, and Warp has only semi-implicit.
- In `main()`, use `argparse` with `--warp` (flag) and `--device` (default `None`).
  Print which backend and device ran, and the total time
  (`time.perf_counter()`).

**Expect**
- `python -m scripts.check_cases --warp` prints `backend: warp on cpu` (Windows:
  `on cuda:0`) and then exactly the NumPy numbers: period 0.6283 s, energy
  deviation 0.50%, Case B 1.0981 m, Case C 0.6924 / 0.5943 / 0.4962 / 0.3981,
  bottom y = -2.181, ALL PASS.
- Time on the Mac: NumPy ≈ 2.0 s (including A2), Warp CPU ≈ 3.1 s.

Warp is *slower* here, and that's expected. Five particles means every step is
three tiny launches, and the fixed cost of each launch is larger than the
work. The NumPy energy loop also takes about 0.8 s of Case C's 2 s. A
5-particle system can't show what a GPU is for, so Step 1.8 uses big ones.

---

### Step 1.8 — `scripts/bench_warp.py`: when does the GPU win? (~45 min)

**Do:** for `make_chain(n, gravity=G)` with n = 10, 1,000, 100,000 and
1,000,000, time **one step**, in µs:
- NumPy: loop `step_semi_implicit_euler(sys, DT)`.
- Warp, on `"cpu"` and on `"cuda:0"` if present. Use two buffer pairs and
  ping-pong them, with no history:
  ```python
  step_semi_implicit_warp(model, x, v, F, DT, x_out, v_out)   # warm-up: compiles
  wp.synchronize_device(device)
  t0 = time.perf_counter()
  for _ in range(steps):
      step_semi_implicit_warp(model, x, v, F, DT, x_out, v_out)
      x, x_out = x_out, x
      v, v_out = v_out, v
  wp.synchronize_device(device)        # without this you time the LAUNCHES, not the work
  per_step = (time.perf_counter() - t0) / steps
  ```
Use fewer steps for the big sizes (e.g. 2000, 2000, 200, 20).

**Think about** the two timing traps:
1. Without the warm-up step, you'd be timing the compiler.
2. Without `synchronize_device`, the GPU timer stops while kernels are still
   queued, and you'd get impossibly fast numbers.

**Expect**

| particles | NumPy | Warp CPU (Mac) | Warp `cuda:0` *(Windows, not measured)* |
|---|---|---|---|
| 10 | 17.9 µs | 17.8 µs | roughly the same or slower: launch cost dominates |
| 1,000 | 130 µs | 29.5 µs | still mostly launch cost |
| 100,000 | 12.6 ms | 0.97 ms | well under the CPU |
| 1,000,000 | 135 ms | 9.2 ms | my estimate: around 1 ms (memory-bandwidth bound) |

Your Windows CPU will give its own NumPy and Warp-CPU numbers, which is fine.
The pattern is what matters:
- At 10 particles, nothing beats NumPy by much.
- At a million particles, Warp CPU is about 15× faster than NumPy (Mac), and the
  GPU should be about 10× faster again.
- This is also the argument for Phase 4: instead of one huge system, the GPU
  fills up by running **1,024 systems at once**.

---

### Phase 1 checkpoint

On the Mac:
- `pytest -q` → 64 passed
- `check_cases --warp` → ALL PASS

Commit `"Warp forward simulator (Phase 1)"` and push. On Windows, `git pull`,
then run `pytest -q` (73 passed), `check_cases --warp` and `bench_warp`. Paste
me the Windows output, and ask me to review `src/warp_sim.py`.

## What Phase 1 sets up for later

**v1 scope (decided Nov 1, 2026):** after Phase 1, go straight to Phase 3
(gradients), then a small system-identification demo. Phase 2 (contact) and
Phase 4 (batching) are future work, not part of v1 (see README).

- **Phase 2 (contact):** one more term in `external_forces` (a smooth softplus
  floor). Nothing else changes.
- **Phase 3 (gradients):** the per-step `xs`, `vs` history and the
  write-new-arrays integrator are already what `wp.Tape` needs. Two things will
  change:
  - `F` must also become one array per step (gradients flow back through
    every step's force).
  - `k_s` and `rest` become per-spring arrays with `requires_grad=True`, so you
    can ask for d(loss)/d(stiffness of spring 7).
- **Phase 4 (batching):** arrays gain a batch dimension, `(B, N)`, and launches
  become `dim=(B, M)`.
