# diffphys: progress report and handoff

**Author:** Anthony (GitHub `ynohtna9918`) · **As of:** 2026-10-31
**Repo:** https://github.com/ynohtna9918/diffphys-spring-euler (branch `main`, last pushed commit `7883330 update warp`)

I'm handing this project to you (a new Claude assistant) without our earlier
conversation. This report covers what the project is, what I've built and
verified, how my two machines are set up, what's unfinished, and how I'd like
you to work with me.

---

## 1. The project

I'm building a **GPU-accelerated, differentiable 3-D mass–spring simulator
with NVIDIA Warp**, following a PDF spec. Its roadmap runs about 10–14 weeks
over milestones M0–M7: forward simulation, ground contact, gradients, GPU
batching, system identification, locomotion, a gradient-quality study, and a
final report. My README frames it as an infrastructure layer for physical AI
(simulation, Sim2Real). It lists these next items: Warp port, single-scene
differentiability, trajectory optimization, Newton PRs, gradient-quality
benchmark.

As the PDF recommends, I first built a **CPU, float64, single-system NumPy
version**. It is now the **reference answer** that every Warp/GPU result gets
checked against.

## 2. What I've finished

### NumPy reference simulator: complete, tested, pushed

I followed a 3-day plan (`Guides/GUIDE.md`) and finished it in two days
(2026-10-30 → 31).

| File | Contents |
|---|---|
| `src/mass_spring.py` | **`SpringSystem` dataclass**, whose `__post_init__` normalizes dtypes, validates shapes, indices, signs and NaN/inf with `ValueError`, and zeroes pinned particles' velocities.<br>**Force helpers:** `_masses`, `_spring_geometry`.<br>**Forces:** `compute_forces` (vectorized with `np.add.at`) and `compute_forces_loop` (the loop reference).<br>**Core:** `derivatives`, `total_energy`.<br>**Integrators:** `step_explicit_euler`, `step_semi_implicit_euler` (the default), `step_midpoint`.<br>**Driver:** `simulate`.<br>**Factories:** `make_single_oscillator`, `make_chain`.<br>**Analysis:** `measure_period` (zero crossings). |
| `src/__init__.py` | Re-exports the public functions (no Warp import) |
| `tests/test_mass_spring.py` (26 tests) | Forces, one-step values per integrator, free-fall closed forms, explicit-Euler energy growth `(1+(ω·dt)²)ⁿ`, bounded semi-implicit energy, drift ranking, period, convergence order (2× Euler, 4× midpoint), pinned bit-identical, `simulate` shapes, factories |
| `tests/test_physics.py` (7) | Momentum conservation (all 3 integrators), damped envelope `0.1·e^(−0.5t)` and damped period 0.6291 s, static equilibrium for (m, k) = (1, 100), (2, 50), (0.5, 400) |
| `tests/test_validation.py` (16) | 12 bad-input cases (`pytest.raises(ValueError, match=...)`) + type normalization, no springs, pinned-velocity zeroing, input arrays copied |
| `tests/test_vectorized.py` (6) | Vectorized vs loop forces < 1e-12 on 5 random systems (`random_system(seed)`), plus a no-springs case |
| `scripts/check_cases.py` | Acceptance cases A, A2, B, C from `Guides/EXPECTED_OUTPUT.md`, with PASS/FAIL |
| `scripts/plot_energy.py` | Writes `figures/energy_comparison.png` (the three integrators' energy) and prints a cost table |
| `scripts/warp_smoke.py` | Checks that Warp compiles and runs a kernel (`add_one`, `atomic_add`) on the GPU if present |
| `README.md`, `requirements.txt`, `LICENSE` (MIT), `.gitignore` | Docs and setup |
| `Guides/` | `SPEC.md`, `TASKS.md`, `EXPECTED_OUTPUT.md`, `GUIDE.md` (the finished 3-day plan), `GUIDE_WARP.md` (Phase 1 plan, not started), this report |

I wrote most of the code myself, with an assistant explaining each step and
reviewing my work. On my request, the assistant wrote these:
- the validation block
- the two force helpers
- `simulate`
- check cases A2, B and C
- `tests/test_validation.py`
- `scripts/warp_smoke.py`
- the step guides in `Guides/`

### Verified results (Mac)

- `python -m pytest -q` → **55 passed**.
- `python -m scripts.check_cases` → **ALL PASS**:

| Case | Analytic | Measured |
|---|---|---|
| A: oscillator period (k = 100, m = 1) | 2π√(m/k) = 0.6283 s | 0.6283 s |
| A: energy, semi-implicit, 4 s | bounded | max deviation 0.50 % |
| A2: energy change after 2 s (explicit / semi-implicit / midpoint) | grows / bounded / ≈ flat | +22.1 % / −0.4 % / +0.0 % |
| B: hanging mass, settled length | L0 + mg/k = 1.0981 m | 1.0981 m |
| C: chain of 5, spring lengths top → bottom | 0.3 + (4−s)·mg/k | 0.6924 / 0.5943 / 0.4962 / 0.3981 m; bottom y = −2.181 |
| Momentum, free pair | conserved | change < 1e-10 |
| Damped peaks | 0.1·e^(−0.5t) | within 0.1 % |

### Repo housekeeping

- Every commit is attributed to my current GitHub account. I rewrote the
  author email on old commits and force-pushed.
- A stray virtualenv (`bin/`, `lib/`, `include/`, `pyvenv.cfg` at the repo
  root) is untracked and git-ignored. The folders still exist locally on my
  Mac.

### Warp works on both machines (Phase 0 done)

- **Mac:** `python -m scripts.warp_smoke` → `running on: cpu`,
  `add_one: sum = 1000000.0`, `atomic_add: [2. 0. 1.]`.
- **Windows (RTX 5070):** the same correct results, `running on: cuda:0`.
  Warp 1.18.0, CUDA Toolkit 13.4, device line
  `"cuda:0" : "NVIDIA GeForce RTX 5070" (12 GiB, sm_120, mempool enabled)`.

## 3. Conventions (keep these consistent)

- **y is up.** Gravity is `[0, -9.81, 0]`. The PDF uses z-up, but gravity is a
  parameter, so changing it later is one line.
- **Spring `(i, j)`:**
  - `d = x_j − x_i`, `L = |d|`, `n = d/L`
  - `f = [k_s (L − L0) + k_d (v_j − v_i)·n] n`
  - `+f` is added to particle `i` and `−f` to particle `j`
  - springs with `L < 1e-12` are skipped
- **Pinned particle:** `inv_mass == 0`. Gravity force is `m·g` with `m = 0` for
  pinned particles, computed without dividing by zero. Pinned positions stay
  **bit-identical** (tests use `np.array_equal`).
- **Energy:** `½ Σ m|v|² + ½ k_s Σ (L − L0)² − Σ m (g·x)`.
- **Integrators** change `sys` in place and return `None`.
- **`simulate(sys, dt, steps, stepper)`** returns `(xs, vs, energies)` with
  shapes `(steps+1, N, dim)`, `(steps+1, N, dim)`, `(steps+1,)`. Index 0 is the
  initial state.
- **float64** throughout the reference.
- **Run everything from the repo root as modules:** `python -m pytest -q`,
  `python -m scripts.check_cases`. Imports are `from src.mass_spring import ...`,
  so running a file by path fails with `ModuleNotFoundError: src`.

## 4. My two machines

| | Mac (where I write code) | Windows PC (GPU runs) |
|---|---|---|
| Hardware | Apple Silicon. No NVIDIA GPU, so Warp runs on CPU only | NVIDIA GeForce RTX 5070, 12 GB, compute capability sm_120 (Blackwell), WDDM driver mode, driver supports CUDA 13.0. Intel CPU |
| Python | `.venv` is **3.9.6** (macOS system Python). **Needs an upgrade:** Python 3.13 is available at `/opt/anaconda3/bin/python3.13` | 3.13 |
| Warp | **1.12.1** (the newest that supports Python 3.9) | **1.18.0** |
| Repo location | `~/Desktop/diffphys` | Inside a OneDrive-synced Desktop folder. Moving it out (e.g. `C:\dev`) is recommended |
| Tests | 55 passed | smoke test passed; `pytest` not yet confirmed |

Workflow: write and test on the Mac (Warp's CPU mode runs the same kernel code),
push, then `git pull` on Windows for `cuda:0` runs.

## 5. Unfinished items, in order

1. **Make both machines use the same Warp.**
   - Mac: `/opt/anaconda3/bin/python3.13 -m venv --clear .venv`, activate it,
     change `requirements.txt` to `warp-lang==1.18.0` (it says
     `warp-lang>=1.12` now), reinstall, check `pytest` gives 55 passed,
     commit, push.
   - Windows: `git pull`, reinstall.
2. **Windows:** move the repo out of OneDrive and confirm `pytest` → 55 passed.
3. **Commit the uncommitted files:** `Guides/GUIDE_WARP.md`,
   `Guides/PROGRESS_REPORT.md`, and a one-blank-line change in `Guides/GUIDE.md`.
4. **Phase 1, the Warp forward simulator.** Follow `Guides/GUIDE_WARP.md`
   (not started). I'll write `src/warp_sim.py`:
   - **Model:** `WarpModel`, which holds the constant data on the device.
   - **Kernels:** `external_forces` (one thread per particle; sets `F` with
     `=`, which also resets it), `spring_forces` (one thread per spring;
     `wp.atomic_add` / `wp.atomic_sub`), `integrate_semi_implicit` (writes new
     `x_out`, `v_out` arrays).
   - **Python functions:** `compute_forces_warp`, `step_semi_implicit_warp`,
     `simulate_warp`, which has the same return shapes as `simulate` and leaves
     `sys` unchanged.

   Then `tests/test_warp.py`, `check_cases --warp`, and
   `scripts/bench_warp.py`. Done when:
   - Mac shows 64 passed and Windows 73 passed (9 new tests per device)
   - `check_cases --warp` prints ALL PASS with the NumPy numbers
   - the benchmark runs on both machines
5. **Optional cleanup:**
   - `README.md` line 8 starts with a lowercase "a NumPy…".
   - `tests/test_mass_spring.py` has scattered imports, duplicate helpers
     (`run`, `zero_crossing_period`, `osc` duplicate `simulate`,
     `measure_period`, `oscillator`), and a module-level block (around lines
     176–186) that runs `simulate` and prints at import time.
   - `notebooks/` is empty. The energy demo became `scripts/plot_energy.py`.

## 6. Roadmap after Phase 1

| Phase (PDF milestone) | Work |
|---|---|
| 2: contact (M1) | Smooth softplus ground force, one extra term in `external_forces` |
| 3: gradients (M2) | `wp.Tape` through a whole rollout. Per-step `x`, `v` and also `F` arrays. `k_s` and rest lengths become per-spring arrays with `requires_grad=True`. Gradcheck against central finite differences in float64 (target relative error < 1e-5) |
| 4: batching (M3) | `(B, N)` arrays, launches with `dim=(B, M)`, B = 1,024 systems. Switch to float32 for throughput; consider CUDA graph capture |
| 5+: experiments (M4–M7) | System identification, locomotion, gradient-quality study, report |

Use the PDF itself for each milestone's exact acceptance criteria.

## 7. Facts and pitfalls I've already learned

- **`np.add.at` vs `F[i] += f`:** fancy-index `+=` silently drops repeated
  indices. `np.add.at` (NumPy) and `wp.atomic_add` (Warp) don't.
- **Warp kernels must live in a `.py` file.** `python -c "..."` doesn't work,
  because Warp reads the function's source.
- **float64 in Warp:**
  - Use `wp.float64` and `wp.vec3d` (`wp.vec3` is float32).
  - A bare literal like `1.0` is float32, so inside a float64 kernel write
    `wp.float64(1.0)`. Otherwise you get
    `Input types must be the same, got ['float64', 'float32']`.
- **Warp's CPU mode runs threads one at a time, so race conditions are invisible
  on the Mac.** I confirmed it: 100,000 threads doing a non-atomic
  `F[0] = F[0] + 1.0` gave exactly 100000.0 on the CPU. **Warp tests must also
  run on `cuda:0`.**
- **GPU launches are asynchronous.** Call `wp.synchronize_device()` before
  stopping a timer. Do one warm-up launch first, because the first launch
  compiles the kernel.
- **Python 3.9 caps Warp at 1.12.1.** Warp ≥ 1.13 needs Python ≥ 3.10. Warp
  1.18's PyPI wheels need NVIDIA driver R580 or newer and a Turing (sm_75) or
  newer GPU.
- **GeForce FP64 runs at about 1/64 of FP32 speed.** Use float64 for correctness
  tests and gradchecks on small systems, and float32 for big batched runs.
- **12 GB of GPU memory is tight for gradients.** The tape keeps every step.
  Example: 1,024 cubes × 125 particles × 1,000 steps, storing x, v and their
  gradients, takes ≈ 6 GB in float32 but ≈ 12.3 GB in float64, which doesn't
  fit.
- **WDDM (Windows) has slightly higher per-launch overhead than Linux.**
  Windows also kills any single kernel that runs longer than about 2 s (TDR).
  The simulator's kernels take microseconds.
- **Expected Phase 1 numbers.** A previous assistant ran a reference Warp
  implementation in its own scratch space to produce the "Expect" values in
  `GUIDE_WARP.md`. That code is **not** in the repo, because I'm writing my
  own. Its measurements on the Mac (Warp 1.18.0, CPU):
  - Warp vs NumPy forces: differences 1e-13 to 5e-13 on random systems (forces
    800–2200 N)
  - Case A trajectory: identical bit for bit over 4000 steps
  - `check_cases --warp`: ALL PASS with the same numbers
  - Time per semi-implicit step:

    | particles | NumPy | Warp CPU (Mac) |
    |---|---|---|
    | 10 | 17.9 µs | 17.8 µs |
    | 1,000 | 130 µs | 29.5 µs |
    | 100,000 | 12.6 ms | 0.97 ms |
    | 1,000,000 | 135 ms | 9.2 ms |

## 8. How I'd like you to work with me

- **Guide me by default.** For each function: what it should do, how to think
  about it, and the **exact output I should expect**. I write the code. Use
  the Do / Think about / Expect format of `Guides/GUIDE.md` and
  `Guides/GUIDE_WARP.md`.
- **Write code directly only when I explicitly ask** ("write it", "help me
  write", "just modify my test"), and explain each part when you do.
- **Verify every expected number before telling me.** Run reference code
  outside my repo. Don't guess. If something isn't verified (e.g. GPU timings
  you can't run), say so.
- **When reviewing,** point to the problem and explain why. Check whether my
  tests would still pass if the code were wrong.
- **Re-read a file right before editing it.** I sometimes edit at the same time.
- Ask before force-pushing, deleting things, or other hard-to-undo actions.

**Suggested first step:** ask me whether I've done item 1 in section 5 (same
Warp on both machines). Then coach me through `GUIDE_WARP.md` Step 1.1.
