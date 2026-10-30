# diffphys — 3-Day Build Guide

You write every line of code. This file tells you, step by step, **what to
build, how to think about it, and what you should see** when it works.
Work top to bottom; don't start a step until the previous step's "Expect"
check passes.

Sources this guide follows (in priority order):
1. `TASKS.md` — the checklist (stage numbers below match it)
2. `SPEC.md` — signatures and formulas
3. `EXPECTED_OUTPUT.md` — the numbers your finished code must hit
4. The PDF spec (Warp/GPU, M0–M7) — the long-term roadmap. **Not** a 3-day job.

## Scope: what "done in 3 days" means

The PDF describes ~10–14 weeks of work (M0–M7: forward sim → gradients → GPU
batching → system ID → locomotion → gradient study → report). In 3 days you
finish the **NumPy core that everything else is built on**, which equals the
PDF's M0 + most of M1 for a CPU/float64 reference implementation:

- `SpringSystem`, `compute_forces`, `derivatives`, `total_energy`
- three integrators + `simulate` driver
- factories (`make_single_oscillator`, `make_chain`)
- 4 required tests + 3 extra physics tests from the PDF test plan
- `scripts/check_cases.py` that prints Cases A / A2 / B / C like `EXPECTED_OUTPUT.md`
- demo notebook with the integrator energy plot
- a vectorized force function (prep for the Warp port)

The PDF itself says "first do CPU, float64, single-system forward sim; get
forward right before touching gradients". This is that step. The float64 NumPy
version becomes the "reference answer" the GPU version is always compared to.

Two conventions to fix now:
- **Up axis.** Repo files use **y-up**, gravity `[0, -9.81, 0]`; the PDF uses
  z-up. Use y-up (the expected numbers assume it). Gravity is a parameter, so
  switching later is one line.
- **Spring sign.** The PDF writes `d = x_a - x_b`, `f = -[...] d̂` added to `a`.
  `SPEC.md` writes `d = x_j - x_i`, `f = +[...] n` added to `i`. These are the
  same physics. Use the `SPEC.md` form and never mix them.

Hardware note: your Mac has no NVIDIA GPU. Warp installs on macOS but runs
**CPU only**. The GPU milestone (PDF M3) will need a CUDA machine (Colab, a
lab box, or cloud) later. Nothing in these 3 days needs a GPU.

---

## Daily schedule (about 6–7 focused hours per day)

| Day | Build | End-of-day proof |
|---|---|---|
| **Day 1** | Stages 0, 1, 2, 3, 5 (data, forces, derivatives, energy) | Hand checks in the REPL give exactly the numbers below |
| **Day 2** | Stages 4, 6, 7 + `scripts/check_cases.py` | `pytest` shows 4 passed; check script prints PASS for A, A2, B, C |
| **Day 3** | Extra tests, vectorized forces, Stage 8 demo, Stage 9 wrap-up; stretch: Warp kernel | 7+ tests pass; energy plot saved; commit pushed |

If you fall behind: Day 3's stretch and the vectorized forces can slip.
Stages 1–7 and the check script cannot.

---

# Day 1 — The physics core

### Setup (Stage 0, ~20 min)

**Do**
- Create a venv, install `numpy matplotlib pytest`.
- Create `notebooks/` (empty, keep it in git with a `.gitkeep`).

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install numpy matplotlib pytest
```

**Expect:** `python -c "import numpy, pytest"` prints nothing (no error).

---

### Step 1.1 — `SpringSystem` (Stage 1, ~1 h)

**Do**
- A `@dataclass` with the 9 fields in `SPEC.md` §1.
- Add `__post_init__` that converts arrays with `np.asarray(..., dtype=np.float64)`
  (springs: `dtype=int`), then validates:
  - `x.shape == v.shape`, and `x.ndim == 2`
  - `inv_masses.shape == (N,)`
  - `springs.shape == (M, 2)`, all indices in `[0, N)`
  - `rest_lengths.shape == (M,)`
  - `gravity.shape == (dim,)`
- Raise `ValueError` with a message that says *which* check failed.

**Think about**
- Why convert dtype in the constructor? (So integer input like `[[0,0,0],[1,0,0]]`
  doesn't make later `x += v*dt` silently truncate or fail.)
- Pinned convention: `inv_mass == 0`. Should a pinned particle with nonzero
  starting velocity be allowed? It would drift forever (a = 0 but v ≠ 0).
  Simplest: zero `v` for pinned rows in `__post_init__`, and say so in the docstring.

**Expect**
- Building a valid 2-particle system works.
- `springs=[[0, 2]]` with N = 2 raises `ValueError` mentioning the index range.
- `rest_lengths` of the wrong length raises `ValueError`.

---

### Step 1.2 — `compute_forces(sys)` (Stage 2, ~2 h — the most important function)

**Do** — exactly in the order of `TASKS.md` 2.1–2.4:
1. `F = np.zeros_like(sys.x)` (fresh every call)
2. masses from inverse masses without dividing by zero:
   `np.where(inv > 0, 1.0 / np.where(inv > 0, inv, 1.0), 0.0)`
   (the inner `where` stops NumPy warning about `1/0`). Put this in a small
   helper `_masses(sys)` — you'll reuse it in `total_energy`.
3. gravity: `F += m[:, None] * g[None, :]`
4. drag: `F -= drag * v`
5. spring loop with the (a)–(g) recipe. Write a helper
   `_spring_geometry(sys, s) -> (d, L)` (or a vectorized one returning all
   `d`, `L`) — `total_energy` must reuse it (TASKS 5.2).

**Think about**
- Does drag act on pinned particles? It adds a force, but `derivatives`
  multiplies by `inv_mass = 0`, so it can't move them. That's the "trace the
  path to zero acceleration" exercise in TASKS 1.4.
- Damping uses `(v_j - v_i) · n`, **relative** velocity. Absolute velocity
  there is a bug.

**Expect** (Case A start: particle 0 pinned at origin, particle 1 at
`(1.1, 0, 0)`, m = 1, k_s = 100, L0 = 1, everything else off):

```
F[0] = [ 10.  0.  0.]     # spring pulls the pin toward particle 1
F[1] = [-10.  0.  0.]     # and pulls particle 1 back
F.sum(axis=0) = [0. 0. 0.]
```

Turn gravity on (`[0, -9.81, 0]`) and you should get
`F[1] = [-10, -9.81, 0]` while `F[0]` stays `[10, 0, 0]` (pinned: no weight).

Put particle 1 exactly on particle 0 (L = 0): no exception, no NaN, spring skipped.

---

### Step 1.3 — `derivatives(sys)` (Stage 3, ~20 min)

**Do:** `F = compute_forces(sys)`, `a = F * inv_masses[:, None]`,
return `(sys.v.copy(), a)`.

**Think about (say it out loud):** why `.copy()`? If you return `sys.v`
itself and an integrator later does `sys.v += ...` in place, your "dx" array
changes under you. The midpoint method hits exactly this bug.

**Expect** on the Case A start: `a[0]` is **exactly** `[0., 0., 0.]`,
`a[1] = [-10, 0, 0]`. `dx is sys.v` → `False`.

---

### Step 1.4 — `total_energy(sys)` (Stage 5, ~45 min)

**Do:** `KE + PE_spring + PE_grav` per `SPEC.md` §5, reusing `_masses` and
your spring-geometry helper.

**Think about:** `PE_grav = -Σ m (g · x)`. With `g = [0, -9.81, 0]` that is
`+Σ m·9.81·y`: higher = more energy. Check the sign on paper.

**Expect**
- Case A start: `0.5` J exactly (`0.5 · 100 · 0.1²`).
- Particle 1 at `(1.0, 0, 0)`, at rest, gravity on: `0.0`.
- Particle 1 at `(0, -1.0, 0)`, at rest, gravity on: `-9.81`.

### Day 1 checkpoint
In a REPL or scratch file, all of the "Expect" values above match. Commit:
`git commit -m "Day 1: SpringSystem, forces, derivatives, energy"`.
Then ask me to review `src/mass_spring.py` before Day 2.

---

# Day 2 — Integrators, factories, tests, acceptance cases

### Step 2.1 — Pick in-place vs return-new (5 min)

Recommendation: **mutate in place and return `None`**. `simulate` then must
append `sys.x.copy()` / `sys.v.copy()` to history. Write that decision in the
module docstring.

### Step 2.2 — Three integrators (Stage 4.1–4.3, ~1.5 h)

**Do**
- `step_explicit_euler`: compute `a` from the current state; `x += v·dt` with
  the **old** `v`; then `v += a·dt`. (Easy trap: if you update `v` first you
  accidentally wrote semi-implicit.)
- `step_semi_implicit_euler`: `v += a·dt` first, then `x += v·dt` with the new `v`.
- `step_midpoint`:
  1. `dx, a = derivatives(sys)` at the start state
  2. build a **separate** midpoint system with `dataclasses.replace(sys, x=x + dx·dt/2, v=v + a·dt/2)`
  3. `dx_mid, a_mid = derivatives(mid)`
  4. `sys.x = x + dx_mid·dt`, `sys.v = v + a_mid·dt`

  Note `replace` re-runs `__post_init__`, which is fine (and validates again).

**Expect** — one step, `dt = 0.001`, from the Case A start:

| integrator | x of particle 1 | v of particle 1 | energy after |
|---|---|---|---|
| explicit | `1.1` (unchanged) | `-0.01` | `0.50005` |
| semi-implicit | `1.09999` | `-0.01` | `0.4999500…` |
| midpoint | `1.099995` | `-0.01` | `≈ 0.5000000` |

If explicit and semi-implicit give the same x, your update order is wrong.

### Step 2.3 — `simulate(sys, dt, steps, stepper)` (Stage 4.4, ~30 min)

**Do:** loop `steps` times: call `stepper(sys, dt)`, append copies of x and v
and `total_energy(sys)`. Return `(xs, vs, energies)` as NumPy arrays of shape
`(steps, N, dim)`, `(steps, N, dim)`, `(steps,)`. Decide whether index 0 is
the initial state or the state after step 1, and document it (including the
initial state is handy: `energies[0]` is then E0).

**Expect:** Case A, semi-implicit, 1000 steps → `xs.shape == (1001, 2, 3)` if
you include the initial state.

### Step 2.4 — Factories (Stage 6, ~45 min)

**Do**
- `make_single_oscillator(k_s=100, m=1, rest=1, dim=3, k_d=0, drag=0, gravity=None)`:
  pin at origin, particle 1 at `(rest, 0, 0)`, one spring. **At natural length** —
  tests displace it themselves.
- `make_chain(n=5, spacing=0.3, k_s=100, m=1, k_d=0, drag=0, gravity=None, dim=3)`:
  particle `i` at `(0, -i·spacing, 0)`, springs `(i, i+1)`, rest length =
  `spacing`, particle 0 pinned.
- `gravity=None` → zero vector of length `dim` (avoid a mutable default argument).

**Expect:** `make_chain(5).springs` is `[[0,1],[1,2],[2,3],[3,4]]`; the chain
starts with zero force from springs (all at rest length).

### Step 2.5 — The four required tests (Stage 7, ~1.5 h)

In `tests/test_mass_spring.py`, import with `from src.mass_spring import ...`
and run `pytest` from the repo root.

1. **Period** — oscillator, set `x[1, 0] = 1.1`, semi-implicit, `dt = 0.001`,
   ~6.3 s (10 periods). Displacement `u = xs[:, 1, 0] - 1.0`. Find upward
   zero crossings (`u[k] < 0 <= u[k+1]`), linearly interpolate each crossing
   time, `period = mean(diff(times))`. Assert within 3% of `2π√(m/k) = 0.6283`.
2. **Energy bounded** — 1000 steps semi-implicit; `max |E - E0| / E0 < 0.05`.
3. **Explicit grows** — 1000 steps explicit; assert `E_final > 1.05 · E0`.
   (Theory: energy grows by a factor `1 + (ω·dt)²` per step → `e^0.1 ≈ 1.105`.)
4. **Pinned + third law** — after many steps, `np.array_equal(xs[-1, 0], xs[0, 0])`
   (bit-identical, not `allclose`); and on a stretched chain with gravity/drag
   off, `np.allclose(compute_forces(sys).sum(axis=0), 0, atol=1e-12)`.

**Expect:** `pytest -q` → `4 passed`. Measured period ≈ `0.6284 s`.

### Step 2.6 — `scripts/check_cases.py` (~1.5 h)

**Do:** one script that builds each case from `EXPECTED_OUTPUT.md`, runs it,
and prints the same format with PASS/FAIL. Use your factories plus small edits
(e.g. set `x[1,0] = 1.1` for Case A). Run as:

```bash
python -m scripts.check_cases
```

How to measure each number:
- **A, period:** same zero-crossing code as test 1 → put it in a helper you can import.
- **A, energy max deviation:** `max |E - E0| / E0` over 4 s.
- **A2:** run 2 s with each stepper, print `E0 → E_final` and % change.
- **B, settled length:** mean of `‖x1 − x0‖` over the last 2000 steps.
- **C, per-spring lengths:** mean of each spring's length over the last 2 s.

**Expect** (your numbers should land on these within tolerance):
```
Case A: period 0.6284 s (analytic 0.6283), energy max dev ~0.5%, pin unchanged
A2:     explicit +22.1%  |  semi-implicit -0.4%  |  midpoint 0.0%
Case B: settled length 1.0981 m, particle 1 at (0.00, -1.0981, 0.00)
Case C: 0.6924, 0.5943, 0.4962, 0.3981 m; bottom y = -2.181
```

If Case B/C don't settle: check `k_d` and `drag` are actually passed through
the factory. If anything explodes: check `dt` against `2/√(k/m)` before
suspecting your logic (SPEC "Definition of done").

### Day 2 checkpoint
`pytest -q` → 4 passed, check script all PASS. Commit
`"Day 2: integrators, factories, tests, acceptance cases"`. Ask me to review
the tests (I'll look for tests that would still pass if the code were wrong).

---

# Day 3 — Harden, demo, ship

### Step 3.1 — Three extra physics tests from the PDF test plan (~1.5 h)

These are the PDF's M1 tests that apply to a NumPy core:

5. **Momentum conservation** — two *free* particles (no pin), springs only,
   no gravity/drag, give them different initial velocities. `Σ m·v` after
   1000 steps equals the start within `1e-10`. (Passes only if your scatter is
   truly equal and opposite.)
6. **Damping decay** — oscillator with `k_d = 1`, gravity/drag off, start at
   `x = 1.1`. Theory: ζ = k_d / (2√(k m)) = 0.05, amplitude envelope
   `0.1 · e^(−ζ ω t)` with ω = 10, i.e. `0.1 · e^(−0.5 t)`. Find the peaks of
   `u(t)` and assert each is within 2% of the envelope. (At t = 2 s: ≈ 0.0368.)
7. **Static equilibrium** — Case B as a test: settled length within 1% of
   `L0 + m·g/k`.

**Expect:** `pytest -q` → `7 passed`.

### Step 3.2 — Vectorized forces (~1.5 h, prep for Warp)

**Do:** write `compute_forces_vec(sys)` with no Python loop over springs:
- `i, j = springs[:, 0], springs[:, 1]`; `d = x[j] - x[i]` → shape `(M, dim)`
- `L = norm(d, axis=1)`; mask out `L < eps` (set their force to 0)
- `f = (k_s·stretch + k_d·rel_v)[:, None] · n`
- scatter with `np.add.at(F, i, f)` and `np.add.at(F, j, -f)` — **not**
  `F[i] += f`, which silently drops repeated indices (SPEC warns about this).

Add test 8: on a random 20-particle chain with random velocities,
`max |compute_forces − compute_forces_vec| < 1e-12`.

**Why it matters:** this is the exact shape of the Warp kernel later (one
thread per spring, `wp.atomic_add` instead of `np.add.at`). Then make
`compute_forces` call the vectorized version (keep the loop version for the test).

**Expect:** test passes; the period test runs noticeably faster on a long chain.

### Step 3.3 — Demo notebook (Stage 8, ~1 h)

**Do:** `notebooks/integrators.ipynb`: chain of 5, no damping, gravity on,
same `dt` for all three integrators; plot energy vs time on one figure; save
it as `notebooks/energy_comparison.png`. Add the cost table:

| integrator | force evals / step | energy behavior |
|---|---|---|
| explicit Euler | 1 | grows (spiral) |
| semi-implicit | 1 | bounded wobble |
| midpoint | 2 | nearly flat |

**Expect:** the explicit curve climbs steadily, semi-implicit wiggles in a
narrow band around E0, midpoint looks flat at this scale.

### Step 3.4 — Wrap-up (Stage 9, ~30 min)

- `src/__init__.py` re-exports the public surface (SPEC §`src/__init__.py`).
- README: tick the W1 boxes, add "how to run tests / check script".
- Optionally add `if __name__ == "__main__":` to `mass_spring.py` that runs
  Case A and prints the period (README says `python src/mass_spring.py`).
- `git pull` first (README says the web side has a commit), then commit
  `"W1: mass-spring core + tests"` and push.

### Step 3.5 — Stretch, only if everything above is green

Pick one:
- **(a) First Warp kernel.** `pip install warp-lang`; write a `@wp.kernel`
  `spring_force` for a **single** system (`s = wp.tid()`, one thread per spring, use
  `wp.atomic_add` / `wp.atomic_sub`), run it on CPU, and assert it matches
  `compute_forces_vec` within `1e-10` (float64). This is the first brick of PDF M0.
- **(b) Cube builder.** `make_cube_grid(nx, ny, nz)` with edge, face-diagonal and
  body-diagonal springs (the PDF warns that without diagonals the cube shears
  and collapses). Drop it under gravity with a pinned top corner and check
  energy stays bounded.

---

## How to work with me during these 3 days

- When a step is done, tell me "Step X.Y done" and paste the output of its
  "Expect" check, or ask me to review the file. I'll check it against the spec
  and point at problems **without rewriting your code**.
- When something's wrong, paste the error or the wrong numbers. I'll tell you
  where to look and why, not hand you the fix (unless you ask).
- After Day 3, the next phase is the PDF's M1 (contact) → M2 (gradients). I'll
  plan that when you get there.
