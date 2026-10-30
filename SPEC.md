# diffphys — Developer Specification (W1)

Goal: a NumPy mass–spring simulation core with semi-implicit Euler integration,
structured so that (a) the physics (forces) is decoupled from the integrator,
exactly like Witkin's generic-solver design, and (b) every piece is verifiable
by an analytic test. You write the implementation; this file tells you what
goes where.

## File tree

```
diffphys/
├── README.md
├── SPEC.md                  <- this file
├── .gitignore
├── src/
│   ├── __init__.py          <- re-exports
│   └── mass_spring.py       <- the core: system, forces, integrators, energy
├── tests/
│   ├── __init__.py
│   └── test_mass_spring.py  <- analytic validation tests
└── notebooks/               <- demo notebook comes in W1-D5
```

## `src/mass_spring.py`

### 1. Data: `SpringSystem`

Hold the system as one object (dataclass or simple class). Shapes are contracts.

```python
@dataclass
class SpringSystem:
    x: np.ndarray          # (N, dim) positions, float64
    v: np.ndarray          # (N, dim) velocities
    inv_masses: np.ndarray # (N,) inverse masses; 0.0 => pinned particle
    springs: np.ndarray    # (M, 2) int, particle index pairs
    rest_lengths: np.ndarray  # (M,)
    k_s: float             # spring stiffness
    k_d: float             # damping along spring direction
    gravity: np.ndarray    # (dim,), e.g. [0, -9.81, 0]
    drag: float            # global velocity damping coefficient
```

**Invariants to enforce in the constructor:** `x`/`v` same shape, `inv_masses`
length N, `springs` indices in `[0, N)`, `rest_lengths` length M.

### 2. Forces: `compute_forces(sys) -> (N, dim)`

Signature:

```python
def compute_forces(sys: SpringSystem) -> np.ndarray:
    """Total force on every particle. Must zero the accumulator every call."""
```

Logic, in order:

1. `F = np.zeros_like(sys.x)` — **reset every call** (classic bug otherwise).
2. Gravity: `F += sys.inv_masses[:, None] ... ` — careful: gravity force is
   `m * g`, i.e. `g / inv_mass` for free particles, **0 for pinned**
   (`inv_mass == 0`). Safest: build `masses = np.where(inv>0, 1/inv, 0)`,
   then `F += masses[:, None] * sys.gravity[None, :]`.
3. Global drag: `F += -sys.drag * sys.v`.
4. Per spring `(i, j)` — the bug-prone core:

```python
d = sys.x[j] - sys.x[i]          # single definition of direction; keep it
L = np.linalg.norm(d)
if L < 1e-12:
    continue                     # degenerate spring; skip, don't NaN the field
n = d / L
stretch = L - sys.rest_lengths[s]
rel_v = np.dot(sys.v[j] - sys.v[i], n)   # relative velocity ALONG the spring
f_mag = sys.k_s * stretch + sys.k_d * rel_v
f = f_mag * n
F[i] += f
F[j] -= f                        # Newton's third law, every time
```

5. Return `F`.

**Watch out:** damping uses the *relative* velocity projected on the spring
direction — using absolute velocity here is a physics error (that's what the
separate `drag` term is for). Vectorizing later: scatter-add with
`np.add.at(F, idx, vals)`; plain `F[idx] += vals` silently drops duplicate
indices.

### 3. Derivative: `derivatives(sys) -> (dx, dv)`

```python
def derivatives(sys: SpringSystem):
    F = compute_forces(sys)
    a = F * sys.inv_masses[:, None]   # pinned particles get exactly 0
    return sys.v.copy(), a
```

This is the Witkin "generic solver" seam: the integrator below only calls
this function and never knows what a spring is.

### 4. Integrators

```python
def step_semi_implicit_euler(sys, dt): ...
def step_explicit_euler(sys, dt): ...     # for the energy comparison demo
def step_midpoint(sys, dt): ...           # two force evaluations per step
```

Contracts: take the system, advance it by `dt`, return nothing (mutate
`sys.x`, `sys.v` in place **or** return new arrays — pick one and document it;
in-place is fine for a loop, but then history recording must copy).

Semi-implicit Euler — **the ordering is the whole point**:

```python
_, a = derivatives(sys)
sys.v = sys.v + a * dt     # velocity FIRST
sys.x = sys.x + sys.v * dt # position uses the NEW velocity
```

Explicit Euler (keep for contrast): position uses the *old* velocity. Energy
grows every step — this is the p. 10 spiral from tonight's reading, now as
code you can watch happen.

Midpoint: half-step velocity, evaluate forces at the half step, full step with
those forces (2 force evaluations per step).

### 5. Energy: `total_energy(sys) -> float`

```python
KE  = 0.5 * sum(masses * |v|^2)
PE_spring = 0.5 * k_s * sum((L - L0)^2)          # over all springs
PE_grav   = -sum(masses * (g · x))               # sign per your gravity vector
```

Used only for validation: semi-implicit energy must stay **bounded**
(oscillates, no monotonic drift); explicit Euler energy must visibly grow.

### 6. Factories (for tests and demos)

```python
def make_single_oscillator(k_s=100.0, m=1.0, rest=1.0, dim=3) -> SpringSystem:
    """Particle 0 pinned, particle 1 free, one spring. The analytic testbed."""

def make_chain(n=5, ...) -> SpringSystem: ...
```

## `src/__init__.py`

Re-export the public surface only:

```python
from .mass_spring import (SpringSystem, compute_forces, derivatives,
                          step_semi_implicit_euler, step_explicit_euler,
                          step_midpoint, total_energy,
                          make_single_oscillator, make_chain)
```

## `tests/test_mass_spring.py`

Four tests. Plain `assert`s (pytest or a `__main__` runner — your choice).

1. **Period test** (`make_single_oscillator`, displaced along the spring axis,
   zero damping, no gravity): simulate ~10 periods, measure zero crossings of
   displacement, assert measured period within 3% of `2π√(m/k)`.
   *This is the one that proves the core is right.*
2. **Energy boundedness**: semi-implicit Euler over 1000 steps — final energy
   within ±5% of initial (bounded oscillation, no secular growth).
3. **Explicit Euler diverges** (documents the spiral): same setup, energy
   after 1000 steps is strictly larger than initial by a clear margin.
4. **Pinned particle never moves** + **Newton's third law**: internal spring
   forces sum to zero (isolated system, no gravity/drag).

## Definition of done (W1)

- [ ] All four tests pass with the semi-implicit integrator.
- [ ] `total_energy` flat-ish; explicit-Euler contrast demo runs.
- [ ] dt chosen against `2/√(k/m)` — if a sim explodes, check dt vs
      stiffness before suspecting logic.
- [ ] README checklist updated; commit `W1: mass-spring core + tests`.
