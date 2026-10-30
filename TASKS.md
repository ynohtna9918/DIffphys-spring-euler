# diffphys — Implementation Task List (W1)

Work top to bottom. Each task lists what to build and how you know it's right.
No solutions here — the *how* contracts live in SPEC.md.

## Stage 0 — Repo & environment
- [ ] 0.1 Create the GitHub repo (public, MIT license, README on).
- [ ] 0.2 Clone locally; place README (local version wins), SPEC.md, src/, tests/, notebooks/.
- [ ] 0.3 Python env with numpy as the only dependency; note it in README.
- [ ] 0.4 First commit: skeleton + spec.

## Stage 1 — Data structure `SpringSystem` (src/mass_spring.py)
- [ ] 1.1 Define the class with fields: x, v, inv_masses, springs, rest_lengths, k_s, k_d, gravity, drag.
- [ ] 1.2 Enforce shapes/dtypes: x/v (N, dim) float64; inv_masses (N,); springs (M, 2) int; rest_lengths (M,).
- [ ] 1.3 Constructor validation: x/v shapes match; spring indices in [0, N); rest_lengths length M; raise on violation.
- [ ] 1.4 Pinned convention: inv_masses == 0 means pinned. Trace, for every formula below, the path that makes its acceleration exactly 0.

## Stage 2 — `compute_forces(sys) -> F` of shape (N, dim)
- [ ] 2.1 First line, no exceptions: F starts as a fresh zero array on every call.
- [ ] 2.2 Gravity term: add m*g per free particle, 0 for pinned. You must derive a masses array from inv_masses without dividing by zero.
- [ ] 2.3 Global drag term: proportional to each particle's own velocity, opposite sign.
- [ ] 2.4 Spring loop — for each spring (i, j), in this order:
  - (a) displacement d = x_j - x_i (the only direction definition in the file)
  - (b) current length L
  - (c) if L < epsilon, skip the spring (no NaN may escape this function)
  - (d) unit direction n = d / L
  - (e) stretch = L - rest_length
  - (f) relative velocity along the spring: rel_v = (v_j - v_i) . n
  - (g) scalar force = k_s * stretch + k_d * rel_v; vector along n; add +f to particle i, -f to particle j
- [ ] 2.5 Self-check: with gravity and drag off, the row-sum of F is the zero vector (third law).

## Stage 3 — `derivatives(sys) -> (dx, dv)`
- [ ] 3.1 Call compute_forces to get F.
- [ ] 3.2 a = F scaled row-wise by inv_masses; pinned rows must come out exactly 0.
- [ ] 3.3 Return (a copy of v, a). Decide why dx must be a copy, not a reference, and be able to say it out loud.

## Stage 4 — Integrators (pick in-place or return-new once; be consistent)
- [ ] 4.1 Explicit Euler: x updated with the OLD v; v updated with current a. Implement its defining flaw faithfully — it is the contrast case.
- [ ] 4.2 Semi-implicit Euler: update v with a first, then update x with the NEW v.
- [ ] 4.3 Midpoint: (a) half-step to a midpoint state, (b) recompute forces/acceleration at the midpoint state, (c) full step using the midpoint acceleration. Two force evaluations per step; the midpoint state must not mutate the original (copy).
- [ ] 4.4 `simulate(sys, dt, steps, stepper)`: driver loop — per step, call the chosen stepper, append copies of x/v to a state history and total_energy to an energy history; return both histories.

## Stage 5 — `total_energy(sys) -> float`
- [ ] 5.1 Kinetic: sum of 0.5 * m * |v|^2 (masses derived as in 2.2; pinned particles contribute 0).
- [ ] 5.2 Spring potential: sum of 0.5 * k_s * (L - L0)^2 over springs. Reuse the d/L computation from 2.4 — do not write a second copy.
- [ ] 5.3 Gravitational potential: -sum of m * (g . x), sign consistent with your gravity vector definition.
- [ ] 5.4 Return KE + spring PE + gravitational PE.

## Stage 6 — Factories
- [ ] 6.1 `make_single_oscillator(k_s, m, rest, dim)`: particle 0 pinned, particle 1 free, one spring, spring at natural length. Parameterized.
- [ ] 6.2 `make_chain(n, ...)`: n particles in a line, springs between neighbors, first particle pinned. For the demo.

## Stage 7 — Tests (tests/test_mass_spring.py)
- [ ] 7.1 Period: oscillator from 6.1, displaced along the spring axis, no damping, no gravity, semi-implicit. Measure the period from zero crossings over ~10 periods; assert within 3% of 2*pi*sqrt(m/k).
- [ ] 7.2 Energy bounded: same system, 1000 steps; assert energy stays inside a +/-5% band around its initial value (no secular drift).
- [ ] 7.3 Explicit Euler diverges: same setup with explicit Euler, 1000 steps; assert final energy is clearly larger than initial (tonight's p. 10 spiral, as an assertion).
- [ ] 7.4 Pinned + third law: isolated system (gravity/drag off); after any steps, assert the pinned particle's position is bit-identical; assert compute_forces row-sum is zero within tolerance.

## Stage 8 — Demo (notebooks/, finish on D5)
- [ ] 8.1 Run the chain with all three integrators at identical parameters; plot the three energy curves on one figure.
- [ ] 8.2 Tabulate force evaluations per step per integrator (1, 1, 2) — the accuracy-vs-cost table.

## Stage 9 — Wrap-up
- [ ] 9.1 Tick the matching boxes in the README W1 checklist.
- [ ] 9.2 Commit "W1: mass-spring core + tests". Pull the web-side README commit before pushing.
