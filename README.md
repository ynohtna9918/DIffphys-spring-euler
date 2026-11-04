# diffphys — Differentiable Mass–Spring Simulation

[![tests](https://github.com/ynohtna9918/diffphys-spring-euler/actions/workflows/test.yml/badge.svg)](https://github.com/ynohtna9918/diffphys-spring-euler/actions/workflows/test.yml)

A 3D mass–spring simulator in two layers: a float64 NumPy reference
implementation, and an NVIDIA Warp (GPU) version that is checked against it
and made differentiable with `wp.Tape`.

## Status

- **Done:** NumPy (float64, CPU) reference simulator with three integrators,
  validated against analytic physics (55 tests; see Validation). It is the
  reference the Warp version is checked against.
- **In progress:** Warp port of the forward simulator.
- **Not yet implemented:** gradients and system identification.

## v1 scope (target: early December 2026)

v1 is deliberately small: a differentiable simulator plus one result showing
the gradients are useful. It is done when every box below is ticked.

- [x] NumPy reference: forces, three integrators, factories, 55 tests
- [x] Acceptance cases A, A2, B, C pass; integrator energy figure
- [ ] Warp forward simulator matches NumPy (Case C, 2000 steps:
      max |Δx|, |Δv| < 1e-10) on CPU and `cuda:0`; `check_cases --warp` ALL PASS
- [ ] GPU benchmark from 10 to 1,000,000 particles
- [ ] Gradients through `wp.Tape`: autodiff vs central finite differences
      (float64) for k, c, L0, x0, v0, relative error < 1e-5
- [ ] System identification: recover spring stiffness from a noisy
      trajectory, error < 5%
- [ ] CI runs the test suite on every push
- [ ] One-page technical note (`docs/v1_note.md`); tag `v1.0`

## Future work (out of scope for v1)

Batching many systems per launch (B ≥ 1,024), smooth ground contact,
cube/voxel builders and actuation, soft-robot locomotion, a first- vs
zeroth-order gradient-quality study, a cross-check against DiffTaichi, and
contributions to Newton.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                 # 55 tests
python -m scripts.check_cases       # acceptance cases A, A2, B, C -> ALL PASS
python -m scripts.plot_energy       # writes figures/energy_comparison.png
```

## Physics model

- State: positions `x` (N, 3), velocities `v` (N, 3), inverse masses (`0` = pinned). y is up.
- Spring `(i, j)`: `d = x_j - x_i`, `L = |d|`, `n = d / L`,
  `f = [k_s (L - L0) + k_d (v_j - v_i)·n] n`, added to `i`, subtracted from `j`.
- Gravity `m g` on free particles; global drag `-c v`.
- Integrators: explicit Euler, semi-implicit (symplectic) Euler (default), midpoint.

## Validation

| Check | Analytic | Measured |
|---|---|---|
| Oscillator period | 2π√(m/k) = 0.6283 s | 0.6283 s |
| Energy, semi-implicit, 4 s | bounded | max deviation 0.50 % |
| Energy after 2 s: explicit / semi-implicit / midpoint | grows / bounded / ≈ flat | +22.1 % / −0.4 % / +0.0 % |
| Hanging mass length | L0 + mg/k = 1.0981 m | 1.0981 m |
| Chain of 5, spring lengths | 0.6924 / 0.5943 / 0.4962 / 0.3981 m | same to 4 decimals |
| Momentum, free pair | conserved | change < 1e-10 |
| Damped peaks | 0.1·e^(−0.5 t) | within 0.1 % |

![Energy of the three integrators](figures/energy_comparison.png)

## Layout

- `src/mass_spring.py` — `SpringSystem`, forces (vectorized + loop reference),
  integrators, `simulate`, factories, `measure_period`
- `tests/` — validation, forces, integrators, physics laws, vectorized-vs-loop
- `scripts/` — `check_cases.py` (acceptance), `plot_energy.py` (figure)
- `figures/` — generated figures
- `Guides/` — spec, task list, expected output, build guide
