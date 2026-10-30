# diffphys — Differentiable Physics Simulation

GPU-accelerated differentiable physics simulation: infra layer for physical AI
(simulation, Sim2Real). Main line of the 2026.10–2028.10 roadmap toward a
Hong Kong PhD (late 2027 applications).

## Plan (W1: Oct 19–25, 2026)

- [ ] Read Witkin & Baraff *Physically Based Modeling* Part B/C
- [ ] NumPy mass-spring simulation (semi-implicit Euler) running
- [ ] Public repo skeleton + README  ← you are here
- [ ] Book CS + AMATH advisor meetings

Later: port to NVIDIA Warp, single-scene differentiability, trajectory
optimization, Newton PRs, gradient-quality benchmark (see roadmap doc).

## Layout

- `src/` — simulation code (start: `mass_spring.py`)
- `notebooks/` — exploration / figures
- `tests/` — numerical checks (NumPy vs finite differences later)

## Repro

python -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib
python src/mass_spring.py
