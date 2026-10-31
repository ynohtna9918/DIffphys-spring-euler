import numpy as np
import pytest
from src.mass_spring import (SpringSystem, simulate, make_single_oscillator, measure_period,
                             step_explicit_euler, step_semi_implicit_euler, step_midpoint)

DT = 0.001
G = 9.81


# 1. Momentum: with no external forces, sum(m * v) never changes.
@pytest.mark.parametrize("stepper", [step_explicit_euler, step_semi_implicit_euler, step_midpoint])
def test_momentum_conserved_free_pair(stepper):
    inv = np.array([1.0, 0.5])                     # masses 1 kg and 2 kg
    m = 1.0 / inv
    sys = SpringSystem(x=[[0, 0, 0], [1.2, 0.3, 0]],
                       v=[[0.5, -1.0, 0.2], [-0.3, 0.4, 0.1]],
                       inv_masses=inv, springs=[[0, 1]], rest_lengths=[1.0],
                       k_s=100, k_d=2, gravity=[0, 0, 0], drag=0)
    xs, vs, _ = simulate(sys, DT, 2000, stepper)

    P = np.sum(m[:, None] * vs, axis=1)            # total momentum at every step, (steps+1, 3)
    assert np.abs(P - P[0]).max() < 1e-10

    com = np.sum(m[:, None] * xs, axis=1) / m.sum()        # centre of mass, (steps+1, 3)
    t = np.arange(len(xs))[:, None] * DT
    assert np.allclose(com, com[0] + P[0] / m.sum() * t, atol=1e-10)

    rel = np.linalg.norm(xs[:, 1] - xs[:, 0], axis=1)      # the pair really does oscillate
    assert rel.max() - rel.min() > 0.1


# 2. Damping: peaks shrink exactly along A * exp(-zeta * omega * t).
def test_damped_oscillator_envelope_and_period():
    k, m, k_d = 100.0, 1.0, 1.0
    omega = np.sqrt(k / m)                         # 10 rad/s
    zeta = k_d / (2 * np.sqrt(k * m))              # 0.05
    sys = make_single_oscillator(k_s=k, m=m, k_d=k_d)
    sys.x[1, 0] = 1.1
    xs, _, _ = simulate(sys, DT, 3000, step_semi_implicit_euler)
    u = xs[:, 1, 0] - 1.0
    t = np.arange(len(u)) * DT

    peaks = np.where((u[1:-1] > u[:-2]) & (u[1:-1] >= u[2:]) & (u[1:-1] > 0))[0] + 1
    assert len(peaks) >= 4
    envelope = 0.1 * np.exp(-zeta * omega * t[peaks])
    assert np.allclose(u[peaks], envelope, rtol=0.02)

    T_damped = 2 * np.pi / (omega * np.sqrt(1 - zeta**2))   # 0.6291 s, slower than 0.6283
    assert measure_period(u, DT) == pytest.approx(T_damped, rel=1e-3)


# 3. Statics: a hanging mass settles where spring force = weight, for several m and k.
@pytest.mark.parametrize("m, k", [(1.0, 100.0), (2.0, 50.0), (0.5, 400.0)])
def test_static_equilibrium(m, k):
    sys = make_single_oscillator(k_s=k, m=m, k_d=10.0, gravity=[0, -G, 0])
    sys.x[1] = [0, -1.0, 0]                        # start straight below the pin, unstretched
    xs, vs, _ = simulate(sys, DT, 8000, step_semi_implicit_euler)

    extension = np.linalg.norm(xs[-1, 1] - xs[-1, 0]) - 1.0
    expected = m * G / k
    assert abs(extension - expected) / expected < 0.01
    assert np.linalg.norm(vs[-1, 1]) < 1e-4        # it has actually stopped
    assert np.allclose(xs[-1, 1, [0, 2]], 0)       # straight below the pin
