from src.mass_spring import SpringSystem, compute_forces
import numpy as np


BASE = dict(x=[[0,0,0],[1.1,0,0]], v=[[0,0,0],[0,0,0]], inv_masses=[0,1],
            springs=[[0,1]], rest_lengths=[1.0], k_s=100, k_d=0,
            gravity=[0,0,0], drag=0)

def make(**change):
    return SpringSystem(**{**BASE, **change})

def test_stretched_spring_pulls_together():
    F = compute_forces(make())
    assert np.allclose(F, [[10, 0, 0], [-10, 0, 0]])

def test_compressed_spring_pushes_apart():
    F = compute_forces(make(x=[[0,0,0],[0.9,0,0]]))
    assert np.allclose(F, [[-10, 0, 0], [10, 0, 0]])

def test_gravity_skips_pinned_particle():
    F = compute_forces(make(gravity=[0, -9.81, 0]))
    assert np.allclose(F, [[10, 0, 0], [-10, -9.81, 0]])

def test_drag_opposes_own_velocity():
    F = compute_forces(make(x=[[0,0,0],[1.0,0,0]], v=[[0,0,0],[2,0,0]], drag=0.5))
    assert np.allclose(F[1], [-1, 0, 0])

def test_damping_uses_relative_velocity():
    # moving apart at 1 m/s at rest length: damping alone resists, f = k_d * 1 = 2
    F = compute_forces(make(x=[[0,0,0],[1.0,0,0]], v=[[0,0,0],[1,0,0]], k_d=2))
    assert np.allclose(F, [[2, 0, 0], [-2, 0, 0]])
    # both free, same velocity: spring length not changing -> no damping force
    F = compute_forces(make(x=[[0,0,0],[1.0,0,0]], v=[[3,0,0],[3,0,0]],
                            inv_masses=[1, 1], k_d=5))
    assert np.allclose(F, 0)

def test_third_law_on_random_chain():
    rng = np.random.default_rng(0)
    N = 6
    sys = SpringSystem(x=rng.normal(size=(N, 3)), v=rng.normal(size=(N, 3)),
                       inv_masses=np.ones(N), springs=[[k, k+1] for k in range(N-1)],
                       rest_lengths=np.full(N-1, 0.5), k_s=100, k_d=3,
                       gravity=[0, 0, 0], drag=0)
    F = compute_forces(sys)
    assert np.abs(F).max() > 1          # forces are real, not all zero
    assert np.allclose(F.sum(axis=0), 0, atol=1e-12)

def test_zero_length_spring_gives_no_nan():
    F = compute_forces(make(x=[[0,0,0],[0,0,0]]))
    assert np.isfinite(F).all()

def test_accumulator_is_fresh_each_call():
    sys = make(gravity=[0, -9.81, 0])
    assert np.array_equal(compute_forces(sys), compute_forces(sys))


import pytest
from src.mass_spring import (SpringSystem, total_energy,
                             step_explicit_euler, step_semi_implicit_euler, step_midpoint)

DT = 0.001
OMEGA = 10.0                       # sqrt(k/m) = sqrt(100/1)
PERIOD = 2 * np.pi / OMEGA         # 0.6283 s
ALL_STEPPERS = [step_explicit_euler, step_semi_implicit_euler, step_midpoint]


def oscillator(**change):
    """Case A: pin at origin, 1 kg particle at x = 1.1, spring k = 100, L0 = 1."""
    base = dict(x=[[0, 0, 0], [1.1, 0, 0]], v=[[0, 0, 0], [0, 0, 0]], inv_masses=[0, 1],
                springs=[[0, 1]], rest_lengths=[1.0], k_s=100, k_d=0,
                gravity=[0, 0, 0], drag=0)
    return SpringSystem(**{**base, **change})


def run(sys, stepper, steps, dt=DT):
    """Advance `steps` times; return positions (steps+1, N, dim) and energies (steps+1,)."""
    xs, es = [sys.x.copy()], [total_energy(sys)]
    for _ in range(steps):
        stepper(sys, dt)
        xs.append(sys.x.copy())
        es.append(total_energy(sys))
    return np.array(xs), np.array(es)


def zero_crossing_period(u, dt):
    """Mean time between upward zero crossings of u, with linear interpolation."""
    k = np.where((u[:-1] < 0) & (u[1:] >= 0))[0]
    t = (k + u[k] / (u[k] - u[k + 1])) * dt
    return np.mean(np.diff(t))


# 1. One step from the Case A start: the update ORDER shows up immediately.
@pytest.mark.parametrize("stepper, x1, e1", [
    (step_explicit_euler,      1.1,      0.50005),
    (step_semi_implicit_euler, 1.09999,  0.49995000500),
    (step_midpoint,            1.099995, 0.50000000125),
])
def test_one_step(stepper, x1, e1):
    sys = oscillator()
    stepper(sys, DT)
    assert sys.x[1, 0] == pytest.approx(x1, abs=1e-12)
    assert sys.v[1, 0] == pytest.approx(-0.01, abs=1e-12)
    assert total_energy(sys) == pytest.approx(e1, abs=1e-10)


# 2. Free fall (no springs): each method has its own exact closed form.
@pytest.mark.parametrize("stepper, factor", [
    (step_explicit_euler,      lambda n: n * (n - 1)),   # x uses OLD v: lags behind
    (step_semi_implicit_euler, lambda n: n * (n + 1)),   # x uses NEW v: runs ahead
    (step_midpoint,            lambda n: n * n),         # exact for constant acceleration
])
def test_free_fall_closed_form(stepper, factor):
    g, n = 9.81, 100
    sys = SpringSystem(x=[[0, 0, 0]], v=[[0, 0, 0]], inv_masses=[1], springs=[],
                       rest_lengths=[], k_s=0, k_d=0, gravity=[0, -g, 0], drag=0)
    for _ in range(n):
        stepper(sys, DT)
    assert sys.x[0, 1] == pytest.approx(-0.5 * g * DT**2 * factor(n), rel=1e-12)
    assert sys.v[0, 1] == pytest.approx(-g * n * DT, rel=1e-12)


# 3. Explicit Euler gains energy on EVERY step (the spiral).
def test_explicit_euler_energy_grows_every_step():
    _, es = run(oscillator(), step_explicit_euler, 1000)
    assert np.all(np.diff(es) > 0)
    assert es[-1] / es[0] == pytest.approx((1 + (OMEGA * DT) ** 2) ** 1000, rel=1e-9)  # ~1.105


# 4. Semi-implicit Euler: energy wobbles but stays in a narrow band.
def test_semi_implicit_energy_bounded():
    _, es = run(oscillator(), step_semi_implicit_euler, 4000)
    assert np.max(np.abs(es - es[0])) / es[0] < 0.01        # measured ~0.5%


# 5. Case A2 ranking after 2 s: explicit drifts most, midpoint least.
def test_energy_drift_ranking():
    drift = {}
    for stepper in ALL_STEPPERS:
        _, es = run(oscillator(), stepper, 2000)
        drift[stepper.__name__] = abs(es[-1] - es[0]) / es[0]
    e, s, m = (drift[f.__name__] for f in ALL_STEPPERS)
    assert e > 0.20                      # explicit: +22%
    assert m < s < e
    assert m < 1e-4                      # midpoint: flat to ~4 decimals


# 6. The oscillation period is right (semi-implicit and midpoint).
@pytest.mark.parametrize("stepper", [step_semi_implicit_euler, step_midpoint])
def test_period(stepper):
    xs, _ = run(oscillator(), stepper, 6300)                # ~10 periods
    T = zero_crossing_period(xs[:, 1, 0] - 1.0, DT)
    assert abs(T - PERIOD) / PERIOD < 0.005


# 7. Accuracy order: halving dt cuts the error ~2x (Euler) or ~4x (midpoint).
@pytest.mark.parametrize("stepper, ratio", [
    (step_semi_implicit_euler, 2.0),
    (step_midpoint,            4.0),
])
def test_convergence_order(stepper, ratio):
    t_end = 0.5
    def error(dt):
        xs, _ = run(oscillator(), stepper, round(t_end / dt), dt)
        exact = 1.0 + 0.1 * np.cos(OMEGA * t_end)
        return abs(xs[-1, 1, 0] - exact)
    assert error(0.002) / error(0.001) == pytest.approx(ratio, rel=0.15)


# 8. A pinned particle never moves, bit for bit, under every method.
@pytest.mark.parametrize("stepper", ALL_STEPPERS)
def test_pinned_particle_bit_identical(stepper):
    xs, _ = run(oscillator(gravity=[0, -9.81, 0]), stepper, 1000)
    assert np.array_equal(xs[-1, 0], xs[0, 0])


from src.mass_spring import SpringSystem, simulate, step_semi_implicit_euler, step_explicit_euler

def osc():
    return SpringSystem([[0,0,0],[1.1,0,0]], np.zeros((2,3)), [0,1], [[0,1]], [1.0], 100, 0, [0,0,0], 0)

xs, vs, es = simulate(osc(), 0.001, 1000, step_semi_implicit_euler)
print(xs.shape, vs.shape, es.shape)     # (1001, 2, 3) (1001, 2, 3) (1001,)
print(es[0], xs[0, 1, 0], xs[1, 1, 0])  # 0.5  1.1  1.09999

_, _, es2 = simulate(osc(), 0.001, 2000, step_explicit_euler)
print(f"{es2[-1] / es2[0] - 1:+.1%}")   # +22.1%  (Case A2, explicit)

def test_simulate_shapes_and_initial_state():
    xs, vs, es = simulate(oscillator(), 0.001, 1000, step_semi_implicit_euler)
    assert xs.shape == (1001, 2, 3) and vs.shape == (1001, 2, 3) and es.shape == (1001,)
    assert xs[0, 1, 0] == 1.1 and es[0] == pytest.approx(0.5)


from src.mass_spring import make_single_oscillator, make_chain

def test_factories():
    osc = make_single_oscillator()
    assert np.allclose(compute_forces(osc), 0)
    ch = make_chain(5)
    assert ch.springs.tolist() == [[0,1],[1,2],[2,3],[3,4]]
    assert np.allclose(ch.x[:, 1], [0, -0.3, -0.6, -0.9, -1.2])
    assert np.allclose(compute_forces(ch), 0)
