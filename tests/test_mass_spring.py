from src.mass_spring import SpringSystem, compute_forces
import numpy as np

"""
Test: nomalization and validation
"""
base = dict(x=[[0,0,0],[1.1,0,0]], v=[[0,0,0],[0,0,0]], inv_masses=[0,1],
            springs=[[0,1]], rest_lengths=[1.0], k_s=100, k_d=0,
            gravity=[0,0,0], drag=0)
def bad(**change):
    try:
        SpringSystem(**{**base, **change}); print("NO ERROR (check missing!)")
    except ValueError as e:
        print("ValueError:", e)

bad(x=[0, 0, 0])                         # 1: x not 2-D
bad(v=[[0,0,0]])                         # 2: v shape (1,3) vs (2,3)
bad(inv_masses=[0, 1, 1])                # 3: inv_masses (3,) vs N=2
bad(inv_masses=[0, -1])                  # 4: negative inverse mass
bad(springs=[0, 1])                      # 5: springs not (M, 2)
bad(springs=[[0, 2]])                    # 6: index 2 out of range for N=2
bad(springs=[[1, 1]])                    # 7: self-spring
bad(rest_lengths=[1.0, 1.0])             # 8: rest_lengths (2,) vs M=1
bad(rest_lengths=[0.0])                  # 9: rest length not > 0
bad(gravity=[0, -9.81])                  # 10: gravity (2,) vs dim=3
bad(k_s=-5)                              # 11: negative stiffness
bad(x=[[0,0,0],[float("nan"),0,0]])      # 12: NaN in x


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