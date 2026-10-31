from src.mass_spring import compute_forces, compute_forces_vec
from src.mass_spring import SpringSystem
import numpy as np
import pytest

def random_system(seed, N=30, M=80):
    rng = np.random.default_rng(seed)
    springs = rng.integers(0, N, size=(M, 2))
    springs = springs[springs[:, 0] != springs[:, 1]]               # no self-springs
    springs = np.vstack([springs, [[0, k] for k in range(1, N)]])   # particle 0 in many springs
    x = rng.normal(size=(N, 3))
    x[N - 1] = x[N - 2]                                              # one zero-length spring
    springs = np.vstack([springs, [[N - 2, N - 1]]])
    inv = rng.uniform(0.5, 2.0, size=N)
    inv[:3] = 0.0                                                    # a few pinned particles
    return SpringSystem(x=x, v=rng.normal(size=(N, 3)), inv_masses=inv, springs=springs,
                        rest_lengths=rng.uniform(0.2, 1.5, size=len(springs)),
                        k_s=50.0, k_d=3.0, gravity=[0, -9.81, 0], drag=0.4)


@pytest.mark.parametrize("seed", range(5))
def test_vectorized_matches_loop(seed):
    sys = random_system(seed)
    assert np.abs(compute_forces_vec(sys) - compute_forces(sys)).max() < 1e-12


def test_vectorized_handles_no_springs():
    sys = SpringSystem(x=[[0, 0, 0], [1, 0, 0]], v=[[0, 0, 0], [1, 0, 0]], inv_masses=[1, 1],
                       springs=[], rest_lengths=[], k_s=100, k_d=1,
                       gravity=[0, -9.81, 0], drag=0.5)
    assert np.allclose(compute_forces_vec(sys), compute_forces(sys))
