import dataclasses

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from src.mass_spring import (SpringSystem, compute_forces, simulate, step_semi_implicit_euler,
                             total_energy, make_single_oscillator, make_chain)
from src.warp_sim import WarpModel, compute_forces_warp, simulate_warp
from tests.test_vectorized import random_system

DEVICES = ["cpu"] + (["cuda:0"] if wp.get_cuda_device_count() > 0 else [])
G = [0, -9.81, 0]


def warp_forces(sys, device):
    model = WarpModel(sys, device)
    x = wp.array(sys.x, dtype=wp.vec3d, device=device)
    v = wp.array(sys.v, dtype=wp.vec3d, device=device)
    F = wp.zeros(model.N, dtype=wp.vec3d, device=device)
    compute_forces_warp(model, x, v, F)
    return F.numpy()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", range(5))
def test_forces_match_numpy(seed, device):
    sys = random_system(seed)
    F_np = compute_forces(sys)
    assert np.abs(warp_forces(sys, device) - F_np).max() < 1e-13 * np.abs(F_np).max()


@pytest.mark.parametrize("device", DEVICES)
def test_zero_length_spring_and_no_springs(device):
    sys = make_single_oscillator()
    sys.x[1] = sys.x[0]
    assert np.array_equal(warp_forces(sys, device), np.zeros((2, 3)))
    sys = SpringSystem(x=[[0, 0, 0], [1, 0, 0]], v=[[0, 0, 0], [1, 0, 0]], inv_masses=[1, 1],
                       springs=[], rest_lengths=[], k_s=100, k_d=1, gravity=G, drag=0.5)
    assert np.allclose(warp_forces(sys, device), [[0, -9.81, 0], [-0.5, -9.81, 0]])


@pytest.mark.parametrize("device", DEVICES)
def test_one_step_case_a(device):
    sys = make_single_oscillator()
    sys.x[1, 0] = 1.1
    xs, vs, es = simulate_warp(sys, 0.001, 1, device)
    assert xs[1, 1, 0] == pytest.approx(1.09999, abs=1e-12)
    assert vs[1, 1, 0] == pytest.approx(-0.01, abs=1e-12)
    assert es[1] == pytest.approx(0.49995000500, abs=1e-10)
    assert sys.x[1, 0] == 1.1                      # input system untouched


@pytest.mark.parametrize("device", DEVICES)
def test_trajectory_matches_numpy(device):
    def chain():
        return make_chain(5, k_d=5.0, drag=0.5, gravity=G)
    xs_w, vs_w, es_w = simulate_warp(chain(), 0.001, 2000, device)
    xs_n, vs_n, es_n = simulate(chain(), 0.001, 2000, step_semi_implicit_euler)
    assert xs_w.shape == xs_n.shape and es_w.shape == es_n.shape
    assert np.abs(xs_w - xs_n).max() < 1e-10
    assert np.abs(vs_w - vs_n).max() < 1e-10


@pytest.mark.parametrize("device", DEVICES)
def test_pinned_particle_bit_identical(device):
    xs, _, _ = simulate_warp(make_chain(5, gravity=G), 0.001, 1000, device)
    assert np.array_equal(xs[-1, 0], xs[0, 0])


@pytest.mark.parametrize("device", DEVICES)
def test_per_spring_parameters_match_numpy(device):
    # Forces are a sum over springs, so the per-spring answer is: external forces
    # plus each spring on its own (as a one-spring NumPy system with its own k_s, k_d).
    sys = random_system(7)
    rng = np.random.default_rng(7)
    k_s = rng.uniform(10, 200, sys.springs.shape[0])
    k_d = rng.uniform(0, 5, sys.springs.shape[0])
    F_np = compute_forces(dataclasses.replace(sys, springs=np.zeros((0, 2), int), rest_lengths=[]))
    for s, (i, j) in enumerate(sys.springs):
        F_np += compute_forces(dataclasses.replace(
            sys, springs=[[i, j]], rest_lengths=[sys.rest_lengths[s]], k_s=k_s[s], k_d=k_d[s],
            gravity=np.zeros(3), drag=0.0))

    model = WarpModel(sys, device)
    model.k_s.assign(k_s)
    model.k_d.assign(k_d)
    x = wp.array(sys.x, dtype=wp.vec3d, device=device)
    v = wp.array(sys.v, dtype=wp.vec3d, device=device)
    F = wp.zeros(model.N, dtype=wp.vec3d, device=device)
    compute_forces_warp(model, x, v, F)
    assert np.abs(F.numpy() - F_np).max() < 1e-12 * np.abs(F_np).max()
