import numpy as np
import pytest

wp = pytest.importorskip("warp")

from src.gradcheck import NAMES, gradcheck, scenario
from src.warp_sim import WarpModel, loss_and_grad, loss_only, rollout, simulate_warp
from src import make_chain

DEVICES = ["cpu"] + (["cuda:0"] if wp.get_cuda_device_count() > 0 else [])


@pytest.fixture(scope="module", params=DEVICES)
def report(request):
    return gradcheck(request.param)


# 1. Autodiff through wp.Tape matches central finite differences (float64).
@pytest.mark.parametrize("name", NAMES)
def test_gradient_matches_finite_differences(report, name):
    norm, errors = report[name]
    assert norm > 0                                  # the gradient is really there
    assert min(errors) < 1e-5                        # measured 1e-9 .. 1e-8 at the best eps


# 2. The differentiable rollout is the same simulation as simulate_warp.
@pytest.mark.parametrize("device", DEVICES)
def test_rollout_matches_simulate_warp(device):
    sys = make_chain(5, k_d=2.0, drag=0.3, gravity=[0, -9.81, 0])
    model = WarpModel(sys, device, requires_grad=True)
    x0 = wp.array(sys.x, dtype=wp.vec3d, device=device, requires_grad=True)
    v0 = wp.array(sys.v, dtype=wp.vec3d, device=device, requires_grad=True)
    xs, _ = rollout(model, x0, v0, 1e-3, 300)
    ref, _, _ = simulate_warp(sys, 1e-3, 300, device)
    assert np.array_equal(np.stack([x.numpy() for x in xs]), ref)


# 3. Calling loss_and_grad twice gives the same answer: gradients do not pile up.
@pytest.mark.parametrize("device", DEVICES)
def test_gradients_reset_between_calls(device):
    model, x0, v0, dt, steps, targets = scenario(device, steps=50)
    loss1, g1 = loss_and_grad(model, x0, v0, dt, steps, targets)
    loss2, g2 = loss_and_grad(model, x0, v0, dt, steps, targets)
    assert loss1 == loss2 == loss_only(model, x0, v0, dt, steps, targets)
    for name in g1:
        assert np.array_equal(g1[name], g2[name])
