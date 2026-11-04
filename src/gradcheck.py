"""Check wp.Tape gradients against central finite differences (float64).

Used by tests/test_gradients.py and scripts/gradcheck.py.
"""
import dataclasses

import numpy as np

from .mass_spring import make_chain
from .warp_sim import WarpModel, loss_and_grad, loss_only, simulate_warp

EPSILONS = [1e-4, 1e-5, 1e-6, 1e-7]
NAMES = ["k_s", "k_d", "rest", "drag", "x0", "v0"]


def scenario(device=None, steps=200, dt=1e-3, seed=0):
    """A swinging 5-particle chain, and target positions from slightly different springs.

    Returns (model, x0, v0, dt, steps, targets). The model's k_s differs per
    spring and from the targets' k_s, so the loss and all gradients are nonzero.
    """
    rng = np.random.default_rng(seed)
    base = make_chain(5, k_d=2.0, drag=0.3, gravity=[0, -9.81, 0])
    x0 = base.x + 0.05 * rng.normal(size=base.x.shape)
    x0[0] = base.x[0]
    v0 = 0.3 * rng.normal(size=base.x.shape)
    v0[0] = 0.0
    truth = dataclasses.replace(base, k_s=120.0, x=x0, v=v0)
    targets, _, _ = simulate_warp(truth, dt, steps, device)
    model = WarpModel(base, device, requires_grad=True)
    model.k_s.assign(np.array([90.0, 110.0, 100.0, 95.0]))
    return model, x0, v0, dt, steps, targets


def central_difference(f, p, eps):
    """(f(p + eps e_k) - f(p - eps e_k)) / (2 eps) for every entry k of p."""
    g = np.zeros_like(p)
    for idx in np.ndindex(p.shape):
        a = p.copy()
        a[idx] += eps
        f_plus = f(a)
        a[idx] -= 2 * eps
        g[idx] = (f_plus - f(a)) / (2 * eps)
    return g


def loss_as_function_of(name, model, x0, v0, dt, steps, targets):
    """The loss as a numpy function of ONE parameter group, others held fixed."""
    if name == "x0":
        return lambda p: loss_only(model, p, v0, dt, steps, targets), x0.copy()
    if name == "v0":
        return lambda p: loss_only(model, x0, p, dt, steps, targets), v0.copy()
    param = model.params()[name]
    original = param.numpy().copy()

    def f(p):
        param.assign(p)
        value = loss_only(model, x0, v0, dt, steps, targets)
        param.assign(original)
        return value
    return f, original


def relative_error(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def gradcheck(device=None, steps=200, epsilons=EPSILONS):
    """{name: (|autodiff grad|, [relative error vs FD for each eps])}."""
    model, x0, v0, dt, steps, targets = scenario(device, steps)
    _, grads = loss_and_grad(model, x0, v0, dt, steps, targets)
    out = {}
    for name in NAMES:
        f, p = loss_as_function_of(name, model, x0, v0, dt, steps, targets)
        errors = [relative_error(grads[name], central_difference(f, p, eps)) for eps in epsilons]
        out[name] = (float(np.linalg.norm(grads[name])), errors)
    return out
