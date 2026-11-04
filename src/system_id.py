"""System identification: recover per-spring stiffness from a noisy trajectory.

A 5-particle chain hangs from a pin and falls under gravity from its rest
spacing. We record positions with Gaussian noise, then start from a wrong
guess of the four stiffnesses and run Adam on log(k), using gradients of the
trajectory loss from wp.Tape.
"""
import numpy as np
import warp as wp

from .mass_spring import make_chain
from .warp_sim import WarpModel, loss_and_grad, rollout

K_TRUE = np.array([150.0, 100.0, 80.0, 120.0])


def make_problem(seed=0, steps=500, sigma=2e-3, dt=1e-3, device=None):
    """Simulate the true chain, add noise, and draw an initial guess 0.5x-2x off per spring."""
    rng = np.random.default_rng(seed)
    base = make_chain(5, k_d=0.5, gravity=[0, -9.81, 0])
    model = WarpModel(base, device, requires_grad=True)
    model.k_s.assign(K_TRUE)
    x0 = wp.array(base.x, dtype=wp.vec3d, device=model.device)
    v0 = wp.array(base.v, dtype=wp.vec3d, device=model.device)
    xs, _ = rollout(model, x0, v0, dt, steps)
    clean = np.stack([x.numpy() for x in xs])                      # (steps+1, 5, 3)
    observed = clean + sigma * rng.normal(size=clean.shape)
    k_init = K_TRUE * np.exp(rng.uniform(np.log(0.5), np.log(2.0), K_TRUE.size))
    return dict(base=base, model=model, dt=dt, steps=steps, sigma=sigma,
                clean=clean, observed=observed, k_init=k_init)


def fit(problem, iters=200, lr=0.05):
    """Adam on log(k). Returns k_history (iters+1, M) and loss_history (iters,)."""
    model, base = problem["model"], problem["base"]
    log_k = np.log(problem["k_init"])
    m = np.zeros_like(log_k)
    v = np.zeros_like(log_k)
    beta1, beta2 = 0.9, 0.999
    k_history, loss_history = [np.exp(log_k)], []
    for it in range(1, iters + 1):
        model.k_s.assign(np.exp(log_k))
        loss, grads = loss_and_grad(model, base.x, base.v, problem["dt"], problem["steps"],
                                    problem["observed"])
        g = grads["k_s"] * np.exp(log_k)                           # chain rule: d/d(log k)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        log_k -= lr * (m / (1 - beta1**it)) / (np.sqrt(v / (1 - beta2**it)) + 1e-12)
        loss_history.append(loss)
        k_history.append(np.exp(log_k))
    return np.array(k_history), np.array(loss_history)
