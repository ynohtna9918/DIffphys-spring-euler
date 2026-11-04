"""Warp port of the mass-spring core (semi-implicit Euler, float64, 3-D), forward and backward."""
import dataclasses

import numpy as np
import warp as wp

from .mass_spring import SpringSystem, total_energy


@wp.kernel
def external_forces(v: wp.array(dtype=wp.vec3d), inv_m: wp.array(dtype=wp.float64),
                    gravity: wp.vec3d, drag: wp.array(dtype=wp.float64),
                    F: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    m = wp.float64(0.0)
    if inv_m[i] > wp.float64(0.0):
        m = wp.float64(1.0) / inv_m[i]
    F[i] = m * gravity - drag[0] * v[i]       # "=" not "+=": this also resets F


@wp.kernel
def spring_forces(x: wp.array(dtype=wp.vec3d), v: wp.array(dtype=wp.vec3d),
                  springs: wp.array2d(dtype=wp.int32), rest: wp.array(dtype=wp.float64),
                  k_s: wp.array(dtype=wp.float64), k_d: wp.array(dtype=wp.float64),
                  F: wp.array(dtype=wp.vec3d)):
    s = wp.tid()
    i = springs[s, 0]
    j = springs[s, 1]
    d = x[j] - x[i]
    L = wp.length(d)
    if L < wp.float64(1.0e-12):
        return
    n = d / L
    rel_v = wp.dot(v[j] - v[i], n)
    f = (k_s[s] * (L - rest[s]) + k_d[s] * rel_v) * n
    wp.atomic_add(F, i, f)
    wp.atomic_sub(F, j, f)


@wp.kernel
def integrate_semi_implicit(x: wp.array(dtype=wp.vec3d), v: wp.array(dtype=wp.vec3d),
                            F: wp.array(dtype=wp.vec3d), inv_m: wp.array(dtype=wp.float64),
                            dt: wp.float64,
                            x_out: wp.array(dtype=wp.vec3d), v_out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    v_new = v[i] + F[i] * inv_m[i] * dt
    v_out[i] = v_new
    x_out[i] = x[i] + v_new * dt


@wp.kernel
def squared_error(x: wp.array(dtype=wp.vec3d), target: wp.array(dtype=wp.vec3d),
                  scale: wp.float64, loss: wp.array(dtype=wp.float64)):
    i = wp.tid()
    d = x[i] - target[i]
    wp.atomic_add(loss, 0, scale * wp.dot(d, d))


class WarpModel:
    """The parts of a SpringSystem that stay fixed during a rollout, copied to one device.

    k_s, k_d and rest are per-spring arrays and drag is a 1-element array, so that
    with requires_grad=True the tape collects d(loss)/d(each of them).
    """

    def __init__(self, sys: SpringSystem, device=None, requires_grad=False):
        if sys.x.shape[1] != 3:
            raise ValueError(f"the Warp backend is 3-D only, got dim = {sys.x.shape[1]}")
        self.device = wp.get_device(device)
        self.N = sys.x.shape[0]
        self.M = sys.springs.shape[0]
        self.inv_m = wp.array(sys.inv_masses, dtype=wp.float64, device=self.device)
        self.springs = wp.array(sys.springs.astype(np.int32).reshape(self.M, 2),
                                dtype=wp.int32, device=self.device)
        self.gravity = wp.vec3d(*sys.gravity)

        def param(values):
            return wp.array(np.asarray(values, dtype=np.float64), dtype=wp.float64,
                            device=self.device, requires_grad=requires_grad)
        self.k_s = param(np.full(self.M, sys.k_s))
        self.k_d = param(np.full(self.M, sys.k_d))
        self.rest = param(sys.rest_lengths)
        self.drag = param([sys.drag])

    def params(self):
        """The differentiable parameters, by name."""
        return {"k_s": self.k_s, "k_d": self.k_d, "rest": self.rest, "drag": self.drag}


def compute_forces_warp(model: WarpModel, x, v, F):
    wp.launch(external_forces, dim=model.N,
              inputs=[v, model.inv_m, model.gravity, model.drag], outputs=[F],
              device=model.device)
    wp.launch(spring_forces, dim=model.M,
              inputs=[x, v, model.springs, model.rest, model.k_s, model.k_d], outputs=[F],
              device=model.device)


def step_semi_implicit_warp(model: WarpModel, x, v, F, dt, x_out, v_out):
    compute_forces_warp(model, x, v, F)
    wp.launch(integrate_semi_implicit, dim=model.N,
              inputs=[x, v, F, model.inv_m, dt], outputs=[x_out, v_out],
              device=model.device)


def simulate_warp(sys: SpringSystem, dt: float, steps: int, device=None):
    """Same contract as simulate(..., step_semi_implicit_euler); sys is not modified."""
    model = WarpModel(sys, device)
    xs0 = np.zeros((steps + 1, model.N, 3))
    vs0 = np.zeros((steps + 1, model.N, 3))
    xs0[0], vs0[0] = sys.x, sys.v
    xs = wp.array(xs0, dtype=wp.vec3d, device=model.device)      # shape (steps+1, N)
    vs = wp.array(vs0, dtype=wp.vec3d, device=model.device)
    F = wp.zeros(model.N, dtype=wp.vec3d, device=model.device)

    for t in range(steps):
        step_semi_implicit_warp(model, xs[t], vs[t], F, dt, xs[t + 1], vs[t + 1])

    xs_np, vs_np = xs.numpy(), vs.numpy()                         # (steps+1, N, 3)
    energies = np.array([total_energy(dataclasses.replace(sys, x=xs_np[t], v=vs_np[t]))
                         for t in range(steps + 1)])
    return xs_np, vs_np, energies


# ---------------------------------------------------------------- gradients


def rollout(model: WarpModel, x0, v0, dt: float, steps: int):
    """Differentiable rollout: lists of steps+1 position and velocity arrays.

    Every step writes NEW arrays (x, v and F), so nothing the tape needs for the
    backward pass is ever overwritten. xs[0] and vs[0] are x0 and v0 themselves,
    so their gradients land in x0.grad and v0.grad.
    """
    grad = x0.requires_grad or v0.requires_grad or model.k_s.requires_grad

    def new():
        return wp.zeros(model.N, dtype=wp.vec3d, device=model.device, requires_grad=grad)

    xs, vs = [x0], [v0]
    for t in range(steps):
        F, x, v = new(), new(), new()
        step_semi_implicit_warp(model, xs[t], vs[t], F, dt, x, v)
        xs.append(x)
        vs.append(v)
    return xs, vs


def trajectory_loss(xs, targets, loss):
    """Add mean over steps and particles of |x - target|^2 into loss[0].

    targets: (len(xs), N, 3) numpy array or a list of wp arrays of the same length.
    """
    T, N = len(xs), xs[0].shape[0]
    scale = 1.0 / (T * N)
    for x, target in zip(xs, targets):
        if not isinstance(target, wp.array):
            target = wp.array(target, dtype=wp.vec3d, device=x.device)
        wp.launch(squared_error, dim=N, inputs=[x, target, scale], outputs=[loss],
                  device=x.device)


def loss_and_grad(model: WarpModel, x0_np, v0_np, dt, steps, targets):
    """One forward + backward pass. Returns (loss, {name: gradient as numpy})."""
    device = model.device
    x0 = wp.array(x0_np, dtype=wp.vec3d, device=device, requires_grad=True)
    v0 = wp.array(v0_np, dtype=wp.vec3d, device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
    for p in model.params().values():
        p.grad.zero_()

    tape = wp.Tape()
    with tape:
        xs, _ = rollout(model, x0, v0, dt, steps)
        trajectory_loss(xs, targets, loss)
    tape.backward(loss=loss)

    grads = {name: p.grad.numpy().copy() for name, p in model.params().items()}
    grads["x0"] = x0.grad.numpy().copy()
    grads["v0"] = v0.grad.numpy().copy()
    tape.zero()
    return float(loss.numpy()[0]), grads


def loss_only(model: WarpModel, x0_np, v0_np, dt, steps, targets):
    """Forward pass only (no tape): the same loss as loss_and_grad."""
    device = model.device
    x0 = wp.array(x0_np, dtype=wp.vec3d, device=device)
    v0 = wp.array(v0_np, dtype=wp.vec3d, device=device)
    loss = wp.zeros(1, dtype=wp.float64, device=device)
    xs, _ = rollout(model, x0, v0, dt, steps)
    trajectory_loss(xs, targets, loss)
    return float(loss.numpy()[0])
