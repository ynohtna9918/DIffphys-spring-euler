"""Warp port of the mass-spring core (forward only, semi-implicit Euler, float64, 3-D)."""
import dataclasses

import numpy as np
import warp as wp

from .mass_spring import SpringSystem, total_energy


@wp.kernel
def external_forces(v: wp.array(dtype=wp.vec3d), inv_m: wp.array(dtype=wp.float64),
                    gravity: wp.vec3d, drag: wp.float64,
                    F: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    m = wp.float64(0.0)
    if inv_m[i] > wp.float64(0.0):
        m = wp.float64(1.0) / inv_m[i]
    F[i] = m * gravity - drag * v[i]          # "=" not "+=": this also resets F


@wp.kernel
def spring_forces(x: wp.array(dtype=wp.vec3d), v: wp.array(dtype=wp.vec3d),
                  springs: wp.array2d(dtype=wp.int32), rest: wp.array(dtype=wp.float64),
                  k_s: wp.float64, k_d: wp.float64,
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
    f = (k_s * (L - rest[s]) + k_d * rel_v) * n
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


class WarpModel:
    """The parts of a SpringSystem that never change, copied to one device."""

    def __init__(self, sys: SpringSystem, device=None):
        if sys.x.shape[1] != 3:
            raise ValueError(f"the Warp backend is 3-D only, got dim = {sys.x.shape[1]}")
        self.device = wp.get_device(device)
        self.N = sys.x.shape[0]
        self.M = sys.springs.shape[0]
        self.inv_m = wp.array(sys.inv_masses, dtype=wp.float64, device=self.device)
        self.springs = wp.array(sys.springs.astype(np.int32), dtype=wp.int32, device=self.device)
        self.rest = wp.array(sys.rest_lengths, dtype=wp.float64, device=self.device)
        self.k_s = sys.k_s
        self.k_d = sys.k_d
        self.drag = sys.drag
        self.gravity = wp.vec3d(*sys.gravity)


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
