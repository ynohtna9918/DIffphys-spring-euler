"""One-figure overview of the project for the README: every number is measured live.

Run from the repo root:  python -m scripts.plot_overview
Writes figures/overview.png (about 40 s on a laptop CPU).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from scripts.bench_warp import time_numpy, time_warp
from src import (make_chain, make_single_oscillator, simulate,
                 step_explicit_euler, step_semi_implicit_euler, step_midpoint)
from src.gradcheck import gradcheck
from src.system_id import K_TRUE, fit, make_problem
from src.warp_sim import simulate_warp

DT = 0.001
G = [0.0, -9.81, 0.0]
# Validated default palette: categorical slots 1-4, a one-hue blue ramp, neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE_RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def style(ax, title, xlabel, ylabel):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.set_title(title, color=INK, loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)


def label_end(ax, x, y, text, colour=INK, dx=4, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                va="center", color=colour, fontsize=9)


def panel_chain(ax):
    """(a) The simulated system: a 5-particle chain swinging under gravity."""
    sys = make_chain(5, k_s=2000.0, gravity=G)
    sys.v[1:, 0] = np.linspace(0.5, 2.0, 4)                     # push it sideways
    xs, _, _ = simulate(sys, DT, 600, step_semi_implicit_euler)
    snapshots = [0, 100, 200, 300, 600]                            # the first forward swing
    for colour, t in zip(BLUE_RAMP[1:], snapshots):
        ax.plot(xs[t, :, 0], xs[t, :, 1], "-o", color=colour, linewidth=2, markersize=6,
                markeredgecolor="white", markeredgewidth=1)
        label_end(ax, xs[t, -1, 0], xs[t, -1, 1], f"{t * DT:.2g} s", INK, dx=-8, dy=-14)
    ax.plot(0, 0, marker="^", color=INK, markersize=10)
    label_end(ax, 0, 0, "pinned", MUTED, dx=8, dy=4)
    ax.text(0.98, 0.62, "k = 2000 N/m\nm = 1 kg, dt = 1 ms\ndarker = later",
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=9)
    ax.set_xlim(-0.25, 0.95)
    ax.set_ylim(-1.45, 0.15)
    ax.set_aspect("equal", adjustable="datalim")
    style(ax, "a  The system: 5 masses, 4 springs", "x (m)", "y (m, up)")


def panel_energy(ax):
    """(b) Case A2: energy drift of the three integrators over 2 s."""
    t = np.arange(2001) * DT
    offsets = {"explicit Euler": 0, "semi-implicit Euler": -7, "midpoint": 7}
    for (name, stepper), colour in zip([("explicit Euler", step_explicit_euler),
                                        ("semi-implicit Euler", step_semi_implicit_euler),
                                        ("midpoint", step_midpoint)], SERIES):
        sys = make_single_oscillator()
        sys.x[1, 0] = 1.1
        _, _, es = simulate(sys, DT, 2000, stepper)
        drift = 100 * (es / es[0] - 1)
        ax.plot(t, drift, color=colour, linewidth=2, label=name)
        label_end(ax, t[-1], drift[-1], f"{drift[-1]:+.1f}%", dy=offsets[name])
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax.set_xlim(0, 2.35)
    style(ax, "b  Energy drift by integrator", "time (s)", "E / E0 − 1  (%)")


def panel_warp(ax):
    """(c) The Warp port reproduces the NumPy reference trajectory."""
    def chain():
        return make_chain(5, k_d=5.0, drag=0.5, gravity=G)
    xs_n, vs_n, _ = simulate(chain(), DT, 2000, step_semi_implicit_euler)
    xs_w, vs_w, _ = simulate_warp(chain(), DT, 2000, "cpu")
    t = np.arange(2001) * DT
    ax.plot(t, xs_n[:, -1, 1], color=SERIES[0], linewidth=2, label="NumPy (reference)")
    ax.plot(t[::80], xs_w[::80, -1, 1], "o", color=SERIES[1], markersize=6,
            markeredgecolor="white", markeredgewidth=1, label="Warp kernels")
    dx, dv = np.abs(xs_w - xs_n).max(), np.abs(vs_w - vs_n).max()
    dx_text = "0 (bit-identical)" if dx == 0 else f"{dx:.1e} m"
    ax.text(0.97, 0.95, f"max |Δx| = {dx_text}\nmax |Δv| = {dv:.1e} m/s",
            transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    style(ax, "c  Warp matches NumPy (2,000 steps)", "time (s)", "bottom mass height y (m)")
    return dv


def panel_gradients(ax):
    """(d) Autodiff gradients vs central finite differences, best eps per parameter."""
    report = gradcheck("cpu", 200)
    names = list(report)
    best = [min(errors) for _, errors in report.values()]
    y = np.arange(len(names))
    ax.barh(y, best, color=SERIES[0], height=0.6)
    for yi, b in zip(y, best):
        label_end(ax, b, yi, f"{b:.0e}")
    ax.axvline(1e-5, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("target < 1e-5", (1e-5, len(names) - 1), xytext=(-4, -18),
                textcoords="offset points", ha="right", color=MUTED, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(1e-11, 1e-4)
    ax.set_yticks(y, ["k (stiffness)", "c (spring damping)", "L0 (rest length)", "drag",
                      "x0 (initial pos.)", "v0 (initial vel.)"])
    ax.invert_yaxis()
    style(ax, "d  Gradients match finite differences", "relative error (lower is better)", "")
    return max(best)


def panel_speed(ax):
    """(e) Time per step vs number of particles."""
    sizes = [(10, 2000), (1_000, 2000), (100_000, 200), (1_000_000, 20)]
    n = np.array([s for s, _ in sizes])
    t_np = np.array([time_numpy(s, k) for s, k in sizes]) * 1e6
    t_wp = np.array([time_warp(s, k, "cpu") for s, k in sizes]) * 1e6
    for times, name, colour in [(t_np, "NumPy", SERIES[0]), (t_wp, "Warp, CPU", SERIES[1])]:
        ax.loglog(n, times, "-o", color=colour, linewidth=2, markersize=6,
                  markeredgecolor="white", markeredgewidth=1, label=name)
    label_end(ax, n[-1], t_np[-1], f"{t_np[-1] / 1e3:.0f} ms", dx=6)
    label_end(ax, n[-1], t_wp[-1], f"{t_wp[-1] / 1e3:.1f} ms", dx=6)
    ax.text(0.03, 0.95, f"{t_np[-1] / t_wp[-1]:.0f}× faster at 1M particles\n"
            "(GPU: to be measured on RTX 5070)", transform=ax.transAxes, va="top",
            color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    ax.set_xlim(5, 1e7)
    style(ax, "e  Time per step (float64)", "particles", "µs per step")
    return t_np[-1] / t_wp[-1]


def panel_sysid(ax):
    """(f) System ID: per-spring stiffness error during the fit."""
    problem = make_problem(seed=0)
    k_hist, _ = fit(problem, iters=200)
    err = 100 * np.abs(k_hist / K_TRUE - 1)
    its = np.arange(len(k_hist))
    for s, colour in enumerate(SERIES):
        ax.semilogy(its, err[:, s], color=colour, linewidth=2, label=f"spring {s} (k = {K_TRUE[s]:.0f})")
        label_end(ax, its[-1], err[-1, s], f"{err[-1, s]:.2g}%")
    ax.axhline(5, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("target < 5%", (its[-1], 5), xytext=(0, -11), textcoords="offset points",
                ha="right", color=MUTED, fontsize=9)
    ax.set_ylim(1e-4, 1e4)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right", ncol=2)
    ax.set_xlim(0, 235)
    style(ax, "f  Recovering stiffness from noisy data", "Adam iteration",
          "|k − k_true| / k_true  (%)")
    return err[-1].max()


def main():
    wp.config.quiet = True
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    panel_chain(axes[0, 0])
    panel_energy(axes[0, 1])
    dv = panel_warp(axes[0, 2])
    worst_grad = panel_gradients(axes[1, 0])
    speedup = panel_speed(axes[1, 1])
    worst_k = panel_sysid(axes[1, 2])
    fig.suptitle("diffphys: a float64 mass–spring simulator, ported to NVIDIA Warp and made "
                 "differentiable", color=INK, fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5, w_pad=2.5)
    out = Path("figures/overview.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Warp vs NumPy max |dv|: {dv:.1e}; worst gradient error: {worst_grad:.1e}; "
          f"Warp CPU speed-up at 1M: {speedup:.1f}x; worst stiffness error: {worst_k:.2f}%")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
