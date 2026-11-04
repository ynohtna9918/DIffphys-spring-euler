"""Recover per-spring stiffness from a noisy trajectory (v1 result).

Run from the repo root:  python -m scripts.system_id [--seeds 8] [--device cuda:0]
Writes figures/system_id.png (seed 0) and prints recovered vs true stiffness.
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from src.system_id import K_TRUE, fit, make_problem

# Categorical slots 1-4 of the validated default palette (light surface).
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def plot(problem, k_hist, loss_hist, path):
    fig, (ax_loss, ax_k) = plt.subplots(1, 2, figsize=(11, 4))
    for ax in (ax_loss, ax_k):
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK)
        ax.set_xlabel("Adam iteration", color=INK)

    floor = 3 * problem["sigma"] ** 2                              # E|noise|^2 per particle
    ax_loss.semilogy(np.arange(1, len(loss_hist) + 1), loss_hist, color=COLOURS[0], linewidth=2)
    ax_loss.axhline(floor, color=MUTED, linewidth=1, linestyle="--")
    ax_loss.annotate(f"noise floor 3σ² = {floor:.1e}", (len(loss_hist), floor),
                     xytext=(0, 6), textcoords="offset points", ha="right", color=MUTED)
    ax_loss.set_ylabel("trajectory loss (m²)", color=INK)
    ax_loss.set_title("Loss", color=INK, loc="left")

    its = np.arange(len(k_hist))
    for s, (k_true, colour) in enumerate(zip(K_TRUE, COLOURS)):
        ax_k.axhline(k_true, color=colour, linewidth=1, linestyle=":", alpha=0.9)
        ax_k.plot(its, k_hist[:, s], color=colour, linewidth=2, label=f"spring {s}")
        ax_k.annotate(f"k{s} = {k_hist[-1, s]:.1f}  (true {k_true:.0f})", (its[-1], k_hist[-1, s]),
                      xytext=(6, 0), textcoords="offset points", va="center", color=INK)
    ax_k.set_ylabel("stiffness k (N/m)", color=INK)
    ax_k.set_title("Stiffness per spring (dotted = true value)", color=INK, loc="left")
    ax_k.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4,
                labelcolor=INK)

    err = np.abs(k_hist[-1] / K_TRUE - 1).max()
    fig.suptitle(f"System ID on a 5-particle chain: {problem['steps']} steps, noise σ = "
                 f"{problem['sigma'] * 1e3:.0f} mm, max stiffness error {err:.2%}", color=INK)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1, help="number of random initial guesses")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", default=None, help='Warp device, e.g. "cpu" or "cuda:0"')
    args = parser.parse_args()
    wp.config.quiet = True

    print(f"true k: {K_TRUE}")
    print(f"{'seed':>4} {'initial k':>32} {'recovered k':>36} {'max err':>8} {'time':>7}")
    errors = []
    for seed in range(args.seeds):
        problem = make_problem(seed, device=args.device)
        t0 = time.perf_counter()
        k_hist, loss_hist = fit(problem, args.iters)
        seconds = time.perf_counter() - t0
        err = np.abs(k_hist[-1] / K_TRUE - 1).max()
        errors.append(err)
        print(f"{seed:>4} {np.array2string(problem['k_init'], precision=1):>32} "
              f"{np.array2string(k_hist[-1], precision=2):>36} {err:>8.2%} {seconds:>6.1f}s")
        if seed == 0:
            out = Path("figures/system_id.png")
            plot(problem, k_hist, loss_hist, out)
            print(f"     final loss {loss_hist[-1]:.3e} (noise floor {3 * problem['sigma']**2:.1e});"
                  f" saved {out}")
    worst = max(errors)
    print(f"worst max error over {args.seeds} seed(s): {worst:.2%} -> "
          f"{'PASS' if worst < 0.05 else 'FAIL'} (< 5%)")


if __name__ == "__main__":
    main()
