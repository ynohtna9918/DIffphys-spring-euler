"""Energy behaviour of the three integrators on the same chain (Step 3.3).

Run from the repo root:  python -m scripts.plot_energy
Writes figures/energy_comparison.png and prints the cost table.
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np

from src import (make_chain, simulate,
                 step_explicit_euler, step_semi_implicit_euler, step_midpoint)

DT = 0.001
STEPS = 5000                               # 5 s
# (label, stepper, force evaluations per step, plot colour)
INTEGRATORS = [("explicit Euler", step_explicit_euler, 1, "tab:red"),
               ("semi-implicit Euler", step_semi_implicit_euler, 1, "tab:blue"),
               ("midpoint", step_midpoint, 2, "tab:green")]


def run_all():
    results = {}
    for label, stepper, evals, colour in INTEGRATORS:
        sys = make_chain(5, gravity=[0.0, -9.81, 0.0])   # no damping: energy should be constant
        start = time.perf_counter()
        _, _, es = simulate(sys, DT, STEPS, stepper)
        seconds = time.perf_counter() - start
        results[label] = (es, evals, seconds / STEPS, colour)
    return results


def plot(results, path):
    t = np.arange(STEPS + 1) * DT
    fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4))
    for label, (es, _, _, colour) in results.items():
        ax_all.plot(t, es - es[0], label=label, color=colour)
        if label != "explicit Euler":
            ax_zoom.plot(t, es - es[0], label=label, color=colour)
    ax_all.set_title("All three integrators")
    ax_zoom.set_title("Zoom: the two that do not drift")
    for ax in (ax_all, ax_zoom):
        ax.set_xlabel("time (s)")
        ax.set_ylabel("E(t) - E0  (J)")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.legend()
    fig.suptitle(f"Chain of 5, no damping, dt = {DT}")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)


def print_table(results):
    print(f"{'integrator':<22}{'force evals/step':>17}{'time/step (us)':>16}{'final dE (J)':>14}")
    for label, (es, evals, sec, _) in results.items():
        print(f"{label:<22}{evals:>17}{sec * 1e6:>16.1f}{es[-1] - es[0]:>+14.4f}")


if __name__ == "__main__":
    results = run_all()
    out = Path("figures/energy_comparison.png")
    plot(results, out)
    print_table(results)
    print(f"saved {out}")
