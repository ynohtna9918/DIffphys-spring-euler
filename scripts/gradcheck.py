"""Print autodiff-vs-finite-difference relative errors over the eps sweep.

Run from the repo root:  python -m scripts.gradcheck [--device cuda:0] [--steps 200]
"""
import argparse

import warp as wp

from src.gradcheck import EPSILONS, gradcheck


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help='Warp device, e.g. "cpu" or "cuda:0"')
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    wp.config.quiet = True

    print(f"device: {wp.get_device(args.device)}, steps: {args.steps}, float64")
    print(f"{'param':>6} {'|grad|':>10}" + "".join(f" {'eps=' + format(e, '.0e'):>11}" for e in EPSILONS)
          + "      best")
    worst = 0.0
    for name, (norm, errors) in gradcheck(args.device, args.steps).items():
        best = min(errors)
        worst = max(worst, best)
        print(f"{name:>6} {norm:>10.3e}" + "".join(f" {e:>11.1e}" for e in errors) + f" {best:>9.1e}")
    print(f"worst best-eps relative error: {worst:.1e} -> {'PASS' if worst < 1e-5 else 'FAIL'} (< 1e-5)")


if __name__ == "__main__":
    main()
