"""Reproduce the acceptance cases in Guides/EXPECTED_OUTPUT.md.

Run from the repo root:  python -m scripts.check_cases
"""
import numpy as np

from src import (make_single_oscillator, make_chain, simulate, measure_period,
                 step_explicit_euler, step_semi_implicit_euler, step_midpoint)

DT = 0.001
G = [0.0, -9.81, 0.0]


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def case_a() -> bool:
    print("Case A: free oscillator")
    sys = make_single_oscillator()
    sys.x[1, 0] = 1.1
    xs, _, es = simulate(sys, DT, 4000, step_semi_implicit_euler)

    T = measure_period(xs[:, 1, 0] - 1.0, DT)
    T_exact = 2 * np.pi * np.sqrt(1.0 / 100.0)
    err = abs(T - T_exact) / T_exact
    dev = np.max(np.abs(es - es[0])) / es[0]
    pinned = np.array_equal(xs[-1, 0], xs[0, 0])

    print(f"  period: measured {T:.4f} s | analytic {T_exact:.4f} s | err {err:.2%}   {verdict(err < 0.03)}")
    print(f"  energy: E0 = {es[0]:.3f} J | max deviation over 4 s = {dev:.2%}   {verdict(dev < 0.05)}")
    print(f"  pinned particle 0 unchanged after 4000 steps   {verdict(pinned)}")
    return err < 0.03 and dev < 0.05 and pinned


def case_a2() -> bool:
    print("Case A2: integrator comparison (dt = 0.001, 2 s)")
    change = {}
    for name, stepper in [("explicit Euler", step_explicit_euler),
                          ("semi-implicit", step_semi_implicit_euler),
                          ("midpoint", step_midpoint)]:
        sys = make_single_oscillator()          # fresh system for every integrator
        sys.x[1, 0] = 1.1
        _, _, es = simulate(sys, DT, 2000, stepper)
        change[name] = (es[-1] - es[0]) / es[0]
        print(f"  {name:<15} E: {es[0]:.3f} -> {es[-1]:.3f}   change {change[name]:+.1%}")

    e, s, m = change.values()
    ok = e > 0 and abs(s) < 0.05 and abs(m) < abs(s)
    print(f"  explicit grows, semi-implicit bounded, midpoint tightest   {verdict(ok)}")
    return ok


def case_b() -> bool:
    print("Case B: hanging mass")
    sys = make_single_oscillator(k_d=10.0, drag=0.5, gravity=G)
    xs, _, _ = simulate(sys, DT, 20000, step_semi_implicit_euler)

    length = np.linalg.norm(xs[-2000:, 1] - xs[-2000:, 0], axis=1).mean()
    expected = 1.0 + 9.81 / 100.0               # L0 + m*g/k_s
    ok = abs(length - expected) < 0.005
    p = xs[-1, 1]

    print(f"  settled spring length (avg, last 2 s): {length:.4f} m | expected {expected:.4f} m   {verdict(ok)}")
    print(f"  final position of particle 1: ({p[0]:.2f}, {p[1]:.4f}, {p[2]:.2f})")
    return ok


def case_c() -> bool:
    print("Case C: chain of 5, settled spring lengths (top -> bottom)")
    sys = make_chain(5, k_d=5.0, drag=0.5, gravity=G)
    xs, _, _ = simulate(sys, DT, 40000, step_semi_implicit_euler)

    # spring k joins particle k and k+1: lengths for every step, shape (steps+1, 4)
    lengths = np.linalg.norm(xs[:, 1:] - xs[:, :-1], axis=2)[-2000:].mean(axis=0)

    ok = True
    for s, L in enumerate(lengths):
        below = 4 - s                           # masses hanging under spring s
        expected = 0.3 + below * 9.81 / 100.0
        good = abs(L - expected) < 0.01
        ok = ok and good
        print(f"  spring {s}: {L:.4f} m | analytic 0.3 + {below}*9.81/100 = {expected:.4f}   {verdict(good)}")
    print(f"  total chain length: {lengths.sum():.3f} m | bottom particle y = {xs[-1, -1, 1]:.3f}")
    return ok


def main() -> None:
    results = [case_a(), case_a2(), case_b(), case_c()]
    print("ALL PASS" if all(results) else "SOME CASES FAILED")


if __name__ == "__main__":
    main()
