# diffphys — Expected Inputs & Outputs (acceptance reference)

This file defines what your program must produce. It is a black-box contract:
given the inputs below, a correct implementation lands within the stated
tolerances of these numbers. All reference numbers were produced by an
independent reference run — use them to check YOUR implementation, not to
copy one.

Conventions: SI units. 3D state, y is up, gravity = [0, -9.81, 0].
Integrator is semi-implicit Euler unless a case says otherwise.

---

## Case A — Free oscillator (proves the core)

### Inputs
| Item | Value |
|---|---|
| Particles | 2: particle 0 pinned at (0,0,0); particle 1 free at (1.1, 0, 0) |
| Masses | particle 1: m = 1 kg (particle 0: inv_mass = 0) |
| Spring | one spring (0, 1), rest length L0 = 1.0 m, k_s = 100 |
| Damping / drag | k_d = 0, drag = 0 |
| Gravity | off |
| Initial velocity | 0 (spring starts stretched by 0.1 m) |
| dt / duration | 0.001 s / 4 s (4000 steps) |

### Expected behavior
Particle 1 oscillates sinusoidally along the x-axis about x = 1.0,
amplitude 0.1 m, forever (no damping). Energy stays at ~0.5 J.

### Expected terminal output (your check run should print numbers like these)
```
Case A: free oscillator
  period: measured 0.6284 s | analytic 2*pi*sqrt(m/k) = 0.6283 s | err 0.01%   PASS (< 3%)
  energy: E0 = 0.500 J | max deviation over 4 s = 0.51%                        PASS (< 5%)
  pinned particle 0: position unchanged after 4000 steps                       PASS
```

---

## Case A2 — Same oscillator, three integrators (energy behavior)

### Inputs
Identical to Case A, but duration 2 s (2000 steps), run once per integrator.

### Expected terminal output
```
Integrator comparison (dt = 0.001, 2 s, E0 ~ 0.500 J)
  explicit Euler     E: 0.500 -> 0.611   change +22.1%   (must GROW: the spiral)
  semi-implicit      E: 0.500 -> 0.498   change  -0.4%   (bounded, no drift)
  midpoint           E: 0.500 -> 0.500   change   0.0%   (tightest of the three)
```
Rule of thumb: explicit visibly gains energy; semi-implicit wobbles inside a
few percent; midpoint is flat to ~3 decimals at this dt. If your explicit
Euler does NOT gain energy, its update order is wrong (see SPEC.md).

---

## Case B — Hanging mass settles at mg/k

### Inputs
| Item | Value |
|---|---|
| Particles | 2: particle 0 pinned at (0,0,0); particle 1 free at (1.0, 0, 0) |
| Masses | particle 1: m = 1 kg |
| Spring | (0, 1), L0 = 1.0 m, k_s = 100 |
| Damping / drag | k_d = 10, drag = 0.5 (so it actually settles) |
| Gravity | ON: [0, -9.81, 0] |
| dt / duration | 0.001 s / 20 s |

### Expected behavior
The mass swings down, oscillates, and comes to rest hanging straight below
the pin. Spring force k_s * stretch balances gravity m*g, so the settled
length is L0 + m*g/k_s = 1.0 + 9.81/100 = 1.0981 m.

### Expected terminal output
```
Case B: hanging mass
  settled spring length (avg, last 2 s): 1.0981 m | expected 1.0981 m   PASS
  final position of particle 1: (0.00, -1.0981, 0.00)
```

---

## Case C — Chain under gravity (top spring stretches most)

### Inputs
| Item | Value |
|---|---|
| Particles | 5 in a vertical line, spacing 0.3 m; particle 0 pinned at origin |
| Masses | particles 1-4: m = 1 kg each |
| Springs | (0,1), (1,2), (2,3), (3,4); L0 = 0.3 m each; k_s = 100 |
| Damping / drag | k_d = 5, drag = 0.5 |
| Gravity | ON: [0, -9.81, 0] |
| dt / duration | 0.001 s / 40 s |

### Expected behavior
The chain hangs straight down and settles. Spring j (from the top) carries
the weight of every particle below it, so the top spring stretches most:
stretch_j = (number of masses below) * m*g / k_s.

### Expected terminal output
```
Case C: chain of 5, settled spring lengths (top -> bottom)
  spring 0: 0.6924 m | analytic 0.3 + 4*9.81/100 = 0.6924   PASS
  spring 1: 0.5943 m | analytic 0.3 + 3*9.81/100 = 0.5943   PASS
  spring 2: 0.4962 m | analytic 0.3 + 2*9.81/100 = 0.4962   PASS
  spring 3: 0.3981 m | analytic 0.3 + 1*9.81/100 = 0.3981   PASS
  total chain length: 2.181 m | bottom particle y = -2.181
```

---

## Acceptance summary

Your core is correct when ALL of these hold simultaneously:

1. Case A period within 3% of 2*pi*sqrt(m/k) = 0.6283 s.
2. Case A energy: semi-implicit stays within +/-5% of 0.500 J over 1000+
   steps; explicit Euler energy strictly grows (Case A2).
3. Case B settles at spring length L0 + m*g/k_s = 1.0981 m (+/- 0.005).
4. Case C per-spring lengths match the analytic table above (+/- 0.01),
   top spring longest.
5. A pinned particle's position is bit-identical after any number of steps,
   and internal spring forces sum to zero with gravity/drag off (TASKS 2.5).
