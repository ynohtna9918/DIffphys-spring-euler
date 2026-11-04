import numpy as np
import pytest

pytest.importorskip("warp")

from src.system_id import K_TRUE, fit, make_problem


# A shorter version of scripts/system_id.py: 120 Adam iterations instead of 200.
# (A 300-step window is NOT enough: the lower springs then come back 25-64% off.)
def test_recovers_stiffness_from_noisy_trajectory():
    problem = make_problem(seed=1, steps=500)
    assert np.abs(problem["k_init"] / K_TRUE - 1).max() > 0.5     # starts up to 1.8x off
    k_hist, loss_hist = fit(problem, iters=120)
    assert loss_hist[-1] < 0.1 * loss_hist[0]
    assert np.abs(k_hist[-1] / K_TRUE - 1).max() < 0.05           # measured 0.64%
