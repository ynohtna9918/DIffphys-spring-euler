"""
Test: normalization and validation in SpringSystem.__post_init__
"""
import numpy as np
import pytest
from src.mass_spring import SpringSystem

BASE = dict(x=[[0, 0, 0], [1.1, 0, 0]], v=[[0, 0, 0], [0, 0, 0]], inv_masses=[0, 1],
            springs=[[0, 1]], rest_lengths=[1.0], k_s=100, k_d=0,
            gravity=[0, 0, 0], drag=0)


def make(**change):
    return SpringSystem(**{**BASE, **change})


@pytest.mark.parametrize("change, message", [
    (dict(x=[0, 0, 0]),                       "x must be 2D"),                  # 1: x not 2-D
    (dict(v=[[0, 0, 0]]),                     "same shape"),                    # 2: v shape (1,3) vs (2,3)
    (dict(inv_masses=[0, 1, 1]),              "inv_masses must have shape"),    # 3: inv_masses (3,) vs N=2
    (dict(inv_masses=[0, -1]),                "inv_masses must be >= 0"),       # 4: negative inverse mass
    (dict(springs=[0, 1]),                    "springs must have shape"),       # 5: springs not (M, 2)
    (dict(springs=[[0, 2]]),                  "out of range"),                  # 6: index 2 out of range for N=2
    (dict(springs=[[1, 1]]),                  "connect a particle to itself"),  # 7: self-spring
    (dict(rest_lengths=[1.0, 1.0]),           "rest_lengths must have shape"),  # 8: rest_lengths (2,) vs M=1
    (dict(rest_lengths=[0.0]),                "rest_lengths must be > 0"),      # 9: rest length not > 0
    (dict(gravity=[0, -9.81]),                "gravity must have shape"),       # 10: gravity (2,) vs dim=3
    (dict(k_s=-5),                            "k_s must be >= 0"),              # 11: negative stiffness
    (dict(x=[[0, 0, 0], [np.nan, 0, 0]]),     "x contains NaN or inf"),         # 12: NaN in x
])
def test_constructor_rejects_bad_input(change, message):
    with pytest.raises(ValueError, match=message):
        make(**change)


def test_constructor_normalizes_types():
    sys = make(x=[[0, 0, 0], [1, 0, 0]], k_s=100)
    assert sys.x.dtype == np.float64
    assert sys.springs.shape == (1, 2)
    assert type(sys.k_s) is float


def test_constructor_accepts_no_springs():
    sys = make(springs=[], rest_lengths=[])
    assert sys.springs.shape == (0, 2)


def test_pinned_velocity_is_zeroed():
    sys = make(v=[[5, 5, 5], [1, 0, 0]])
    assert np.array_equal(sys.v, [[0, 0, 0], [1, 0, 0]])


def test_constructor_copies_input_arrays():
    x0 = np.array([[0.0, 0, 0], [1.1, 0, 0]])
    sys = make(x=x0)
    sys.x[1, 0] = 9.0
    assert x0[1, 0] == 1.1
