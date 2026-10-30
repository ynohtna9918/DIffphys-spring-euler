from dataclasses import dataclass
import numpy as np

@dataclass
class SpringSystem:
    x: np.ndarray
    v: np.ndarray
    inv_masses: np.ndarray
    springs: np.ndarray
    rest_lengths: np.ndarray
    k_s: float
    k_d: float
    gravity: np.ndarray
    drag: float

    def __post_init__(self):
        self.x = np.array(self.x, dtype = np.float64)
        self.v = np.array(self.v, dtype = np.float64)
        self.inv_masses = np.array(self.inv_masses, dtype = np.float64)
        self.rest_lengths = np.array(self.rest_lengths, dtype = np.float64)
        self.gravity = np.array(self.gravity, dtype = npfloat64)

        self.springs = np.array(self.springs, dtype = int)

        if self.springs.size == 0:
            self.springs = self.springs.reshape(0, 2)

        self.k_s = float(self.k_s)
        self.k_d = float(self.k_d)
        self.drag = float(self.drag)

    