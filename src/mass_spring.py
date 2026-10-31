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

        #normalize#
        self.x = np.array(self.x, dtype = np.float64)
        self.v = np.array(self.v, dtype = np.float64)
        self.inv_masses = np.array(self.inv_masses, dtype = np.float64)
        self.rest_lengths = np.array(self.rest_lengths, dtype = np.float64)
        self.gravity = np.array(self.gravity, dtype = np.float64)

        self.springs = np.array(self.springs, dtype = int)

        if self.springs.size == 0:
            self.springs = self.springs.reshape(0, 2)

        self.k_s = float(self.k_s)
        self.k_d = float(self.k_d)
        self.drag = float(self.drag)

        #validate
        if self.x.ndim != 2:
            raise ValueError(f"x must be 2D (N,dim), got shape {self.x.shape}")
        N, dim = self.x.shape

        if self.v.shape != self.x.shape:
            raise ValueError(f"v and x should have the same shape, got v:{self.v.shape}, x:{self.x.shape}")

        if self.inv_masses.shape != (N,):
            raise ValueError(f"inv_masses must have shape({N}, ), got {self.inv_masses.shape}")
        if (self.inv_masses < 0).any():
            raise ValueError(f"inv_masses must be >= 0, got {self.inv_masses}")

        if self.springs.ndim != 2 or self.springs.shape[1] != 2:
            raise ValueError(f"springs must have shape (M, 2), got {self.springs.shape}")
        M = self.springs.shape[0]

        out_of_range = (self.springs < 0) | (self.springs >= N)
        if out_of_range.any():
            rows = np.argwhere(out_of_range.any(axis = 1)).ravel()
            raise ValueError(f"spring index out of range [0, {N}): spring {rows.tolist()} = {self.springs[rows].tolist()}")

        self_loops = self.springs[:, 0] == self.springs[:, 1]
        if self_loops.any():
            rows = np.argwhere(self_loops).ravel()
            raise ValueError(f"spring(s) {rows.tolist()} connect a particle to itself")

        if self.rest_lengths.shape != (M,):
            raise ValueError(f"rest_lengths must have shape ({M},), got {self.rest_lengths.shape}")
        if (self.rest_lengths <= 0).any():
            raise ValueError(f"rest_lengths must be > 0, got {self.rest_lengths}")

        if self.gravity.shape != (dim,):
            raise ValueError(f"gravity must have shape ({dim},), got {self.gravity.shape}")

        for name, value in (("k_s", self.k_s), ("k_d", self.k_d), ("drag", self.drag)):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        for name, arr in (("x", self.x), ("v", self.v), ("inv_masses", self.inv_masses),
                          ("rest_lengths", self.rest_lengths), ("gravity", self.gravity)):
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} contains NaN or inf")

        # pinned particles cannot move
        self.v[self.inv_masses == 0] = 0.0


def compute_forces(sys: SpringSystem) -> np.ndarray:
    '''Calculate the total force on every particle.
Must zero the accumulator every call.  '''
    F = np.zeros_like(sys.x)
    masses = np.where(sys.inv_masses > 0, 
                      1.0 / np.where(sys.inv_masses > 0, sys.inv_masses, 1.0),
                      0)
    F += masses[:, None] * sys.gravity[None, :]
    F += -sys.drag * sys.v

    # Calculate each spring
    for s, (i, j) in enumerate(sys.springs):

        d = sys.x[j] - sys.x[i]
        L = np.linalg.norm(d)
        if L < 1e-12:
            continue
        n = d / L
        stretch = L - sys.rest_lengths[i]
        rel_v = np.dot(sys.v[j] - sys.v[i], n)
        f = sys.k_s * stretch + sys.k_d * rel_v
        f_vec = f * n
        F[i] += f_vec
        F[j] -= f_vec

    return F

