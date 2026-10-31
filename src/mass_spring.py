from dataclasses import dataclass
import dataclasses
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

def _masses(sys: SpringSystem) -> np.ndarray:
    masses = np.where(sys.inv_masses > 0, 
                      1.0 / np.where(sys.inv_masses > 0, sys.inv_masses, 1.0),
                      0)
    return masses

def _spring_geometry(sys: SpringSystem):
    i, j = sys.springs[:, 0], sys.springs[:, 1]
    d = sys.x[j] - sys.x[i]
    L = np.linalg.norm(d, axis = 1)
    return d, L

def compute_forces(sys: SpringSystem) -> np.ndarray:
    '''Calculate the total force on every particle.
Must zero the accumulator every call.  '''
    F = np.zeros_like(sys.x)
    masses = _masses(sys)
    F += masses[:, None] * sys.gravity[None, :]
    F += -sys.drag * sys.v

    d_all, L_all = _spring_geometry(sys)
    for s, (i, j) in enumerate(sys.springs):
        d, L = d_all[s], L_all[s]
        if L < 1e-12:
            continue
        n = d / L
        stretch = L - sys.rest_lengths[s]
        rel_v = np.dot(sys.v[j] - sys.v[i], n)
        f = sys.k_s * stretch + sys.k_d * rel_v
        f_vec = f * n
        F[i] += f_vec
        F[j] -= f_vec

    return F

def derivatives(sys: SpringSystem):
    F = compute_forces(sys)
    a = F * sys.inv_masses[:, None]
    return sys.v.copy(), a

def total_energy(sys: SpringSystem):
    masses = _masses(sys)
    KE = 0.5 * np.sum(np.sum(sys.v ** 2, axis = 1) * masses)
    _, L = _spring_geometry(sys)
    PE_spring = np.sum((L - sys.rest_lengths) ** 2) * 0.5 * sys.k_s
    PE_gravity = -np.sum((sys.x @ sys.gravity) * masses)
    return float(KE + PE_spring + PE_gravity)


def step_explicit_euler(sys, dt):
    _, a = derivatives(sys)
    sys.x = sys.x + sys.v * dt
    sys.v = sys.v + a * dt

def step_semi_implicit_euler(sys, dt):
    _, a = derivatives(sys)
    sys.v = sys.v + a * dt  
    sys.x = sys.x + sys.v * dt

def step_midpoint(sys, dt):
    _, a = derivatives(sys)
    x_mid = sys.x + sys.v * (dt / 2)
    v_mid = sys.v + a * (dt / 2)
    sys.x = sys.x + v_mid * dt
    mid = dataclasses.replace(sys, x = x_mid, v = v_mid)
    _, a_mid = derivatives(mid)
    sys.v = sys.v + a_mid * dt

def simulate(sys, dt, steps, stepper):
    xs = [sys.x.copy()]
    vs = [sys.v.copy()]
    energies = [total_energy(sys)]
    for _ in range(steps):
        stepper(sys, dt)
        xs.append(sys.x.copy())
        vs.append(sys.v.copy())
        energies.append(total_energy(sys))
    return np.array(xs), np.array(vs), np.array(energies)

def make_single_oscillator(k_s=100.0, m=1.0, rest=1.0, dim=3,
                           k_d=0.0, drag=0.0, gravity=None) -> SpringSystem:
    x = np.zeros((2, dim))
    x[1, 0] = rest
    if gravity is None:
        gravity = np.zeros(dim)
    return SpringSystem(x=x, v=np.zeros((2, dim)), inv_masses=[0.0, 1.0 / m],
                        springs=[[0, 1]], rest_lengths=[rest],
                        k_s=k_s, k_d=k_d, gravity=gravity, drag=drag)

def make_chain(n=5, spacing=0.3, k_s=100.0, m=1.0,
               k_d=0.0, drag=0.0, gravity=None, dim=3) -> SpringSystem:
    x = np.zeros((n, dim))
    x[:, 1] = -spacing * np.arange(n)
    v = np.zeros((n, dim))
    inv_masses = np.full(n, 1.0 / m)
    inv_masses[0] = 0.0
    springs = [[k, k + 1] for k in range(n - 1)]
    rest_lengths = np.full(n - 1, spacing)
    if gravity is None:
        gravity = np.zeros(dim)
    
    return SpringSystem(x=x, v=v, inv_masses=inv_masses, springs=springs, rest_lengths=rest_lengths,
                        k_s=k_s, k_d=k_d, gravity=gravity, drag=drag)

def measure_period(u: np.ndarray, dt: float) -> float:
    k = np.where((u[:-1] < 0) & (u[1:] >= 0))[0]
    if len(k) < 2:
        raise ValueError(f"need at least 2 upward zero crossings, found {len(k)}")
    t = (k + u[k] / (u[k] - u[k + 1])) * dt
    return float(np.mean(np.diff(t)))

def compute_forces_vec(sys: SpringSystem) -> np.ndarray:
    F = np.zeros_like(sys.x)
    masses = _masses(sys)
    F += masses[:, None] * sys.gravity[None, :]
    F += -sys.drag * sys.v

    i, j = sys.springs[:, 0], sys.springs[:, 1]
    d, L = _spring_geometry(sys)
    ok = L >= 1e-12
    n = np.zeros_like(d)
    n[ok] = d[ok] / L[ok, None]
    stretch = L - sys.rest_lengths
    rel_v = np.sum((sys.v[j] - sys.v[i]) * n, axis = 1)
    f = sys.k_s * stretch + sys.k_d * rel_v
    f_vec = f[:, None] * n
    np.add.at(F, i, f_vec)
    np.add.at(F, j, -f_vec)
    return F
    
    