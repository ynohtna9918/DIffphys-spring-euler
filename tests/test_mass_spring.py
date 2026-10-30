@dataclass
class SpringSystem:
    x: np.ndarry
    v: np.ndarray
    inv_masses: np.ndarry
    springs: np.ndarray
    rest_lengths: np.ndarray
    k_x: float
    k_d: float
    gravity: np.ndarray
    drag: float

    