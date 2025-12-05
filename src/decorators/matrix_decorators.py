import numpy as np
from typing import Callable as function

def check_can_mult_mats(f: function) -> function:
    """
    Decorator to check if two matrices can be multiplied.
    """
    def wrapper(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
        if mat_a.shape[1] != mat_b.shape[0]:
            raise ValueError("Incompatible matrix dimensions for multiplication.")
        return f(mat_a, mat_b)
    return wrapper