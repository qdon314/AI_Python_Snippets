import numpy as np
import math

def soft_max(v: np.ndarray) -> np.ndarray:
    """Compute the softmax of vector v in a numerically stable way.
        i.e., subtract the maximum value from v, then divide by the sum of exponentials.
    Args:
        v (np.ndarray): Input vector."""
    
    c = np.max(v)
    exp_v = np.exp(v - c)
    sum_exp_v: int = np.sum(exp_v)
    return exp_v / sum_exp_v

def soft_max_naive(v: list) -> list:
    """Compute the softmax of vector v in a naive way.
    Args:
        v (np.ndarray): Input vector."""
    
    exp_v = [x**math.e for x in v]
    sum_exp_v: int = sum(exp_v)
    return [x / sum_exp_v for x in exp_v]