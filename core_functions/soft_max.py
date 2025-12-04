import numpy as np
def soft_max(v: np.ndarray) -> np.ndarray:
    """Compute the softmax of vector v in a numerically stable way.
        i.e., subtract the maximum value from v, then divide by the sum of exponentials.
    Args:
        v (np.ndarray): Input vector."""
    
    c = np.max(v)
    exp_v = np.exp(v - c)
    sum_exp_v: int = np.sum(exp_v)
    return exp_v / sum_exp_v