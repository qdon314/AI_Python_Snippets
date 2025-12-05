import numpy as np
from decorators.matrix_decorators import check_can_mult_mats

@check_can_mult_mats
def matrix_product(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Compute the product of two matrices.
    Args:
        mat_a (np.ndarray): First input matrix.
        mat_b (np.ndarray): Second input matrix.
    Returns:
        np.ndarray: The product of mat_a and mat_b.
    """
    
    # Initialize the result matrix with zeros
    result = np.zeros((mat_a.shape[0], mat_b.shape[1]))
    
    
    for i in range(mat_a.shape[0]):
        for j in range(mat_b.shape[1]):
            for k in range(mat_a.shape[1]):
                result[i, j] += mat_a[i, k] * mat_b[k, j]
    return result