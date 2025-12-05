import math
from matrix_product import matrix_product
from soft_max import soft_max

# Self-attention function
def scaled_dot_product_attention(q, k, v, mask=None):
    """
        Calculate the attention weights.
    """
    matmul_qk = matrix_product(q, k.transpose())
    scaled_attention_logits = matmul_qk / math.sqrt(k.shape[-1])

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = soft_max(scaled_attention_logits)
    output = matrix_product(attention_weights, v)
    return output, attention_weights

