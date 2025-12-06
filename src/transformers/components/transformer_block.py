from multi_head_self_attention import MultiHeadSelfAttention
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """A single Transformer block consisting of multi-head self-attention and feedforward network.
     Args:
        d_model (int): Dimension of the model.
        num_heads (int): Number of attention heads.
        d_ff (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
    Returns:
        torch.Tensor: Output tensor after applying the Transformer block.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Forward pass for the Transformer block.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor | None): Optional mask tensor.
        Returns:
            torch.Tensor: Output tensor after applying the Transformer block.
        """
        # Self-attention + residual + norm
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feedforward + residual + norm
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
