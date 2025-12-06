

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Multi-Head Self-Attention module.
        
        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Forward pass for multi-head self-attention.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor | None): Optional mask tensor.
        Returns:
            torch.Tensor: Output tensor after applying multi-head self-attention.
        """
        # x: (batch, seq_len, d_model)
        bsz, seq_len, _ = x.size()

        Q = self.W_q(x)  # (b, s, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # reshape to (b, h, s, d_k)
        def split_heads(t):
            return t.view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = map(split_heads, (Q, K, V))

        # scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (b, h, s, s)
        if mask is not None:
            # mask: (b, 1, 1, s) or broadcastable
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = attn @ V  # (b, h, s, d_k)

        # combine heads
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.W_o(context)  # (b, s, d_model)
        return out
