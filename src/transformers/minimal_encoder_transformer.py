import math
import torch
import torch.nn as nn
from components.multi_head_self_attention import MultiHeadSelfAttention
from components.positional_encoding import PositionalEncoding
from components.transformer_block import TransformerBlock

# ---- Tiny Transformer LM -----------------------------------------------------


class TinyTransformerLM(nn.Module):
    """A minimal Transformer-based language model.
    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the model.
        num_heads (int): Number of attention heads.
        d_ff (int): Dimension of the feedforward network.
        num_layers (int): Number of Transformer layers.
        max_len (int): Maximum length of the input sequences.
        dropout (float): Dropout rate.
    Returns:
        torch.Tensor: Logits over the vocabulary for each position in the input sequence."""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 512,
        num_layers: int = 2,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Forward pass for the Tiny Transformer LM.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing token ids.
            mask (torch.Tensor | None): Optional mask tensor.
        Returns:
            torch.Tensor: Logits over the vocabulary for each position in the input sequence.
        """
        # x: (batch, seq_len) token ids
        emb = self.tok_emb(x) * math.sqrt(self.tok_emb.embedding_dim)
        h = self.dropout(self.pos_enc(emb))
        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        logits = self.head(h)  # (batch, seq_len, vocab_size)
        return logits

