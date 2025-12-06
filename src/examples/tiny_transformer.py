
import torch
from transformers.minimal_encoder_transformer import TinyTransformerLM

if __name__ == "__main__":
    vocab_size = 10000
    model = TinyTransformerLM(vocab_size)
    x = torch.randint(0, vocab_size, (2, 16))  # (batch=2, seq_len=16)
    # causal mask example: allow attending to current and previous positions only
    seq_len = x.size(1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    logits = model(x, mask=causal_mask)
    print(logits.shape)  # (2, 16, 10000)
