import torch
import torch.nn as nn

from layers import TransformerBlock


class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers, num_heads, d_model):
        super().__init__()
        self.tkn_embedding = nn.Embedding(vocab_size, d_model)
        self.pstnl_embedding = nn.Embedding(max_seq_len, d_model)
        self.transformer_blocks = nn.ModuleList((TransformerBlock(d_model, num_heads) for _ in range(num_layers)))
        self.layernorm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask):
        batch_size, seq_len = x.shape

        token_embedding = self.tkn_embedding(x)
        indices = torch.arange(seq_len)
        positional_embedding = self.pstnl_embedding(indices)
        word_embeddings = token_embedding + positional_embedding

        out = word_embeddings

        for block in self.transformer_blocks:
            out = block(out, mask)

        out = self.layernorm(out)
        out = self.linear(out)

        return out
