import torch
import torch.nn as nn

from .layers import TransformerBlock


class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers, num_heads, d_model):
        super().__init__()
        self.tkn_embedding = nn.Embedding(vocab_size, d_model)
        self.pstnl_embedding = nn.Embedding(max_seq_len, d_model)
        self.transformer_blocks = nn.ModuleList((TransformerBlock(d_model, num_heads) for _ in range(num_layers)))
        self.layernorm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        ones_matrix = torch.ones(seq_len, seq_len)
        casual_mask = torch.tril(ones_matrix)  # created in order to mask some of the word embeddings to aid in learning

        token_embedding = self.tkn_embedding(x)
        indices = torch.arange(seq_len)
        positional_embedding = self.pstnl_embedding(indices)
        word_embeddings = token_embedding + positional_embedding

        out = word_embeddings

        for block in self.transformer_blocks:
            out = block(out, casual_mask)

        out = self.layernorm(out)
        out = self.linear(out)

        return out


# TESTING GPT VALIDITY
# if __name__ == '__main__':
#     model = GPT(
#         vocab_size=50,
#         max_seq_len=20,
#         num_layers=2,
#         num_heads=2,
#         d_model=16
#     )
#
#     x = torch.randint(0, 50, (2, 10))  # (batch=2, seq_len=10), random token IDs in [0, vocab_size)
#     out = model(x)
#
#     print(out.shape)  # should output torch.Size([2, 10, 50]) - (batch, seq_len, vocab_size)
