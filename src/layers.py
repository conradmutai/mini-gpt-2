import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.omega_c = nn.Linear(d_model, d_model)

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        mh_sa_array = []

        for h in range(self.num_heads):
            q_slice = Q[:, :, h, :]  # (batch, seq_len, head_dim)
            k_slice = K[:, :, h, :]
            v_slice = V[:, :, h, :]

            dot_product = q_slice @ k_slice.transpose(1, 2)  # (batch, seq_len, seq_len)
            dot_product_div = dot_product / (self.head_dim ** 0.5)

            if mask is not None:
                dot_product_div = torch.where(mask == 0, torch.tensor(-1e9), dot_product_div)

            activations = torch.softmax(dot_product_div, dim=-1)
            sa = activations @ v_slice  # (batch, seq_len, head_dim)

            mh_sa_array.append(sa)

        mh_sa_concat = torch.cat(mh_sa_array, dim=-1)
        mh_sa = self.omega_c(mh_sa_concat)

        return mh_sa


class MultilayerPerceptron(nn.Module):
    def __init__(self, d_model):
        super(MultilayerPerceptron, self).__init__()
        dff = 4 * d_model
        self.fc1 = nn.Linear(d_model, dff)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(dff, d_model)

    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.fc2(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.mlp = MultilayerPerceptron(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x_out = x + self.mha(x, mask)
        x_out = self.layernorm1(x_out)
        x_out = x_out + self.mlp(x_out)
        x_out = self.layernorm2(x_out)

        return x_out


# TESTING BLOCK FOR TRANSFORMER
# if __name__ == '__main__':
#     block = TransformerBlock(d_model=16, num_heads=2)
#     x = torch.randn(2, 5, 16)  # (batch, seq_len, d_model)
#     out = block(x)
#     print(out.shape)  # should be (2, 5, 16) — identical to input
