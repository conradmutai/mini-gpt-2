import torch
from src.layers import MultiHeadSelfAttention, MultilayerPerceptron, TransformerBlock


def test_attention_output_shape():
    d_model, num_heads = 32, 4
    batch, seq_len = 2, 10

    mha = MultiHeadSelfAttention(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model)
    out = mha(x)

    assert out.shape == (batch, seq_len, d_model)


def test_attention_with_causal_mask():
    d_model, num_heads = 32, 4
    batch, seq_len = 2, 10

    mha = MultiHeadSelfAttention(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    out = mha(x, mask)

    assert out.shape == (batch, seq_len, d_model)


def test_head_dim_divides_evenly():
    mha = MultiHeadSelfAttention(d_model=64, num_heads=8)
    assert mha.head_dim == 8
    assert mha.head_dim * mha.num_heads == mha.d_model


def test_mlp_output_shape():
    d_model = 32
    batch, seq_len = 2, 10

    mlp = MultilayerPerceptron(d_model)
    x = torch.randn(batch, seq_len, d_model)
    out = mlp(x)

    assert out.shape == (batch, seq_len, d_model)


def test_transformer_block_preserves_shape():
    d_model, num_heads = 32, 4
    batch, seq_len = 2, 10

    block = TransformerBlock(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    out = block(x, mask)

    assert out.shape == x.shape


def test_transformer_block_gradient_flows():
    d_model, num_heads = 16, 2
    batch, seq_len = 2, 5

    block = TransformerBlock(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)
    out = block(x)
    out.sum().backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

