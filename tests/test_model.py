import torch
from src.model import GPT


def make_small_model():
    return GPT(vocab_size=50, max_seq_len=20, num_layers=2, num_heads=2, d_model=16)


def test_forward_output_shape():
    model = make_small_model()
    x = torch.randint(0, 50, (2, 10))
    out = model(x)

    assert out.shape == (2, 10, 50)


def test_output_is_raw_logits_not_softmaxed():
    model = make_small_model()
    x = torch.randint(0, 50, (2, 10))
    out = model(x)

    # a real probability distribution sums to 1 along vocab dim;
    # raw logits should not
    sums = out.sum(dim=-1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


def test_handles_varying_seq_len():
    model = make_small_model()
    for seq_len in (1, 5, 20):
        x = torch.randint(0, 50, (2, seq_len))
        out = model(x)
        assert out.shape == (2, seq_len, 50)


def test_gradient_flows_through_full_model():
    model = make_small_model()
    x = torch.randint(0, 50, (2, 10))
    out = model(x)
    out.sum().backward()

    # spot-check a weight in the first and last block both received gradient
    first_grad = model.transformer_blocks[0].mha.q_proj.weight.grad
    last_grad = model.transformer_blocks[-1].mha.q_proj.weight.grad

    assert first_grad is not None and not torch.isnan(first_grad).any()
    assert last_grad is not None and not torch.isnan(last_grad).any()


def test_different_num_layers_still_works():
    for n in (1, 3):
        model = GPT(vocab_size=50, max_seq_len=20, num_layers=n, num_heads=2, d_model=16)
        x = torch.randint(0, 50, (2, 5))
        out = model(x)
        assert out.shape == (2, 5, 50)