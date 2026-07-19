import pytest
import torch

from model import GPT, CausalSelfAttention


def _causal_mask(seq_len: int) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)


def test_output_shape():
    model = GPT(vocab_size=50, block_size=8, d_model=32, ffn_multiplier=4)
    out = model(torch.zeros(4, 8, dtype=torch.long))
    assert out.shape == (4, 8, 32)


def test_shorter_sequences_are_allowed():
    model = GPT(vocab_size=50, block_size=8, d_model=32, ffn_multiplier=4)
    out = model(torch.zeros(4, 3, dtype=torch.long))
    assert out.shape == (4, 3, 32)


def test_sequence_longer_than_block_size_is_rejected():
    model = GPT(vocab_size=50, block_size=8, d_model=32, ffn_multiplier=4)
    with pytest.raises(AssertionError):
        model(torch.zeros(4, 9, dtype=torch.long))


def test_position_changes_the_vector_of_the_same_token():
    model = GPT(vocab_size=50, block_size=8, d_model=32, ffn_multiplier=4)
    out = model(torch.full((1, 2), 7, dtype=torch.long))
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_same_token_at_the_same_position_is_identical_across_the_batch():
    model = GPT(vocab_size=50, block_size=8, d_model=32, ffn_multiplier=4)
    out = model(torch.full((4, 8), 7, dtype=torch.long))
    # allclose, not equal: batched matmul may sum in a different order per row.
    assert torch.allclose(out[0], out[3], atol=1e-6)


def test_both_embedding_tables_receive_gradients():
    model = GPT(vocab_size=50, block_size=8, d_model=32, ffn_multiplier=4)
    model(torch.zeros(4, 8, dtype=torch.long)).sum().backward()
    assert model.token_embedding.weight.grad.abs().sum() > 0
    assert model.position_embedding.weight.grad.abs().sum() > 0


def test_a_token_does_not_see_the_future():
    model = GPT(vocab_size=50, block_size=8, d_model=32, ffn_multiplier=4)
    ids = torch.randint(50, (1, 8))
    out = model(ids)

    # Rewriting the last token must leave every earlier position untouched.
    ids[0, -1] = (ids[0, -1] + 1) % 50
    assert torch.allclose(model(ids)[0, :-1], out[0, :-1], atol=1e-6)


def test_the_first_token_depends_on_nothing_but_itself():
    # Position 0 has no past to attend to, so the rest of the window cannot
    # reach it — this also catches a mask that hides the diagonal itself.
    model = GPT(vocab_size=50, block_size=8, d_model=32, ffn_multiplier=4)
    ids = torch.randint(50, (1, 8))
    out = model(ids)

    assert not out.isnan().any()
    assert torch.allclose(model(ids[:, :1])[0, 0], out[0, 0], atol=1e-6)


def test_attention_receives_gradients():
    model = GPT(vocab_size=50, block_size=8, d_model=32, ffn_multiplier=4)
    model(torch.zeros(4, 8, dtype=torch.long)).sum().backward()
    attention = model.attention
    for layer in (attention.query, attention.key, attention.value, attention.proj):
        assert layer.weight.grad.abs().sum() > 0


def test_attention_keeps_the_shape_of_its_input():
    attention = CausalSelfAttention(d_model=32)
    x = torch.randn(4, 8, 32)
    assert attention(x, _causal_mask(8)).shape == x.shape


def test_a_fully_masked_row_would_collapse_the_softmax():
    # Guards the mask convention itself: True means "cannot look", so masking a
    # whole row leaves softmax with nothing to normalise and yields NaN.
    attention = CausalSelfAttention(d_model=32)
    mask = torch.ones(8, 8, dtype=torch.bool)
    assert attention(torch.randn(1, 8, 32), mask).isnan().any()


def test_attention_averages_the_values_it_can_see():
    # With all-zero q/k every visible score ties, so softmax averages uniformly:
    # position i returns the mean of v[0..i]. Checked before the output
    # projection by making proj an identity.
    attention = CausalSelfAttention(d_model=4)
    with torch.no_grad():
        for layer in (attention.query, attention.key):
            layer.weight.zero_()
            layer.bias.zero_()
        attention.proj.weight.copy_(torch.eye(4))
        attention.proj.bias.zero_()

    x = torch.randn(1, 5, 4)
    out = attention(x, _causal_mask(5))

    v = attention.value(x)
    expected = torch.stack([v[0, : i + 1].mean(dim=0) for i in range(5)])
    assert torch.allclose(out[0], expected, atol=1e-6)


def test_ffn_hidden_layer_is_widened_by_the_multiplier():
    model = GPT(vocab_size=50, block_size=8, d_model=32, ffn_multiplier=4)
    assert model.ffn_in.out_features == 128
    assert model.ffn_out.in_features == 128


def test_relu_zeroes_the_hidden_layer():
    # A strongly negative bias pushes every hidden unit below zero, so ReLU
    # zeroes them and nothing but the output bias survives.
    model = GPT(vocab_size=50, block_size=8, d_model=32, ffn_multiplier=4)
    with torch.no_grad():
        model.ffn_in.bias.fill_(-1e4)
    out = model(torch.zeros(1, 4, dtype=torch.long))
    assert torch.allclose(out[0, 0], model.ffn_out.bias, atol=1e-6)


def test_ffn_receives_gradients():
    model = GPT(vocab_size=50, block_size=8, d_model=32, ffn_multiplier=4)
    model(torch.zeros(4, 8, dtype=torch.long)).sum().backward()
    assert model.ffn_in.weight.grad.abs().sum() > 0
    assert model.ffn_out.weight.grad.abs().sum() > 0
