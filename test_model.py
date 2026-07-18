import pytest
import torch

from model import GPT


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
