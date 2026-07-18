import pytest
import torch

from model import GPT


def test_output_shape():
    model = GPT(vocab_size=50, block_size=8, d_model=32)
    out = model(torch.zeros(4, 8, dtype=torch.long))
    assert out.shape == (4, 8, 32)


def test_shorter_sequences_are_allowed():
    model = GPT(vocab_size=50, block_size=8, d_model=32)
    out = model(torch.zeros(4, 3, dtype=torch.long))
    assert out.shape == (4, 3, 32)


def test_sequence_longer_than_block_size_is_rejected():
    model = GPT(vocab_size=50, block_size=8, d_model=32)
    with pytest.raises(AssertionError):
        model(torch.zeros(4, 9, dtype=torch.long))


def test_position_changes_the_vector_of_the_same_token():
    model = GPT(vocab_size=50, block_size=8, d_model=32)
    out = model(torch.full((1, 2), 7, dtype=torch.long))
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_same_token_at_the_same_position_is_identical_across_the_batch():
    model = GPT(vocab_size=50, block_size=8, d_model=32)
    out = model(torch.full((4, 8), 7, dtype=torch.long))
    assert torch.equal(out[0], out[3])


def test_both_embedding_tables_receive_gradients():
    model = GPT(vocab_size=50, block_size=8, d_model=32)
    model(torch.zeros(4, 8, dtype=torch.long)).sum().backward()
    assert model.token_embedding.weight.grad.abs().sum() > 0
    assert model.position_embedding.weight.grad.abs().sum() > 0
