import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, CausalSelfAttention


def _causal_mask(seq_len: int) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)


def test_output_shape():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    out = model(torch.zeros(4, 8, dtype=torch.long))
    assert out.shape == (4, 8, 50)


def test_shorter_sequences_are_allowed():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    out = model(torch.zeros(4, 3, dtype=torch.long))
    assert out.shape == (4, 3, 50)


def test_sequence_longer_than_block_size_is_rejected():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    with pytest.raises(AssertionError):
        model(torch.zeros(4, 9, dtype=torch.long))


def test_position_changes_the_vector_of_the_same_token():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    out = model(torch.full((1, 2), 7, dtype=torch.long))
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_same_token_at_the_same_position_is_identical_across_the_batch():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    out = model(torch.full((4, 8), 7, dtype=torch.long))
    # allclose, not equal: batched matmul may sum in a different order per row.
    assert torch.allclose(out[0], out[3], atol=1e-6)


def test_both_embedding_tables_receive_gradients():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    model(torch.zeros(4, 8, dtype=torch.long)).sum().backward()
    assert model.token_embedding.weight.grad.abs().sum() > 0
    assert model.position_embedding.weight.grad.abs().sum() > 0


def test_a_token_does_not_see_the_future():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    ids = torch.randint(50, (1, 8))
    out = model(ids)

    # Rewriting the last token must leave every earlier position untouched.
    ids[0, -1] = (ids[0, -1] + 1) % 50
    assert torch.allclose(model(ids)[0, :-1], out[0, :-1], atol=1e-6)


def test_the_first_token_depends_on_nothing_but_itself():
    # Position 0 has no past to attend to, so the rest of the window cannot
    # reach it — this also catches a mask that hides the diagonal itself.
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    ids = torch.randint(50, (1, 8))
    out = model(ids)

    assert not out.isnan().any()
    assert torch.allclose(model(ids[:, :1])[0, 0], out[0, 0], atol=1e-6)


def test_attention_receives_gradients():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    model(torch.zeros(4, 8, dtype=torch.long)).sum().backward()
    attention = model.attention
    for layer in (attention.query, attention.key, attention.value, attention.proj):
        assert layer.weight.grad.abs().sum() > 0


def test_attention_keeps_the_shape_of_its_input():
    attention = CausalSelfAttention(d_model=32, num_heads=4)
    x = torch.randn(4, 8, 32)
    assert attention(x, _causal_mask(8)).shape == x.shape


def test_heads_carve_up_the_model_dimension():
    attention = CausalSelfAttention(d_model=32, num_heads=4)
    assert attention.head_dim == 8

    weights = attention.weights(torch.randn(2, 5, 32), _causal_mask(5))
    assert weights.shape == (2, 4, 5, 5)


def test_head_count_does_not_change_the_parameter_count():
    # Heads are a reshape of the same projections, not extra ones.
    def parameters(num_heads: int) -> int:
        attention = CausalSelfAttention(d_model=32, num_heads=num_heads)
        return sum(p.numel() for p in attention.parameters())

    assert parameters(1) == parameters(8)


def test_d_model_must_split_evenly_across_the_heads():
    with pytest.raises(AssertionError):
        CausalSelfAttention(d_model=32, num_heads=5)


def test_the_heads_attend_differently():
    # Each head reads its own slice of q/k, so their weights must disagree —
    # a broadcast bug would copy one head's scores over all of them.
    attention = CausalSelfAttention(d_model=32, num_heads=4)
    weights = attention.weights(torch.randn(1, 5, 32), _causal_mask(5))
    assert not torch.allclose(weights[0, 0], weights[0, 1])


def test_every_head_stays_causal():
    attention = CausalSelfAttention(d_model=32, num_heads=4)
    weights = attention.weights(torch.randn(2, 6, 32), _causal_mask(6))
    assert (weights.masked_select(_causal_mask(6)) == 0).all()


def test_a_fully_masked_row_would_collapse_the_softmax():
    # Guards the mask convention itself: True means "cannot look", so masking a
    # whole row leaves softmax with nothing to normalise and yields NaN.
    attention = CausalSelfAttention(d_model=32, num_heads=4)
    mask = torch.ones(8, 8, dtype=torch.bool)
    assert attention(torch.randn(1, 8, 32), mask).isnan().any()


def test_attention_averages_the_values_it_can_see():
    # With all-zero q/k every visible score ties, so softmax averages uniformly:
    # position i returns the mean of v[0..i]. Checked before the output
    # projection by making proj an identity. Two heads, so the expected vector
    # only comes out whole if the split and the merge are each other's inverse.
    attention = CausalSelfAttention(d_model=4, num_heads=2)
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
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    assert model.ffn_in.out_features == 128
    assert model.ffn_out.in_features == 128


def test_relu_zeroes_the_hidden_layer():
    # A strongly negative bias pushes every hidden unit below zero, so ReLU
    # zeroes them and the ffn contributes nothing but its output bias. Attention
    # is silenced too, leaving the embeddings to carry the residual stream.
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    with torch.no_grad():
        model.ffn_in.bias.fill_(-1e4)
        model.attention.proj.weight.zero_()
        model.attention.proj.bias.zero_()

    ids = torch.zeros(1, 4, dtype=torch.long)
    expected = model.lm_head(model.norm_f(model.embed(ids) + model.ffn_out.bias))
    assert torch.allclose(model(ids), expected, atol=1e-6)


def test_ffn_receives_gradients():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    model(torch.zeros(4, 8, dtype=torch.long)).sum().backward()
    assert model.ffn_in.weight.grad.abs().sum() > 0
    assert model.ffn_out.weight.grad.abs().sum() > 0


def test_layernorm_matches_the_formula():
    # nn.LayerNorm does the work, but the formula is what the step is about:
    # centre each token vector over its own features, scale to unit variance,
    # then apply the learned gain and shift. Randomised so the affine part is
    # actually exercised instead of the identity it starts as.
    norm = nn.LayerNorm(32)
    with torch.no_grad():
        norm.weight.normal_()
        norm.bias.normal_()

    x = torch.randn(4, 8, 32)
    centred = x - x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    expected = centred / torch.sqrt(variance + norm.eps) * norm.weight + norm.bias

    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_silent_sublayers_leave_the_residual_stream_untouched():
    # Zeroing both sublayer outputs turns each line of the block into x = x + 0,
    # so the embeddings must reach the head unchanged. Catches a dropped
    # residual, and post-norm too: that would rescale the stream on the way.
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    with torch.no_grad():
        for layer in (model.attention.proj, model.ffn_out):
            layer.weight.zero_()
            layer.bias.zero_()

    ids = torch.randint(50, (2, 6))
    expected = model.lm_head(model.norm_f(model.embed(ids)))
    assert torch.allclose(model(ids), expected, atol=1e-6)


def test_the_norms_receive_gradients():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    model(torch.zeros(4, 8, dtype=torch.long)).sum().backward()
    for norm in (model.norm1, model.norm2, model.norm_f):
        assert norm.weight.grad.abs().sum() > 0
        assert norm.bias.grad.abs().sum() > 0


def test_loss_of_an_untrained_model_is_the_uniform_baseline():
    # An untrained head has no preference, so every token of the vocabulary is
    # about equally likely and cross-entropy sits near ln(vocab_size). The final
    # norm keeps the head's input in scale, but the learned gain still spreads
    # the logits a little, so the loss sits slightly above the baseline.
    vocab_size = 200
    model = GPT(vocab_size=vocab_size, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    ids = torch.randint(vocab_size, (16, 8))

    logits = model(ids)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), ids.reshape(-1))

    assert loss.item() == pytest.approx(math.log(vocab_size), abs=0.35)


def test_loss_drops_when_the_head_predicts_the_target():
    # Boosting the logit of the correct next token must lower the loss.
    vocab_size = 50
    model = GPT(vocab_size=vocab_size, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    ids = torch.randint(vocab_size, (4, 8))
    targets = ids.reshape(-1)

    logits = model(ids).reshape(-1, vocab_size)
    before = F.cross_entropy(logits, targets)

    logits = logits.clone()
    logits[torch.arange(targets.numel()), targets] += 10.0
    assert F.cross_entropy(logits, targets) < before


def test_the_head_receives_gradients():
    vocab_size = 50
    model = GPT(vocab_size=vocab_size, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    ids = torch.randint(vocab_size, (4, 8))

    logits = model(ids)
    F.cross_entropy(logits.reshape(-1, vocab_size), ids.reshape(-1)).backward()

    assert model.lm_head.weight.grad.abs().sum() > 0
    assert model.token_embedding.weight.grad.abs().sum() > 0


def test_generate_extends_the_prompt():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    prompt = torch.randint(50, (1, 3))

    out = model.generate(prompt, max_new_tokens=5)

    assert out.shape == (1, 8)
    assert torch.equal(out[:, :3], prompt)


def test_generate_is_deterministic():
    # Greedy decoding takes the argmax, so nothing is left to chance: the same
    # prompt through the same weights must give the very same continuation.
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    prompt = torch.randint(50, (1, 3))

    assert torch.equal(
        model.generate(prompt, max_new_tokens=5), model.generate(prompt, max_new_tokens=5)
    )


def test_generate_slides_the_window_past_block_size():
    # The prompt alone already fills the window, and every new token pushes it
    # further; without the sliding window forward would hit its block_size assert.
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    prompt = torch.randint(50, (1, 8))

    assert model.generate(prompt, max_new_tokens=4).shape == (1, 12)


def test_generate_leaves_no_gradients_behind():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)

    out = model.generate(torch.randint(50, (1, 3)), max_new_tokens=5)

    assert not out.requires_grad
    assert model.lm_head.weight.grad is None
