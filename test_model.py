import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, CausalSelfAttention, RotaryPositionalEmbedding, TransformerBlock


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
    # The same token twice, each time with a different past to look at, so the
    # rotation shifts the attention weights and the two outputs come out apart.
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    out = model(torch.tensor([[3, 7, 7]]))
    assert not torch.allclose(out[0, 1], out[0, 2])


def test_values_carry_no_position():
    # The flip side, and a property worth pinning down: only q and k are
    # rotated, values are not. A run of one repeated token therefore mixes
    # identical values in whatever proportion the weights ask for and returns
    # that same vector at every place. Position rides on the attention weights
    # alone — which is exactly what makes it relative. A learned position table
    # added to the embeddings would have pulled these two apart instead.
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    out = model(torch.full((1, 4), 7, dtype=torch.long))
    assert torch.allclose(out[0, 0], out[0, 3], atol=1e-6)


def test_same_token_at_the_same_position_is_identical_across_the_batch():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    out = model(torch.full((4, 8), 7, dtype=torch.long))
    # allclose, not equal: batched matmul may sum in a different order per row.
    assert torch.allclose(out[0], out[3], atol=1e-6)


def test_the_embedding_table_receives_gradients():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    model(torch.zeros(4, 8, dtype=torch.long)).sum().backward()
    assert model.token_embedding.weight.grad.abs().sum() > 0


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
    attention = model.block.attention
    for layer in (attention.query, attention.key, attention.value, attention.proj):
        assert layer.weight.grad.abs().sum() > 0


def test_attention_keeps_the_shape_of_its_input():
    attention = CausalSelfAttention(d_model=32, num_heads=4, block_size=8)
    x = torch.randn(4, 8, 32)
    assert attention(x, _causal_mask(8)).shape == x.shape


def test_heads_carve_up_the_model_dimension():
    attention = CausalSelfAttention(d_model=32, num_heads=4, block_size=8)
    assert attention.head_dim == 8

    weights = attention.weights(torch.randn(2, 5, 32), _causal_mask(5))
    assert weights.shape == (2, 4, 5, 5)


def test_head_count_does_not_change_the_parameter_count():
    # Heads are a reshape of the same projections, not extra ones.
    def parameters(num_heads: int) -> int:
        attention = CausalSelfAttention(d_model=32, num_heads=num_heads, block_size=8)
        return sum(p.numel() for p in attention.parameters())

    assert parameters(1) == parameters(8)


def test_d_model_must_split_evenly_across_the_heads():
    with pytest.raises(AssertionError):
        CausalSelfAttention(d_model=32, num_heads=5, block_size=8)


def test_the_heads_attend_differently():
    # Each head reads its own slice of q/k, so their weights must disagree —
    # a broadcast bug would copy one head's scores over all of them.
    attention = CausalSelfAttention(d_model=32, num_heads=4, block_size=8)
    weights = attention.weights(torch.randn(1, 5, 32), _causal_mask(5))
    assert not torch.allclose(weights[0, 0], weights[0, 1])


def test_every_head_stays_causal():
    attention = CausalSelfAttention(d_model=32, num_heads=4, block_size=8)
    weights = attention.weights(torch.randn(2, 6, 32), _causal_mask(6))
    assert (weights.masked_select(_causal_mask(6)) == 0).all()


def test_a_fully_masked_row_would_collapse_the_softmax():
    # Guards the mask convention itself: True means "cannot look", so masking a
    # whole row leaves softmax with nothing to normalise and yields NaN.
    attention = CausalSelfAttention(d_model=32, num_heads=4, block_size=8)
    mask = torch.ones(8, 8, dtype=torch.bool)
    assert attention(torch.randn(1, 8, 32), mask).isnan().any()


def test_attention_averages_the_values_it_can_see():
    # With all-zero q/k every visible score ties, so softmax averages uniformly:
    # position i returns the mean of v[0..i]. Checked before the output
    # projection by making proj an identity. Two heads, so the expected vector
    # only comes out whole if the split and the merge are each other's inverse.
    attention = CausalSelfAttention(d_model=4, num_heads=2, block_size=8)
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


def test_rope_head_dim_must_be_even():
    with pytest.raises(AssertionError):
        RotaryPositionalEmbedding(head_dim=7, block_size=16)


def test_rope_adds_no_learned_parameters():
    # The angles follow from the position instead of being trained: this is the
    # cost a learned position table carries and the rotation does not.
    rope = RotaryPositionalEmbedding(head_dim=8, block_size=16)
    assert list(rope.parameters()) == []


def test_rope_leaves_the_first_position_untouched():
    # Position 0 turns by a zero angle, so a window always starts unrotated.
    rope = RotaryPositionalEmbedding(head_dim=8, block_size=16)
    x = torch.randn(1, 1, 16, 8)
    assert torch.allclose(rope(x)[0, 0, 0], x[0, 0, 0], atol=1e-6)


def test_rope_preserves_the_length_of_a_vector():
    # A rotation turns a vector without stretching it, so position cannot shift
    # the scale of the dot products the way an additive encoding would.
    rope = RotaryPositionalEmbedding(head_dim=8, block_size=16)
    x = torch.randn(2, 4, 16, 8)
    assert torch.allclose(rope(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_scores_depend_only_on_the_distance_between_tokens():
    # The defining property. Hold q and k fixed and vary only where they sit:
    # the score for (i, j) must then equal the one for (i + 1, j + 1), i.e. the
    # score matrix is constant along its diagonals. Attention reads the gap
    # between two tokens, never their absolute places in the window.
    rope = RotaryPositionalEmbedding(head_dim=8, block_size=16)
    q = rope(torch.randn(8).expand(1, 1, 16, 8))
    k = rope(torch.randn(8).expand(1, 1, 16, 8))

    scores = q[0, 0] @ k[0, 0].T
    assert torch.allclose(scores[:-1, :-1], scores[1:, 1:], atol=1e-5)


def test_rope_scores_change_with_the_distance():
    # Guards the test above from passing on a rotation that does nothing at all:
    # constant diagonals are only meaningful if the diagonals differ.
    rope = RotaryPositionalEmbedding(head_dim=8, block_size=16)
    q = rope(torch.randn(8).expand(1, 1, 16, 8))
    k = rope(torch.randn(8).expand(1, 1, 16, 8))

    scores = q[0, 0] @ k[0, 0].T
    assert not torch.allclose(scores[8, 8], scores[8, 0])


def test_attention_reads_position_from_the_rotation():
    # Every token here is the very same vector, so without the rotation q and k
    # would be identical everywhere, every visible score would tie and softmax
    # would return a flat average. The rotation is the only thing left that can
    # tell the positions apart — this catches it not being wired into q and k.
    attention = CausalSelfAttention(d_model=32, num_heads=4, block_size=8)
    x = torch.randn(1, 1, 32).expand(1, 6, 32)

    last_row = attention.weights(x, _causal_mask(6))[0, 0, -1]
    assert not torch.allclose(last_row, torch.full((6,), 1 / 6), atol=1e-3)


def test_ffn_hidden_layer_is_widened_by_the_multiplier():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    assert model.block.ffn_in.out_features == 128
    assert model.block.ffn_out.in_features == 128


def test_relu_zeroes_the_hidden_layer():
    # A strongly negative bias pushes every hidden unit below zero, so ReLU
    # zeroes them and the ffn contributes nothing but its output bias. Attention
    # is silenced too, leaving the embeddings to carry the residual stream.
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    with torch.no_grad():
        model.block.ffn_in.bias.fill_(-1e4)
        model.block.attention.proj.weight.zero_()
        model.block.attention.proj.bias.zero_()

    ids = torch.zeros(1, 4, dtype=torch.long)
    expected = model.lm_head(model.norm_f(model.token_embedding(ids) + model.block.ffn_out.bias))
    assert torch.allclose(model(ids), expected, atol=1e-6)


def test_ffn_receives_gradients():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    model(torch.zeros(4, 8, dtype=torch.long)).sum().backward()
    assert model.block.ffn_in.weight.grad.abs().sum() > 0
    assert model.block.ffn_out.weight.grad.abs().sum() > 0


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
        for layer in (model.block.attention.proj, model.block.ffn_out):
            layer.weight.zero_()
            layer.bias.zero_()

    ids = torch.randint(50, (2, 6))
    expected = model.lm_head(model.norm_f(model.token_embedding(ids)))
    assert torch.allclose(model(ids), expected, atol=1e-6)


def test_block_keeps_the_shape_of_its_input():
    # What makes the block stackable: it hands back what it was given.
    block = TransformerBlock(d_model=32, num_heads=4, block_size=8, ffn_multiplier=4)
    x = torch.randn(4, 8, 32)
    assert block(x, _causal_mask(8)).shape == x.shape


def test_both_sublayers_write_into_the_same_stream():
    # Silencing attention alone must leave the feed-forward reading the stream
    # the embeddings put there, not a rewritten one: the second line of the
    # block adds to x rather than replacing it.
    block = TransformerBlock(d_model=32, num_heads=4, block_size=8, ffn_multiplier=4)
    with torch.no_grad():
        block.attention.proj.weight.zero_()
        block.attention.proj.bias.zero_()

    x = torch.randn(2, 6, 32)
    expected = x + block.ffn_out(F.relu(block.ffn_in(block.norm2(x))))
    assert torch.allclose(block(x, _causal_mask(6)), expected, atol=1e-6)


def test_the_norms_receive_gradients():
    model = GPT(vocab_size=50, block_size=8, d_model=32, num_heads=4, ffn_multiplier=4)
    model(torch.zeros(4, 8, dtype=torch.long)).sum().backward()
    for norm in (model.block.norm1, model.block.norm2, model.norm_f):
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
