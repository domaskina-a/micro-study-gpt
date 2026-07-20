import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """True strictly above the diagonal, i.e. the future a token must not see."""
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model={d_model} must split evenly across num_heads={num_heads}."
        )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # One projection per role, shared by every head: the heads are carved out
        # of its output, so head count does not change the parameter count.
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)"""
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def weights(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Attention weights per head, (batch, num_heads, seq_len, seq_len)."""
        q, k = self.split_heads(self.query(x)), self.split_heads(self.key(x))

        # Scaling by sqrt(head_dim) keeps the dot products from growing with
        # dimension, which would otherwise saturate the softmax.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask, float("-inf"))

        return F.softmax(scores, dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """mask: (seq_len, seq_len), True where a token must not look."""
        batch, seq_len, _ = x.shape
        heads = self.weights(x, mask) @ self.split_heads(self.value(x))

        # Concatenate the heads back into one vector per token.
        merged = heads.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.proj(merged)


class GPT(nn.Module):
    """Decoder-only language model"""
    # Now token + learned positional embeddings, summed, then a pre-norm block
    # (attention and feed-forward, each normalised on input and added back to
    # the residual stream), then a final norm and logits
    # TODO: try fixed sinusoidal positional encoding instead of learned

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int,
        num_heads: int,
        ffn_multiplier: int,
    ):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, d_model * ffn_multiplier)
        self.ffn_out = nn.Linear(d_model * ffn_multiplier, d_model)
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = token_ids.shape
        assert seq_len <= self.block_size, (
            f"seq_len={seq_len} > block_size={self.block_size}: "
            "the context must fit the positional embedding window."
        )

        positions = torch.arange(seq_len, device=token_ids.device)
        return self.token_embedding(token_ids) + self.position_embedding(positions)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)
        mask = causal_mask(x.shape[1], x.device)

        normed = self.norm1(x)
        attention_output = self.attention(normed, mask)
        x = x + attention_output

        ffn_output = self.ffn_out(F.relu(self.ffn_in(self.norm2(x))))
        x = x + ffn_output

        # Returns (batch, seq_len, vocab_size) logits
        return self.lm_head(self.norm_f(x))

    def generate(self, token_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy decoding: append the most likely next token, max_new_tokens times.

        Takes and returns ids, (batch, seq_len) -> (batch, seq_len + max_new_tokens);
        encoding and decoding stay outside the model.
        """
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Sliding window: the model only ever sees the last block_size
                # tokens, which is the window its positional embeddings cover.
                logits = self(token_ids[:, -self.block_size :])
                next_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
                token_ids = torch.cat([token_ids, next_ids], dim=1)

        return token_ids


if __name__ == "__main__":
    from config.hyperparams import (
        BATCH_SIZE,
        BLOCK_SIZE,
        D_MODEL,
        DATASET_PATH,
        FFN_MULTIPLIER,
        NUM_HEADS,
        SEED,
    )
    from utils.data_utils import get_batch, load_corpus
    from utils.seed import set_seed

    set_seed(SEED)

    data, _, itos = load_corpus(DATASET_PATH)
    x, _ = get_batch(data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)

    model = GPT(
        vocab_size=len(itos),
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        ffn_multiplier=FFN_MULTIPLIER,
    )
    logits = model(x)

    print(f"ids: {tuple(x.shape)} -> logits: {tuple(logits.shape)}")
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")

    # Attention table of the first sequence, first head: rows ask, columns answer.
    # Untrained weights, so the point is the causal shape, not the numbers.
    with torch.no_grad():
        embedded = model.embed(x[:1])
        weights = model.attention.weights(embedded, causal_mask(BLOCK_SIZE))[0, 0]

    tokens = [itos[i] for i in x[0].tolist()]
    width = 7
    print("\nattention weights of head 0 (row attends to column):")
    print(" " * width + "".join(f"{t[:width - 1]:>{width}}" for t in tokens))
    for i, token in enumerate(tokens):
        row = "".join(
            f"{weights[i, j]:>{width}.2f}" if j <= i else f"{'·':>{width}}"
            for j in range(len(tokens))
        )
        print(f"{token[:width - 1]:>{width}}" + row)
