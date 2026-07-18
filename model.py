import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT(nn.Module):
    """Decoder-only language model"""
    # Now token + learned positional embeddings, summed, then a feed-forward layer
    # TODO: Attention
    # TODO: try fixed sinusoidal positional encoding instead of learned

    def __init__(self, vocab_size: int, block_size: int, d_model: int, ffn_multiplier: int):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.ffn_in = nn.Linear(d_model, d_model * ffn_multiplier)
        self.ffn_out = nn.Linear(d_model * ffn_multiplier, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = token_ids.shape
        assert seq_len <= self.block_size, (
            f"seq_len={seq_len} > block_size={self.block_size}: "
            "the context must fit the positional embedding window."
        )

        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        return self.ffn_out(F.relu(self.ffn_in(x)))


if __name__ == "__main__":
    from config.hyperparams import (
        BATCH_SIZE,
        BLOCK_SIZE,
        D_MODEL,
        DATASET_PATH,
        FFN_MULTIPLIER,
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
        ffn_multiplier=FFN_MULTIPLIER,
    )
    out = model(x)

    print(f"ids: {tuple(x.shape)} -> hidden states: {tuple(out.shape)}")
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")
