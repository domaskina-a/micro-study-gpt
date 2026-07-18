import torch
import torch.nn as nn


class GPT(nn.Module):
    """Decoder-only language model"""
    # Now just token embeddings + learned positional embeddings, summed
    # TODO: Attention, FFN
    # TODO: try fixed sinusoidal positional encoding instead of learned

    def __init__(self, vocab_size: int, block_size: int, d_model: int):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = token_ids.shape
        assert seq_len <= self.block_size, (
            f"seq_len={seq_len} > block_size={self.block_size}: "
            "the context must fit the positional embedding window."
        )

        positions = torch.arange(seq_len, device=token_ids.device)
        return self.token_embedding(token_ids) + self.position_embedding(positions)


if __name__ == "__main__":
    from config.hyperparams import (
        BATCH_SIZE,
        BLOCK_SIZE,
        D_MODEL,
        DATASET_PATH,
        SEED,
    )
    from utils.data_utils import get_batch, load_corpus
    from utils.seed import set_seed

    set_seed(SEED)

    data, _, itos = load_corpus(DATASET_PATH)
    x, _ = get_batch(data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)

    model = GPT(vocab_size=len(itos), block_size=BLOCK_SIZE, d_model=D_MODEL)
    out = model(x)

    print(f"ids: {tuple(x.shape)} -> embeddings: {tuple(out.shape)}")
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")
