import math

import torch
import torch.nn as nn

from config.hyperparams import (
    BATCH_SIZE,
    BLOCK_SIZE,
    D_MODEL,
    DATASET_PATH,
    FFN_MULTIPLIER,
    LEARNING_RATE,
    LOG_INTERVAL,
    MAX_STEPS,
    SEED,
)
from model import GPT
from utils.data_utils import get_batch, load_corpus
from utils.seed import set_seed


def train(model, data, loss_fn, optimizer):
    """Run MAX_STEPS optimisation steps over random windows of data.

    Returns the model.
    """
    for step in range(MAX_STEPS):
        x, y = get_batch(data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)

        optimizer.zero_grad()

        logits = model(x)

        # CrossEntropyLoss wants (N, vocab_size) and (N,), so the batch and time
        # axes are folded together.
        loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))

        loss.backward()
        optimizer.step()

        done = step + 1
        if done % LOG_INTERVAL == 0 or done == 1:
            print(f"step={done} loss={loss.item():.4f}")

    return model


def setup_and_train():
    """Full training pipeline. Returns: model."""
    data, _, itos = load_corpus(DATASET_PATH)
    vocab_size = len(itos)
    print(f"corpus: {len(data)} tokens, vocab {vocab_size}")
    print(f"untrained baseline: {math.log(vocab_size):.4f}\n")

    model = GPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        ffn_multiplier=FFN_MULTIPLIER,
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    return train(model, data, loss_fn, optimizer)


if __name__ == "__main__":
    set_seed(SEED)
    model = setup_and_train()
