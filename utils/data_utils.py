import torch

from utils.tokenizer import build_vocab, encode, tokenize


def load_corpus(path: str) -> tuple[torch.Tensor, dict[str, int], list[str]]:
    """Read the corpus and return it as one flat stream of token ids."""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    stoi, itos = build_vocab(tokenize(text))
    return torch.tensor(encode(text, stoi), dtype=torch.long), stoi, itos


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random training windows.

    Targets are the inputs shifted by one: predicting the next token at every
    position gives block_size training signals per window instead of one.
    """
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in starts])
    return x, y


if __name__ == "__main__":
    from config.hyperparams import BATCH_SIZE, BLOCK_SIZE, DATASET_PATH, SEED
    from utils.seed import set_seed

    set_seed(SEED)

    data, stoi, itos = load_corpus(DATASET_PATH)
    x, y = get_batch(data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)

    print(f"corpus: {len(data)} tokens, vocab {len(itos)}")
    print(f"x: {tuple(x.shape)}, y: {tuple(y.shape)}\n")
    for row_x, row_y in zip(x, y):
        print(f"  in : {' '.join(itos[i] for i in row_x)}")
        print(f"  out: {' '.join(itos[i] for i in row_y)}\n")
