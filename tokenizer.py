import re

TOKEN_PATTERN = re.compile(r"[a-z']+|[.,!?]")


def tokenize(text: str) -> list[str]:
    """Split text into word-level tokens; punctuation becomes its own token."""
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(tokens: list[str]) -> tuple[dict[str, int], list[str]]:
    """Map every distinct token to an id, sorted for a stable vocabulary."""
    itos = sorted(set(tokens))
    stoi = {token: i for i, token in enumerate(itos)}
    return stoi, itos


if __name__ == "__main__":
    with open("dataset.txt", encoding="utf-8") as f:
        tokens = tokenize(f.read())

    stoi, itos = build_vocab(tokens)
    print(f"tokens: {len(tokens)}")
    print(f"vocab size: {len(itos)}")
    print(f"vocab: {' '.join(itos)}")
