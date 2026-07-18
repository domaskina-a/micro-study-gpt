import re

PUNCTUATION = ".,!?"
TOKEN_PATTERN = re.compile(rf"[a-z']+|[{re.escape(PUNCTUATION)}]")


def tokenize(text: str) -> list[str]:
    """Split text into word-level tokens; punctuation becomes its own token."""
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(tokens: list[str]) -> tuple[dict[str, int], list[str]]:
    """Map every distinct token to an id, sorted for a stable vocabulary."""
    itos = sorted(set(tokens))
    stoi = {token: i for i, token in enumerate(itos)}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    """Turn raw text into token ids."""
    return [stoi[token] for token in tokenize(text)]


def decode(ids: list[int], itos: list[str]) -> str:
    """Turn token ids back into text. Punctuation sticks to the previous word."""
    text = ""
    for i in ids:
        token = itos[i]
        if text and token not in PUNCTUATION:
            text += " "
        text += token
    return text


if __name__ == "__main__":
    with open("dataset.txt", encoding="utf-8") as f:
        tokens = tokenize(f.read())

    stoi, itos = build_vocab(tokens)
    print(f"tokens: {len(tokens)}")
    print(f"vocab size: {len(itos)}")
    print(f"vocab: {' '.join(itos)}")
