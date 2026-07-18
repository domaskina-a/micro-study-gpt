from tokenizer import build_vocab, decode, encode, tokenize


def test_tokenize_lowercases():
    assert tokenize("The Sun") == ["the", "sun"]


def test_tokenize_splits_punctuation_into_own_token():
    assert tokenize("the sun shines.") == ["the", "sun", "shines", "."]


def test_tokenize_keeps_apostrophe_inside_word():
    assert tokenize("the moon's light") == ["the", "moon's", "light"]


def test_tokenize_drops_unsupported_characters():
    assert tokenize("the sun (bright) 42") == ["the", "sun", "bright"]


def test_build_vocab_deduplicates():
    stoi, itos = build_vocab(["the", "sun", "the"])
    assert itos == ["sun", "the"]
    assert len(stoi) == 2


def test_build_vocab_ids_are_stable_across_runs():
    # Iterating a set gives an order that varies per process, so ids must come
    # from sorting: a vocabulary saved with a model has to survive a restart.
    _, itos = build_vocab(["sun", "the", "."])
    assert itos == sorted(itos)
    assert build_vocab(["sun", "the", "."]) == build_vocab([".", "the", "sun"])


def test_build_vocab_maps_ids_back_to_tokens():
    tokens = tokenize("the sun shines in the sky.")
    stoi, itos = build_vocab(tokens)
    assert all(itos[stoi[token]] == token for token in tokens)


def test_corpus_vocab_covers_every_token():
    with open("dataset.txt", encoding="utf-8") as f:
        tokens = tokenize(f.read())

    stoi, _ = build_vocab(tokens)
    assert set(stoi) == set(tokens)


def test_encode_returns_vocab_ids():
    stoi, itos = build_vocab(tokenize("the sun shines."))
    assert [itos[i] for i in encode("the sun.", stoi)] == ["the", "sun", "."]


def test_decode_attaches_punctuation_to_previous_word():
    _, itos = build_vocab(["the", "sun", "."])
    assert decode([2, 1, 0], itos) == "the sun."


def test_round_trip_on_corpus():
    with open("dataset.txt", encoding="utf-8") as f:
        text = f.read()

    stoi, itos = build_vocab(tokenize(text))
    assert tokenize(decode(encode(text, stoi), itos)) == tokenize(text)
