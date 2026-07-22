"""End-to-end tests: text in, text out, across tokenizer, data and model.

Every module is covered by its own tests; these guard the seams between them,
where a mismatch keeps each part green but breaks the pipeline.
"""

import torch
import torch.nn as nn

import train
from config.hyperparams import (
    BLOCK_SIZE,
    D_MODEL,
    DATASET_PATH,
    FFN_MULTIPLIER,
    MAX_STEPS,
    N_LAYERS,
    NUM_HEADS,
)
from model import GPT
from utils.data_utils import load_corpus
from utils.seed import set_seed
from utils.tokenizer import decode, encode, tokenize

PROMPT = "the sun"


def _corpus_and_model():
    data, stoi, itos = load_corpus(DATASET_PATH)
    model = GPT(
        vocab_size=len(itos),
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        ffn_multiplier=FFN_MULTIPLIER,
        n_layers=N_LAYERS,
    )
    return data, stoi, itos, model


def _generate_text(model, stoi, itos, max_new_tokens: int) -> str:
    ids = torch.tensor([encode(PROMPT, stoi)])
    return decode(model.generate(ids, max_new_tokens)[0].tolist(), itos)


def test_generated_text_uses_only_known_words():
    # An untrained model still has to produce decodable text: its head is sized
    # from the same vocabulary the tokenizer built, so every sampled id must be
    # a valid index into itos. A vocab_size the model does not share with the
    # tokenizer passes every per-module test and only fails here.
    _, stoi, itos, model = _corpus_and_model()

    text = _generate_text(model, stoi, itos, max_new_tokens=20)

    assert set(tokenize(text)) <= set(stoi)


def test_generation_keeps_the_prompt():
    # Checked on ids in test_model; here it survives encode and decode too.
    _, stoi, itos, model = _corpus_and_model()

    assert _generate_text(model, stoi, itos, max_new_tokens=5).startswith(PROMPT)


def test_a_trained_model_reproduces_corpus_word_pairs(monkeypatch):
    # The stage-1 goal: after training, greedy decoding should give back the
    # patterns of the corpus rather than arbitrary word soup. Adjacent pairs are
    # the weakest form of that which is still meaningful on a 380-token corpus.
    # Most of them, not all: the model recombines learned fragments into
    # sentences the corpus never spells out, and that is not a failure.
    set_seed(1337)
    data, stoi, itos, model = _corpus_and_model()

    monkeypatch.setattr(train, "LOG_INTERVAL", MAX_STEPS + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    train.train(model, data, nn.CrossEntropyLoss(), optimizer)

    tokens = tokenize(_generate_text(model, stoi, itos, max_new_tokens=20))
    corpus = [itos[i] for i in data.tolist()]
    known_pairs = set(zip(corpus, corpus[1:]))

    pairs = list(zip(tokens, tokens[1:]))
    known = sum(pair in known_pairs for pair in pairs)
    assert known / len(pairs) >= 0.9
