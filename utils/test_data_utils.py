import torch

from utils.data_utils import get_batch, load_corpus
from seed import set_seed
from tokenizer import tokenize


def test_load_corpus_keeps_every_token():
    with open("dataset.txt", encoding="utf-8") as f:
        tokens = tokenize(f.read())

    data, _, itos = load_corpus("dataset.txt")
    assert len(data) == len(tokens)
    assert [itos[i] for i in data] == tokens


def test_get_batch_shapes():
    data = torch.arange(100)
    x, y = get_batch(data, block_size=8, batch_size=4)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)


def test_targets_are_inputs_shifted_by_one():
    data = torch.arange(100)
    x, y = get_batch(data, block_size=8, batch_size=4)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_rows_are_contiguous_slices_of_the_corpus():
    # arange makes each id equal to its position, so a valid window is a run of
    # consecutive integers.
    data = torch.arange(100)
    x, _ = get_batch(data, block_size=8, batch_size=16)
    assert torch.equal(x.diff(dim=1), torch.ones(16, 7, dtype=data.dtype))


def test_windows_stay_inside_the_corpus():
    # The last target is data[start + block_size], so start must stop one short
    # of the usual bound; sample hard against the smallest possible corpus.
    data = torch.arange(9)
    x, y = get_batch(data, block_size=8, batch_size=64)
    assert x.min() >= 0 and y.max() <= 8


def test_batches_are_reproducible_after_seeding():
    data = torch.arange(100)

    set_seed(1337)
    first = get_batch(data, block_size=8, batch_size=4)
    set_seed(1337)
    second = get_batch(data, block_size=8, batch_size=4)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
