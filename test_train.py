import torch
import torch.nn as nn

import train
from model import GPT

VOCAB_SIZE = 10


def _setup(monkeypatch, steps: int, lr: float = 3e-3):
    monkeypatch.setattr(train, "MAX_STEPS", steps)
    monkeypatch.setattr(train, "LOG_INTERVAL", steps + 1)

    model = GPT(vocab_size=VOCAB_SIZE, block_size=8, d_model=32, ffn_multiplier=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # A short cycle repeated over and over: the model can only lower the loss by
    # picking up the pattern.
    data = torch.tensor([1, 2, 3, 4] * 50)

    return model, data, nn.CrossEntropyLoss(), optimizer


def _loss_on(model, loss_fn, data):
    x, y = data[:8].unsqueeze(0), data[1:9].unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
    return loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1)).item()


def test_loss_drops_over_training(monkeypatch):
    model, data, loss_fn, optimizer = _setup(monkeypatch, steps=200)
    before = _loss_on(model, loss_fn, data)

    train.train(model, data, loss_fn, optimizer)

    assert _loss_on(model, loss_fn, data) < before / 2


def test_a_step_updates_the_parameters(monkeypatch):
    model, data, loss_fn, optimizer = _setup(monkeypatch, steps=1)
    before = model.lm_head.weight.clone()

    train.train(model, data, loss_fn, optimizer)

    assert not torch.equal(model.lm_head.weight, before)


def test_gradients_do_not_accumulate_between_steps(monkeypatch):
    # With lr=0 the parameters never move, so the gradient left after two steps
    # is the one of the last backward alone — unless the earlier one was never
    # cleared, in which case the two sum up to roughly double.
    torch.manual_seed(0)
    model, data, loss_fn, optimizer = _setup(monkeypatch, steps=1, lr=0.0)
    train.train(model, data, loss_fn, optimizer)
    one_step = model.lm_head.weight.grad.norm()

    torch.manual_seed(0)
    monkeypatch.setattr(train, "MAX_STEPS", 2)
    train.train(model, data, loss_fn, optimizer)

    assert model.lm_head.weight.grad.norm() < one_step * 1.5
