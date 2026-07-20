# Micro-study-gpt

A Transformer language model built and trained from scratch, step by step.
A personal, non-commercial educational project.

The goal is not the final model but the *process*: attention, FFN and the
training loop are first implemented by hand on top of basic PyTorch operations,
and only later compared against the library equivalents.

## Hardware

Development targets a single laptop GPU. The **6 GB VRAM** budget is the main
constraint and drives model size, batch size and context length throughout.

| Component | Spec |
|-----------|------|
| CPU | Intel Core i7-13650HX (14C / 20T) |
| RAM | 16 GB + 8 GB swap |
| GPU | NVIDIA RTX 4050 Laptop, 6 GB VRAM (Ada Lovelace, CUDA) |

## Project layout

```
check_cuda.py         # environment check: prints the GPU and runs a tensor on CUDA
model.py              # the model, built by hand: embeddings, attention, FFN, head, generation
train.py              # training loop and the pipeline that wires data, model and optimizer
dataset.txt           # stage-1 toy corpus: 60 hand-crafted sentences
requirements.txt      # pinned dependencies
config/
  hyperparams.py      # seed, dataset path, model and training sizes
utils/
  seed.py             # set_seed(): reproducible runs across random / numpy / torch
  tokenizer.py        # word-level tokenizer: vocabulary, encode / decode
  data_utils.py       # corpus as a stream of token ids, random training batches
```

Modules under `utils/` are imported by other code; run them from the project root
(`python -m utils.data_utils` prints a sample batch). Scripts in the root are meant
to be run directly.

The stage-1 toy dataset (`dataset.txt`) is original, hand-crafted for this project
and covered by the repository's MIT license.

## Getting started

Requires a CUDA-capable PyTorch. Install the pinned dependencies and verify the
setup:

```bash
pip install -r requirements.txt
python check_cuda.py
```

Optionally isolate the dependencies in a virtual environment first:

```bash
python -m venv .venv && source .venv/bin/activate
```

## Stage 1 — a toy model, everything by hand

The first model is deliberately tiny: one attention head, one FFN, no LayerNorm and
no residual connections yet, trained on the 60-sentence toy corpus. It exists to show
the training loop and the internals of a Transformer block working end to end.

```bash
python train.py
```

Run with the fixed seed (`SEED = 1337`), 500 steps, `d_model = 32`, `block_size = 8`,
`batch_size = 4`:

| | Loss |
|---|---|
| Uniform-guess baseline, `ln(88)` | 4.4773 |
| Step 1 | 4.4461 |
| Step 500 | 1.2353 |

The loss starts at the baseline — an untrained model spreads its probability evenly
over the 88-word vocabulary — and drops well below it, so the model is genuinely
learning the corpus. The logged value is a single batch, not an average, so it
bounces between steps; a proper evaluation loop over train and validation splits
comes later.

Greedy generation from the prompt `the sun`:

```
the sun. the sun. the sun. the sky. the sky. the sky. the sky. the
```

The model has clearly picked up the sentence template of the corpus, but greedy
decoding always takes the single most likely token, so on a corpus this small it
falls into a loop. That is expected rather than a defect: temperature and top-k
sampling are added in a later stage.

## License

This project's code is released under the [MIT License](LICENSE).
