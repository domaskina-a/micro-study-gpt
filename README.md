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
dataset.txt           # stage-1 toy corpus: 60 hand-crafted sentences
requirements.txt      # pinned dependencies
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

## License

This project's code is released under the [MIT License](LICENSE).
