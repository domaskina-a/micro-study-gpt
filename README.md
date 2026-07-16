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
check_cuda.py      # environment check: prints the GPU and runs a tensor on CUDA
requirements.txt   # pinned dependencies
```

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
