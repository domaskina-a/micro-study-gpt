import torch


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")

    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    if not available:
        raise SystemExit("CUDA is not available; check the driver and PyTorch install.")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    x = torch.rand(3, device=device)
    print(f"Test tensor on {x.device}: {x}")


if __name__ == "__main__":
    main()
