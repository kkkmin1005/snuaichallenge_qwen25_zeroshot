from __future__ import annotations

import importlib.metadata as metadata
import platform
import subprocess

import torch


PACKAGES = [
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "qwen-vl-utils",
    "pandas",
    "pillow",
    "tqdm",
]


def main() -> None:
    print(f"python: {platform.python_version()}")
    print(f"torch: {torch.__version__}")
    print(f"torch cuda runtime: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print()
    for package in PACKAGES:
        print(f"{package}: {metadata.version(package)}")
    print()
    run(["nvidia-smi"])


def run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print(f"{cmd[0]}: not found")


if __name__ == "__main__":
    main()
