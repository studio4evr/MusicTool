"""Shared helpers for the optional PyTorch models."""

import functools


@functools.cache
def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@functools.cache
def device() -> str:
    """"cuda" when an NVIDIA GPU is usable, otherwise "cpu"."""
    if not torch_available():
        return "cpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
