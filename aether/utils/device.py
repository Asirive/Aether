from __future__ import annotations

import torch


def select_device(prefer_discrete: bool = True) -> torch.device:
    """Return the CUDA device with the most VRAM, and make it the current device.

    On machines with an integrated GPU the default device 0 may be the iGPU,
    which fails (driver access violation) on kernels it does not support.
    Calling `torch.cuda.set_device` also makes the bare `"cuda"` alias resolve
    to the chosen device, so existing code that says `"cuda"` is fixed too.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    count = torch.cuda.device_count()
    if count <= 1 or not prefer_discrete:
        torch.cuda.set_device(0)
        return torch.device("cuda", 0)

    best = max(
        range(count),
        key=lambda i: torch.cuda.get_device_properties(i).total_memory,
    )
    torch.cuda.set_device(best)
    return torch.device("cuda", best)


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return f"{props.name} ({props.total_memory / 1e9:.2f} GB, index {index})"
