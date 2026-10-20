from __future__ import annotations

import torch


def compute_aux_losses(model_outputs: dict) -> dict[str, torch.Tensor]:
    aux = model_outputs.get("aux_losses", {})
    return {
        "mod": aux.get("mod", torch.zeros(())),
        "fsa_ortho": aux.get("fsa_ortho", torch.zeros(())),
        "polymamba_kbal": aux.get("polymamba_kbal", torch.zeros(())),
    }
