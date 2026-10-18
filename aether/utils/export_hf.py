from __future__ import annotations

import json
from pathlib import Path

import torch

_CONFIG_PY = """from transformers import PretrainedConfig


class AetherHFConfig(PretrainedConfig):
    model_type = "aether"

    def __init__(self, **kwargs):
        self.core_config = kwargs.pop("core_config", {})
        super().__init__(**kwargs)
"""

_MODELING_PY = """from __future__ import annotations

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from aether.model import AetherConfig, AetherForCausalLM

from .configuration_aether import AetherHFConfig


class AetherHFForCausalLM(PreTrainedModel):
    config_class = AetherHFConfig

    def __init__(self, config: AetherHFConfig):
        super().__init__(config)
        core_cfg = AetherConfig.from_dict(config.core_config)
        self.core = AetherForCausalLM(core_cfg)
        self.post_init()

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        out = self.core(input_ids, labels=labels, return_logits=True)
        return CausalLMOutputWithPast(logits=out.get("logits"), loss=out.get("loss"))

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
"""


def export_hf(model, tokenizer, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "configuration_aether.py").write_text(_CONFIG_PY)
    (output_dir / "modeling_aether.py").write_text(_MODELING_PY)

    state = {f"core.{k}": v for k, v in model.state_dict().items()}
    weight_dtype = str(next(iter(state.values())).dtype).replace("torch.", "")
    config_json = {
        "model_type": "aether",
        "auto_map": {
            "AutoConfig": "configuration_aether.AetherHFConfig",
            "AutoModelForCausalLM": "modeling_aether.AetherHFForCausalLM",
        },
        "core_config": dict(model.config.__dict__),
        "torch_dtype": weight_dtype,
    }
    with (output_dir / "config.json").open("w") as f:
        json.dump(config_json, f, indent=2, default=str)

    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in state.items()}, str(output_dir / "model.safetensors"))

    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def ensure_hf_export(model_path: str | Path) -> Path:
    from aether.model import AetherConfig, AetherForCausalLM
    from aether.training.tokenizer import build_tokenizer

    path = Path(model_path)
    if path.is_dir() and (path / "config.json").exists():
        return path

    if path.is_file():
        ckpt = torch.load(path, map_location="cpu")
        cfg_dict = ckpt.get("metadata", {}).get("model_config")
        if cfg_dict is None:
            raise ValueError(
                f"Checkpoint {path} has no model_config in metadata; cannot auto-export. "
                "Re-save the checkpoint with a recent trainer."
            )
        cfg = AetherConfig.from_dict(
            {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg_dict.items()}
        )
        model = AetherForCausalLM(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        tokenizer = build_tokenizer()
        export_dir = path.parent / f"{path.stem}_hf"
        return export_hf(model, tokenizer, export_dir)

    raise ValueError(f"Cannot interpret model path: {path}")
