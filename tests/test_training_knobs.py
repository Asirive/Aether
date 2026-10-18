from __future__ import annotations

import types

import torch

from aether.model import AetherConfig, AetherForCausalLM
from aether.training.curriculum import (
    CurriculumScheduler,
    aux_loss_weights,
    fsa_gumbel_noise,
    fsa_temperature,
)
from aether.training.optim import _split_param_groups
from aether.training.trainer import PretrainTrainer, compute_grad_accum


def _stub_trainer(
    grad_clip: float = 1.0,
    threshold: float = 10.0,
    grad_accum: int = 1,
    max_consecutive_skips: int = 20,
) -> PretrainTrainer:
    """A PretrainTrainer with only the fields the guard paths touch."""
    cfg = AetherConfig(
        vocab_size=64,
        d_model=16,
        d_inner=32,
        n_layers=1,
        n_heads=2,
        d_k=8,
        max_seq_len=32,
        mod_routing_points=(),
        mod_keep_ratios=(),
    )
    trainer = object.__new__(PretrainTrainer)
    trainer.model = AetherForCausalLM(cfg)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.1)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda s: 1.0)
    trainer.logger = types.SimpleNamespace(log=lambda _msg: None)
    trainer.cfg = types.SimpleNamespace(optim={"grad_clip": grad_clip, "grad_spike_threshold": threshold})
    trainer.state = types.SimpleNamespace(step=0)
    trainer.grad_accum = grad_accum
    trainer._grad_spike_threshold = threshold
    trainer._force_update_once = False
    trainer._max_consecutive_skips = max_consecutive_skips
    trainer._skipped_spikes = 0
    trainer._loss_ema = 0.0
    trainer._consecutive_skips = 0
    trainer._skip_streak = 0
    trainer._wedged_skip_limit = 200
    return trainer


def _set_grads(trainer: PretrainTrainer, value: float) -> None:
    for p in trainer.model.parameters():
        p.grad = torch.full_like(p, value)


def _weights(trainer: PretrainTrainer) -> list[torch.Tensor]:
    return [p.detach().clone() for p in trainer.model.parameters()]


def _unchanged(before: list[torch.Tensor], trainer: PretrainTrainer) -> bool:
    return all(torch.equal(b, p.detach()) for b, p in zip(before, trainer.model.parameters()))


def test_optimizer_step_open_accumulation_window():
    trainer = _stub_trainer(grad_accum=4)
    norm, skipped = trainer._optimizer_step(0)
    assert torch.isnan(torch.tensor(norm)) and skipped is False


def test_optimizer_step_applies_healthy_update():
    torch.manual_seed(0)
    trainer = _stub_trainer()
    n_params = sum(p.numel() for p in trainer.model.parameters())
    _set_grads(trainer, 0.01)
    before = _weights(trainer)
    norm, skipped = trainer._optimizer_step(0)
    assert skipped is False
    assert abs(norm - 0.01 * n_params**0.5) < 1e-4
    assert not _unchanged(before, trainer)


def test_optimizer_step_skips_grad_spike_without_touching_weights():
    trainer = _stub_trainer(threshold=10.0)
    _set_grads(trainer, 100.0)
    before = _weights(trainer)
    norm, skipped = trainer._optimizer_step(0)
    assert skipped is True
    assert norm > 10.0
    assert _unchanged(before, trainer)
    assert trainer._skipped_spikes == 1


def test_optimizer_step_skips_non_finite_gradients():
    trainer = _stub_trainer()
    _set_grads(trainer, float("nan"))
    before = _weights(trainer)
    norm, skipped = trainer._optimizer_step(0)
    assert torch.isnan(torch.tensor(norm)) and skipped is True
    assert _unchanged(before, trainer)


def test_forced_update_never_applies_non_finite_gradients():
    """Force update exists to escape a skip storm; it must still refuse nan/inf."""
    trainer = _stub_trainer()
    trainer._force_update_once = True
    _set_grads(trainer, float("nan"))
    before = _weights(trainer)
    norm, skipped = trainer._optimizer_step(0)
    assert skipped is True
    assert _unchanged(before, trainer)
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
    assert trainer._force_update_once is True


def test_forced_update_applies_finite_spike():
    trainer = _stub_trainer(threshold=10.0)
    trainer._force_update_once = True
    _set_grads(trainer, 100.0)
    before = _weights(trainer)
    norm, skipped = trainer._optimizer_step(0)
    assert skipped is False
    assert norm > 10.0
    assert not _unchanged(before, trainer)
    assert trainer._force_update_once is False
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())


def test_skip_state_arms_forced_update_after_threshold():
    trainer = _stub_trainer(max_consecutive_skips=3)
    trainer._loss_ema = 3.0
    for _ in range(2):
        trainer._update_skip_state(True, 4.0)
    assert trainer._force_update_once is False
    assert trainer._consecutive_skips == 2
    trainer._update_skip_state(True, 4.0)
    assert trainer._force_update_once is True
    assert trainer._consecutive_skips == 0
    assert trainer._skip_streak == 3


def test_skip_state_advances_counters_when_loss_is_non_finite():
    """A nan-gradient regime presents as a nan loss; counters must still advance."""
    trainer = _stub_trainer(max_consecutive_skips=3)
    trainer._loss_ema = 3.0
    for _ in range(3):
        trainer._update_skip_state(True, float("nan"))
    assert trainer._force_update_once is True
    assert trainer._skip_streak == 3
    assert trainer._loss_ema == 3.0


def test_skip_state_resets_and_tracks_ema_on_applied_steps():
    trainer = _stub_trainer()
    trainer._consecutive_skips = 5
    trainer._skip_streak = 7
    trainer._loss_ema = 0.0
    trainer._update_skip_state(False, 4.0)
    assert trainer._consecutive_skips == 0
    assert trainer._skip_streak == 0
    assert trainer._loss_ema == 4.0
    trainer._update_skip_state(False, 3.0)
    assert abs(trainer._loss_ema - (0.95 * 4.0 + 0.05 * 3.0)) < 1e-9


def test_skip_state_tracks_loss_slowly_while_skipping():
    trainer = _stub_trainer(max_consecutive_skips=1000)
    trainer._loss_ema = 10.0
    trainer._update_skip_state(True, 20.0)
    assert 10.0 < trainer._loss_ema < 20.0
    assert trainer._skip_streak == 1


def test_compute_grad_accum_tracks_seq_len():
    assert compute_grad_accum(65536, 4, 1024) == 16
    assert compute_grad_accum(65536, 4, 2048) == 8
    assert compute_grad_accum(8192, 4, 1024) == 2
    assert compute_grad_accum(1, 4, 4096) == 1


def test_curriculum_scheduler_phases():
    sched = CurriculumScheduler(1000, phases=((0.0, 0.2, 1024), (0.2, 0.5, 2048), (0.5, 1.0, 2048)))
    assert sched.seq_len_for_step(0) == 1024
    assert sched.seq_len_for_step(199) == 1024
    assert sched.seq_len_for_step(200) == 2048
    assert sched.seq_len_for_step(999) == 2048


def test_fsa_temperature_anneals_to_point_one():
    assert fsa_temperature(0, 1000, 0.1) == 1.0
    mid = fsa_temperature(50, 1000, 0.1)
    assert 0.4 < mid < 0.7
    assert abs(fsa_temperature(100, 1000, 0.1) - 0.1) < 1e-6
    assert abs(fsa_temperature(999, 1000, 0.1) - 0.1) < 1e-9


def test_fsa_gumbel_anneals_to_zero():
    assert fsa_gumbel_noise(0, 1000, 0.1) == 1.0
    assert fsa_gumbel_noise(100, 1000, 0.1) == 0.0
    assert fsa_gumbel_noise(999, 1000, 0.1) == 0.0


def test_aux_loss_weights_ramp():
    w0 = aux_loss_weights(0, 1000, 0.001, 0.1, 0.01, 0.05, 0.01, 0.05)
    assert w0["fsa_ortho"] == 0.0
    assert w0["polymamba_kbal"] == 0.0
    assert w0["mod"] == 0.0
    w_end = aux_loss_weights(999, 1000, 0.001, 0.1, 0.01, 0.05, 0.01, 0.05)
    assert abs(w_end["fsa_ortho"] - 0.001) < 1e-9
    assert abs(w_end["polymamba_kbal"] - 0.01) < 1e-9
    assert abs(w_end["mod"] - 0.01) < 1e-9


def test_weight_decay_exclusions():
    torch.manual_seed(0)
    config = AetherConfig(
        vocab_size=128, d_model=64, d_inner=128, n_layers=4, n_heads=4, d_k=16,
        max_seq_len=64, mod_routing_points=(2,), mod_keep_ratios=(0.5,),
    )
    model = AetherForCausalLM(config)
    groups = _split_param_groups(model, weight_decay=0.1)
    decay_names = {n for n, p in model.named_parameters() if any(p is q for q in groups[0]["params"])}
    no_decay_names = {n for n, p in model.named_parameters() if any(p is q for q in groups[1]["params"])}

    assert any("token_embedding" in n for n in no_decay_names), "embedding must be no-decay"
    assert any("A_log" in n for n in no_decay_names), "A_log must be no-decay"
    assert any("mask_logits" in n for n in no_decay_names), "mask_logits must be no-decay"
    assert any("route_proj" in n for n in no_decay_names), "route_proj must be no-decay"
    assert any("mixer.gate" in n for n in no_decay_names), "polymamba gate must be no-decay"
    assert any("norm" in n.lower() for n in no_decay_names), "norms must be no-decay"
    assert any("w_q" in n for n in decay_names), "attention projections must be decayed"
    assert any("ffn.w_gate" in n for n in decay_names), "SwiGLU gate projection must be decayed"
    assert not decay_names & no_decay_names


def test_model_set_component_knobs():
    torch.manual_seed(0)
    config = AetherConfig(
        vocab_size=128, d_model=64, d_inner=128, n_layers=4, n_heads=4, d_k=16, max_seq_len=64,
    )
    model = AetherForCausalLM(config)
    model.set_component_knobs(fsa_temperature=0.55, fsa_gumbel=0.42, mod_temperature=0.33)
    from aether.components.fsa import FreqSparseAttention
    for block in model.blocks:
        if isinstance(block.mixer, FreqSparseAttention):
            assert block.mixer._temperature == 0.55
            assert block.mixer._gumbel_noise == 0.42
    assert model.mod_router._temperature == 0.33