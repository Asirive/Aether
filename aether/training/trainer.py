from __future__ import annotations

import math
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.checkpoint import checkpoint

from aether.model import AetherConfig, AetherForCausalLM
from aether.training.curriculum import (
    CurriculumScheduler,
    aux_loss_weights,
    fsa_gumbel_noise,
    fsa_temperature,
)
from aether.training.data import FineWebEduStream
from aether.training.eval import (
    ARC_EASY_DEV_SAMPLES,
    CANARY_PROMPTS,
    GENERATION_PROMPTS,
    HELLASWAG_DEV_SAMPLES,
    evaluate_choices,
    generate_canary_samples,
    generate_samples,
    save_eval_log,
)
from aether.training.optim import build_optimizer, build_scheduler
from aether.training.tokenizer import build_tokenizer
from aether.utils.checkpoint import load_checkpoint, save_checkpoint
from aether.utils.device import describe_device, select_device
from aether.utils.tee_log import TeeLogger
from aether.utils.tui import TrainSnapshot, TrainingDashboard


@dataclass
class PretrainConfig:
    model: dict
    fsa: dict
    polymamba: dict
    mod: dict
    training: dict
    optim: dict
    data: dict
    eval: dict
    logging: dict
    tui: dict
    output: dict

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PretrainConfig":
        with Path(path).open() as f:
            raw = yaml.safe_load(f)
        return cls(**raw)


@dataclass
class RuntimeState:
    step: int = 0
    tokens_seen: int = 0
    last_eval_step: int = -1
    last_checkpoint_step: int = -1
    curriculum_seq_len: int = 0
    rng_seed_at_resume: int = 0


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def compute_grad_accum(effective_batch_tokens: int, micro_batch: int, seq_len: int) -> int:
    return max(1, effective_batch_tokens // (micro_batch * seq_len))


def _aux_value(t) -> float:
    if isinstance(t, torch.Tensor):
        return float(t.item())
    if isinstance(t, (int, float)):
        return float(t)
    return 0.0


def _apply_aux_ramping(
    model: AetherForCausalLM,
    step: int,
    total_steps: int,
    cfg: PretrainConfig,
) -> dict[str, float]:
    weights = aux_loss_weights(
        step=step,
        total_steps=total_steps,
        fsa_weight=cfg.fsa["ortho_loss_weight"],
        fsa_warmup=cfg.fsa.get("fsa_ortho_warmup_frac", 0.10),
        polymamba_weight=cfg.polymamba["kbal_loss_weight"],
        polymamba_warmup=cfg.polymamba.get("kbal_warmup_frac", 0.05),
        mod_weight=cfg.mod["loss_weight"],
        mod_warmup=cfg.mod.get("mod_warmup_frac", 0.05),
    )
    model.config.fsa_ortho_loss_weight = weights["fsa_ortho"]
    model.config.polymamba_kbal_loss_weight = weights["polymamba_kbal"]
    model.config.mod_loss_weight = weights["mod"]
    return weights


def _build_model(cfg: PretrainConfig) -> AetherForCausalLM:
    m = cfg.model
    aether_cfg = AetherConfig(
        vocab_size=m["vocab_size"],
        d_model=m["d_model"],
        d_inner=m["d_inner"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        d_k=m["d_k"],
        max_seq_len=max(m["max_seq_len"], cfg.training["seq_len_phases"][-1][2]),
        rope_base=m.get("rope_base", 10_000.0),
        tie_embeddings=m.get("tie_embeddings", True),
        fsa_topk_ratio=cfg.fsa.get("topk_ratio", 0.25),
        fsa_ortho_loss_weight=cfg.fsa.get("ortho_loss_weight", 0.001),
        polymamba_orders=cfg.polymamba.get("orders", 2),
        polymamba_state_dim=cfg.polymamba.get("state_dim", 16),
        polymamba_kbal_loss_weight=cfg.polymamba.get("kbal_loss_weight", 0.01),
        mod_routing_points=tuple(cfg.mod["routing_points"]),
        mod_keep_ratios=tuple(cfg.mod["keep_ratios"]),
        mod_loss_weight=cfg.mod.get("loss_weight", 0.01),
    )
    model = AetherForCausalLM(aether_cfg)
    return model


class PretrainTrainer:
    def __init__(self, cfg: PretrainConfig, resume_path: str | Path | None = None) -> None:
        self.cfg = cfg
        self.device = select_device()
        self.amp_dtype = torch.bfloat16

        self.total_steps = cfg.training["total_steps"]
        self.effective_batch_tokens = cfg.training["effective_batch_tokens"]
        self.micro_batch = cfg.training["micro_batch"]
        self.grad_accum = compute_grad_accum(
            self.effective_batch_tokens, self.micro_batch, cfg.training["seq_len_phases"][0][2]
        )

        self.tokenizer = build_tokenizer(cfg.data.get("tokenizer", "unsloth/Llama-3.2-1B-Instruct"))
        self.model = _build_model(cfg).to(self.device)

        self.dataset = FineWebEduStream(
            tokenizer=self.tokenizer,
            seq_len=cfg.training["seq_len_phases"][0][2],
            seed=42,
        )

        self.optimizer = build_optimizer(
            self.model,
            lr=cfg.optim.get("lr", 5e-4),
            betas=tuple(cfg.optim.get("betas", (0.9, 0.95))),
            weight_decay=cfg.optim.get("weight_decay", 0.1),
            embed_lr_scale=cfg.optim.get("embed_lr_scale", 1.0),
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=self.total_steps,
            warmup_steps=cfg.optim.get("warmup_steps", 1000),
            min_lr_ratio=cfg.optim.get("min_lr_ratio", 0.1),
        )

        self.curriculum = CurriculumScheduler(
            self.total_steps,
            phases=tuple(tuple(p) for p in cfg.training["seq_len_phases"]),
        )
        self.state = RuntimeState()
        self._crash_signaled = False
        self._loss_window: deque[float] = deque(maxlen=200)
        self._loss_ema: float = 0.0
        self._skipped_spikes: int = 0
        self._grad_spike_threshold: float = float(cfg.optim.get("grad_spike_threshold", 10.0))
        self._max_consecutive_skips: int = int(cfg.optim.get("max_consecutive_skips", 20))
        self._wedged_skip_limit: int = int(cfg.optim.get("wedged_skip_limit", 200))
        self._consecutive_skips: int = 0
        self._skip_streak: int = 0
        self._force_update_once: bool = False

        if resume_path is not None and Path(resume_path).exists():
            ckpt = load_checkpoint(resume_path, device="cpu")
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.state.step = ckpt["step"]
            self.state.tokens_seen = ckpt.get("metadata", {}).get("tokens_seen", 0)
            print(f"[resume] loaded step={self.state.step} from {resume_path}", flush=True)

        output_dir = Path(self.cfg.output["dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = TeeLogger(output_dir / "train.log")

        self._grad_checkpointing = bool(cfg.training.get("gradient_checkpointing", True))
        if self._grad_checkpointing:
            self._wire_gradient_checkpointing()

    def _wire_gradient_checkpointing(self) -> None:
        for block in self.model.blocks:
            block.forward = self._ckpt_block_forward(block)  # type: ignore[assignment]

    def _ckpt_block_forward(self, block):
        original_forward = block.forward

        def ckpt_forward(x: torch.Tensor) -> torch.Tensor:
            return checkpoint(original_forward, x, use_reentrant=False)

        return ckpt_forward

    def _save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            self.model,
            self.optimizer,
            step=self.state.step,
            path=path,
            scheduler=self.scheduler,
            metadata={
                "tokens_seen": self.state.tokens_seen,
                "loss_window_mean": (
                    sum(self._loss_window) / len(self._loss_window) if self._loss_window else 0.0
                ),
                "model_config": dict(self.model.config.__dict__),
                "config_snapshot": {
                    "seq_len": self.state.curriculum_seq_len,
                    "lr": self.optimizer.param_groups[0]["lr"],
                },
            },
        )
        atomic_latest = path.parent / "latest.pt"
        if atomic_latest.exists() or atomic_latest.is_symlink():
            atomic_latest.unlink()
        try:
            atomic_latest.symlink_to(path.name)
        except OSError:
            try:
                (path.parent / "latest.txt").write_text(path.name)
            except OSError:
                pass

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):
            self._crash_signaled = True
            print(f"\n[signal] Caught signal {signum}, will save checkpoint after current step", flush=True)

        if sys.platform != "win32":
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)

    def _install_keybind_listener(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import msvcrt
        except ImportError:
            return

        def _listener() -> None:
            while not self._crash_signaled:
                try:
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in (b"q", b"Q", b"s", b"S"):
                            self._crash_signaled = True
                            self.logger.log(
                                f"[keybind] '{ch.decode()}' pressed — saving checkpoint and stopping after current step"
                            )
                            break
                except Exception:
                    break
                time.sleep(0.25)

        import threading
        t = threading.Thread(target=_listener, name="keybind-listener", daemon=True)
        t.start()
        self.logger.log("[keybind] press 'q' or 's' at any time to save checkpoint and stop")

    def _train_step(
        self,
        batch: dict,
        seq_len: int,
        accum_loss: list[float],
        accum_mod: list[float],
        accum_fsa: list[float],
        accum_pm: list[float],
        accum_tokens: list[int],
        step_time_acc: list[float],
    ) -> None:
        batch = _move_batch(batch, self.device)
        start = time.perf_counter()
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            out = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
        loss = out["loss"] / self.grad_accum
        loss.backward()
        step_time_acc.append(time.perf_counter() - start)
        accum_loss.append(float(out["loss"].item()))
        aux = out.get("aux_losses", {})
        accum_mod.append(_aux_value(aux.get("mod")))
        accum_fsa.append(_aux_value(aux.get("fsa_ortho")))
        accum_pm.append(_aux_value(aux.get("polymamba_kbal")))
        accum_tokens.append(int(batch["input_ids"].numel()))

    def _optimizer_step(self, micro_step_in_accum: int, skip: bool = False) -> tuple[float, bool]:
        """Run one optimizer step when the accumulation window closes.

        Returns (pre_clip_grad_norm, skipped). `skipped` is True only when an
        update was due and was deliberately dropped; it is False while the
        accumulation window is still open and when the update was applied.
        """
        if (micro_step_in_accum + 1) % self.grad_accum == 0:
            params = [p for p in self.model.parameters() if p.grad is not None]
            grad_norm_val = float(
                torch.nn.utils.clip_grad_norm_(params, float("inf")).item()
            )
            finite = math.isfinite(grad_norm_val)
            grad_bad = (not finite) or grad_norm_val > self._grad_spike_threshold
            force = self._force_update_once and finite
            if (skip or grad_bad) and not force:
                self._log_grad_hotspots()
                self.optimizer.zero_grad(set_to_none=True)
                self._skipped_spikes += 1
                reason = "loss-spike" if skip else ("grad-non-finite" if not finite else "grad-norm")
                self.logger.log(
                    f"[spike-guard] skipped update step={self.state.step + 1} reason={reason} "
                    f"grad_norm={grad_norm_val:.2e} (total skipped: {self._skipped_spikes})"
                )
                return grad_norm_val, True
            if force:
                if skip or grad_bad:
                    self.logger.log(
                        f"[spike-guard] forced clipped update step={self.state.step + 1} "
                        f"grad_norm={grad_norm_val:.2e} after {self._skip_streak} consecutive skips"
                    )
                self._force_update_once = False
            torch.nn.utils.clip_grad_norm_(params, self.cfg.optim.get("grad_clip", 1.0))
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            return grad_norm_val, False
        return float("nan"), False

    def _log_grad_hotspots(self) -> None:
        try:
            norms = [
                (float(p.grad.detach().norm().item()), name)
                for name, p in self.model.named_parameters()
                if p.grad is not None
            ]
            top = sorted(norms, reverse=True)[:5]
            self.logger.log(
                "[grad-diagnostic] top-5 param grads (pre-clip): "
                + ", ".join(f"{n}={v:.1e}" for v, n in top)
            )
        except Exception:
            pass

    def _update_skip_state(self, step_was_skipped: bool, loss_mean: float) -> None:
        """Anti-freeze bookkeeping, driven by whether an update was applied.

        Runs even when the loss is non-finite, otherwise a skip storm caused by
        non-finite gradients would never arm the forced update. The EMA re-seed
        itself only uses finite losses; the mean of the last 20 applied losses
        is the reference the loss-spike test compares against.
        """
        if step_was_skipped:
            self._consecutive_skips += 1
            self._skip_streak += 1
            if math.isfinite(loss_mean) and self._loss_ema > 0:
                self._loss_ema = 0.995 * self._loss_ema + 0.005 * loss_mean
            if self._consecutive_skips >= self._max_consecutive_skips:
                if math.isfinite(loss_mean):
                    self._loss_ema = loss_mean
                    detail = f"re-seeding loss EMA to {loss_mean:.4f}"
                else:
                    detail = "loss still non-finite, keeping previous EMA"
                self.logger.log(
                    f"[spike-guard] {self._consecutive_skips} consecutive skips — "
                    f"{detail} and arming a forced clipped update"
                )
                self._consecutive_skips = 0
                self._force_update_once = True
            return

        self._consecutive_skips = 0
        self._skip_streak = 0
        if math.isfinite(loss_mean):
            if self._loss_ema <= 0:
                self._loss_ema = loss_mean
            else:
                self._loss_ema = 0.95 * self._loss_ema + 0.05 * loss_mean

    def _run_periodic_eval(self, step: int) -> dict:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        was_training = self.model.training
        self.model.eval()

        def forward_fn(ids):
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                return self.model(ids, labels=None)

        hella = evaluate_choices(forward_fn, self.tokenizer, HELLASWAG_DEV_SAMPLES, self.device, task="hellaswag")
        arc = evaluate_choices(forward_fn, self.tokenizer, ARC_EASY_DEV_SAMPLES, self.device, task="arc_easy")
        gens = generate_samples(
            forward_fn,
            self.tokenizer,
            self.device,
            prompts=GENERATION_PROMPTS,
            max_new_tokens=self.cfg.eval.get("max_new_tokens", 24),
        )
        eval_log_path = Path(self.cfg.output["dir"]) / f"eval_step_{step:08d}.json"
        save_eval_log(
            {"step": step, "hellaswag": hella, "arc_easy": arc},
            gens,
            eval_log_path,
        )
        result = {
            "step": step,
            "hellaswag_acc": hella["accuracy"],
            "arc_easy_acc": arc["accuracy"],
            "hellaswag_details": hella["details"],
            "arc_easy_details": arc["details"],
            "generations": gens,
        }
        if was_training:
            self.model.train()
        return result

    def _gather_health(self) -> tuple[dict[int, float], dict[int, float], float, float]:
        ratios: dict[int, float] = {}
        targets: dict[int, float] = {}
        score_rms = 0.0
        state = getattr(self.model, "_last_router_state", {}) or {}
        for k, v in state.items():
            if k.startswith("_last_score_rms_"):
                score_rms = max(score_rms, float(v))
                continue
            if not k.startswith("_last_ratio_"):
                continue
            layer_id = int(k.removeprefix("_last_ratio_"))
            ratios[layer_id] = float(v)
            if layer_id in self.cfg.mod["routing_points"]:
                idx = self.cfg.mod["routing_points"].index(layer_id)
                targets[layer_id] = self.cfg.mod["keep_ratios"][idx]
        pm_max = 0.0
        for block in self.model.blocks:
            from aether.components.polymamba import PolyMamba2
            if isinstance(block.mixer, PolyMamba2):
                try:
                    radii = block.mixer.spectral_radius()
                    if radii.numel():
                        pm_max = max(pm_max, float(radii.detach().max().item()))
                except Exception:
                    pass
        return ratios, targets, pm_max, score_rms

    def run(self) -> None:
        self._install_signal_handlers()
        self._install_keybind_listener()

        eval_every = self.cfg.eval.get("every_steps", 500)
        save_every = self.cfg.output.get("save_every_steps", 5000)
        log_every = self.cfg.logging.get("log_every_steps", 50)
        use_tui = self.cfg.tui.get("enabled", True)
        output_dir = Path(self.cfg.output["dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.log(
            f"start step={self.state.step} total_steps={self.total_steps} "
            f"seq_len={self.curriculum.seq_len_for_step(self.state.step)} "
            f"micro_batch={self.micro_batch} grad_accum={self.grad_accum} "
            f"device={describe_device(self.device)} master=fp32 compute=bf16 use_tui={use_tui} "
            f"grad_spike_threshold={self._grad_spike_threshold}"
        )
        self.run_start_time = time.time()

        try:
            if use_tui:
                self._run_loop(
                    eval_every, save_every, log_every, output_dir,
                    snapshot_handler=lambda snap: None,
                    dashboard_factory=lambda: TrainingDashboard(
                        total_steps=self.total_steps,
                        refresh_per_second=self.cfg.tui.get("refresh_hz", 4),
                    ),
                )
            else:
                self._run_loop(
                    eval_every, save_every, log_every, output_dir,
                    snapshot_handler=lambda snap: None,
                    dashboard_factory=None,
                )
        except KeyboardInterrupt:
            self.logger.log("interrupted by user (Ctrl+C) — saving checkpoint")
            self._save(output_dir / "interrupted.pt")
            self.logger.log(f"checkpoint saved; resume with --resume={output_dir / 'interrupted.pt'}")

    def _run_loop(
        self,
        eval_every: int,
        save_every: int,
        log_every: int,
        output_dir: Path,
        snapshot_handler,
        dashboard_factory,
    ) -> None:
        from contextlib import nullcontext

        ctx = (
            dashboard_factory()
            if dashboard_factory is not None
            else nullcontext()
        )
        with ctx as dashboard:
            last_batch: dict | None = None
            step_time_window: deque[float] = deque(maxlen=20)
            running_loss = 0.0
            running_tokens = 0
            running_aux_mod = 0.0
            running_aux_fsa = 0.0
            running_aux_pm = 0.0
            running_count = 0
            last_samples: list[tuple[str, str]] = []
            last_canary_step: int = 0
            last_batch_preview: str = ""
            last_eval: dict | None = None
            recent_warns: deque[str] = deque(maxlen=5)

            step_idx = self.state.step
            while step_idx < self.total_steps and not self._crash_signaled:
                seq_len = self.curriculum.seq_len_for_step(step_idx)
                self.state.curriculum_seq_len = seq_len
                self.dataset.set_seq_len(seq_len)
                self.grad_accum = compute_grad_accum(
                    self.effective_batch_tokens, self.micro_batch, seq_len
                )

                weights = _apply_aux_ramping(self.model, step_idx, self.total_steps, self.cfg)
                fsa_warmup = self.cfg.fsa.get("fsa_ortho_warmup_frac", 0.10)
                fsa_temp = fsa_temperature(step_idx, self.total_steps, fsa_warmup)
                gumbel = fsa_gumbel_noise(step_idx, self.total_steps, fsa_warmup)
                self.model.set_component_knobs(fsa_temperature=fsa_temp, fsa_gumbel=gumbel)
                step_wall_start = time.time()

                accum_count = 0
                accum_loss: list[float] = []
                accum_tokens: list[int] = []
                accum_mod: list[float] = []
                accum_fsa: list[float] = []
                accum_pm: list[float] = []
                accum_grad_norm = 0.0
                accum_grad_n = 0
                step_times: list[float] = []
                skip_this_step = False
                step_was_skipped = False
                while accum_count < self.grad_accum:
                    try:
                        batch = self.dataset.next_batch(self.micro_batch)
                    except Exception:
                        self.dataset._iter = None
                        batch = self.dataset.next_batch(self.micro_batch)

                    self._train_step(
                        batch, seq_len,
                        accum_loss, accum_mod, accum_fsa, accum_pm, accum_tokens,
                        step_times,
                    )
                    last_batch = _move_batch(batch, self.device)
                    accum_count += 1

                    if accum_count == self.grad_accum and accum_loss and self._loss_ema > 0:
                        pending_loss = sum(accum_loss) / len(accum_loss)
                        if math.isfinite(pending_loss) and pending_loss > self._loss_ema * 1.5:
                            skip_this_step = True

                    gn, skipped = self._optimizer_step(accum_count - 1, skip=skip_this_step)
                    if math.isfinite(gn):
                        accum_grad_norm += gn
                        accum_grad_n += 1
                    if skipped:
                        step_was_skipped = True

                step_idx += 1
                self.state.step = step_idx
                total_accum_tokens = sum(accum_tokens)
                self.state.tokens_seen += total_accum_tokens

                loss_mean = sum(accum_loss) / max(1, len(accum_loss))
                mod_mean = sum(accum_mod) / max(1, len(accum_mod))
                fsa_mean = sum(accum_fsa) / max(1, len(accum_fsa))
                pm_mean = sum(accum_pm) / max(1, len(accum_pm))
                grad_norm = (accum_grad_norm / accum_grad_n) if accum_grad_n else 0.0
                step_time_sum = sum(step_times) if step_times else 1e-9
                step_wall = time.time() - step_wall_start
                tokens_per_sec = total_accum_tokens / max(step_wall, 1e-9)

                if math.isfinite(loss_mean):
                    self._loss_window.append(loss_mean)
                    running_loss = loss_mean

                self._update_skip_state(step_was_skipped, loss_mean)
                if self._skip_streak >= self._wedged_skip_limit:
                    self.logger.log(
                        f"[spike-guard] FATAL: {self._skip_streak} consecutive skipped updates — "
                        f"training cannot progress (last grad_norm={grad_norm:.2e}); "
                        f"stopping at step={step_idx} so the checkpoint can be inspected"
                    )
                    self._crash_signaled = True
                running_mod = mod_mean
                running_fsa = fsa_mean
                running_pm = pm_mean
                running_tokens += total_accum_tokens
                running_count += 1

                canary_interval = self.cfg.eval.get("canary_every_steps", 10)
                if step_idx % canary_interval == 0:
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        was_training = self.model.training
                        self.model.eval()

                        def forward_fn(ids):
                            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                                return self.model(ids, labels=None)

                        canary = generate_canary_samples(
                            forward_fn,
                            self.tokenizer,
                            self.device,
                            prompts=CANARY_PROMPTS[:3],
                            max_new_tokens=self.cfg.eval.get("canary_max_new_tokens", 20),
                        )
                        if was_training:
                            self.model.train()
                        last_samples = canary
                        last_canary_step = step_idx
                        self.logger.log(f"canary step={step_idx}: " + " | ".join(f"{p} -> {c[:30]}" for p, c in canary))
                    except Exception as e:
                        last_samples = [(CANARY_PROMPTS[0], f"(gen error: {e})")]
                        last_canary_step = step_idx
                        self.logger.log(f"canary ERROR step={step_idx}: {e}")

                if last_batch is not None and step_idx % log_every == 0:
                    with torch.no_grad():
                        tail_ids = last_batch["input_ids"][0, -64:].tolist()
                        try:
                            last_batch_preview = self.tokenizer.decode(tail_ids, skip_special_tokens=True)
                        except Exception:
                            last_batch_preview = "(decode failed)"

                ratios, targets, pm_max, route_score_rms = self._gather_health()
                mem_alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                mem_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
                if torch.cuda.is_available():
                    mem_total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
                    mem_peak = torch.cuda.max_memory_allocated(self.device) / 1e9
                else:
                    mem_total = 0.0
                    mem_peak = 0.0

                elapsed = time.time() - self.run_start_time
                eta = (self.total_steps - step_idx) * (elapsed / max(1, step_idx))

                warn_msg = ""
                if not math.isfinite(loss_mean):
                    warn_msg = "non-finite loss"
                    recent_warns.append(warn_msg)
                if pm_max > 1.0:
                    warn_msg = f"PolyMamba spectral radius {pm_max:.3f} > 1.0"
                    recent_warns.append(warn_msg)
                if grad_norm > self._grad_spike_threshold:
                    warn_msg = f"grad norm {grad_norm:.2f} very large"
                    recent_warns.append(warn_msg)
                warn_msg = "; ".join(list(recent_warns)[-3:])

                snap = TrainSnapshot(
                    step=step_idx,
                    total_steps=self.total_steps,
                    tokens_seen=self.state.tokens_seen,
                    seq_len=seq_len,
                    lr=self.optimizer.param_groups[0]["lr"],
                    loss=loss_mean,
                    task_loss=loss_mean,
                    mod_loss=running_mod,
                    fsa_ortho=running_fsa,
                    polymamba_kbal=running_pm,
                    grad_norm=grad_norm,
                    tokens_per_sec=tokens_per_sec,
                    samples_per_sec=total_accum_tokens / max(step_wall, 1e-9),
                    step_time_ms=(step_time_sum / max(1, len(step_times))) * 1000,
                    elapsed_sec=elapsed,
                    eta_sec=eta,
                    fsa_temperature=fsa_temp,
                    pm_max_spectral_radius=pm_max,
                    mod_selection_ratios=ratios,
                    mod_selection_targets=targets,
                    last_samples=last_samples,
                    last_canary_step=last_canary_step,
                    last_batch_preview=last_batch_preview,
                    last_eval=last_eval,
                    memory_alloc_gb=mem_alloc,
                    memory_reserved_gb=mem_reserved,
                    memory_total_gb=mem_total,
                    memory_peak_gb=mem_peak,
                    warn=warn_msg,
                )
                if dashboard is not None:
                    dashboard.update(snap)
                    dashboard.refresh()
                snapshot_handler(snap)

                if eval_every > 0 and step_idx % eval_every == 0:
                    self.logger.log(f"eval step={step_idx}")
                    last_eval = self._run_periodic_eval(step_idx)
                    snap.last_eval = last_eval
                    if dashboard is not None:
                        dashboard.update(snap)
                        dashboard.refresh()
                    self.logger.log(
                        f"eval result step={step_idx} "
                        f"hellaswag={last_eval['hellaswag_acc']:.2%} "
                        f"arc_easy={last_eval['arc_easy_acc']:.2%}"
                    )

                if save_every > 0 and step_idx % save_every == 0:
                    rolling = output_dir / "rolling.pt"
                    self._save(rolling)
                    self.state.last_checkpoint_step = step_idx
                    self.logger.log(f"rolling checkpoint saved step={step_idx} path={rolling}")

                if step_idx % log_every == 0 or step_idx == 1:
                    self.logger.log(
                        f"step {step_idx} loss={loss_mean:.4f} "
                        f"grad_norm={grad_norm:.3f} "
                        f"tokens_per_sec={tokens_per_sec:.0f} "
                        f"route_rms={route_score_rms:.3f} "
                        f"pm_rho={pm_max:.4f} "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                    )

            self._save(output_dir / "final.pt")
            self.logger.log(f"final checkpoint saved step={step_idx}")

    @property
    def num_parameters(self) -> int:
        return self.model.num_parameters()