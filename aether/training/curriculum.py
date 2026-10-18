from __future__ import annotations


class CurriculumScheduler:
    DEFAULT_PHASES: list[tuple[float, float, int]] = [
        (0.0, 0.20, 1024),
        (0.20, 0.50, 2048),
        (0.50, 1.00, 4096),
    ]

    def __init__(
        self,
        total_steps: int,
        phases: tuple[tuple[float, float, int], ...] | None = None,
    ) -> None:
        self.total_steps = total_steps
        self._schedule: list[tuple[int, int, int]] = []
        phases_to_use = phases if phases else self.DEFAULT_PHASES
        for start_frac, end_frac, seq_len in phases_to_use:
            start_step = int(start_frac * total_steps)
            end_step = int(end_frac * total_steps)
            self._schedule.append((start_step, end_step, seq_len))

    def seq_len_for_step(self, step: int) -> int:
        for start_step, end_step, seq_len in self._schedule:
            if start_step <= step < end_step:
                return seq_len
        return self._schedule[-1][2]

    def all_seq_lens(self) -> list[int]:
        return sorted({seq_len for _, _, seq_len in self._schedule})


def linear_ramp_weight(step: int, total_steps: int, warmup_frac: float) -> float:
    if warmup_frac <= 0.0:
        return 1.0
    ramp_steps = max(1, int(warmup_frac * total_steps))
    return min(1.0, step / ramp_steps)


def fsa_temperature(step: int, total_steps: int, warmup_frac: float = 0.10) -> float:
    if warmup_frac <= 0.0:
        return 0.1
    ramp_steps = max(1, int(warmup_frac * total_steps))
    t = min(1.0, step / ramp_steps)
    return 1.0 - 0.9 * t


def fsa_gumbel_noise(step: int, total_steps: int, warmup_frac: float = 0.10) -> float:
    if warmup_frac <= 0.0:
        return 0.0
    ramp_steps = max(1, int(warmup_frac * total_steps))
    t = min(1.0, step / ramp_steps)
    return 1.0 - t


def aux_loss_weights(
    step: int,
    total_steps: int,
    fsa_weight: float,
    fsa_warmup: float,
    polymamba_weight: float,
    polymamba_warmup: float,
    mod_weight: float,
    mod_warmup: float,
) -> dict[str, float]:
    return {
        "fsa_ortho": fsa_weight * linear_ramp_weight(step, total_steps, fsa_warmup),
        "polymamba_kbal": polymamba_weight * linear_ramp_weight(step, total_steps, polymamba_warmup),
        "mod": mod_weight * linear_ramp_weight(step, total_steps, mod_warmup),
    }