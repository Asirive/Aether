from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


@dataclass
class TrainSnapshot:
    step: int = 0
    total_steps: int = 0
    tokens_seen: int = 0
    seq_len: int = 0
    lr: float = 0.0
    loss: float = 0.0
    task_loss: float = 0.0
    mod_loss: float = 0.0
    fsa_ortho: float = 0.0
    polymamba_kbal: float = 0.0
    grad_norm: float = 0.0
    tokens_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    step_time_ms: float = 0.0
    elapsed_sec: float = 0.0
    eta_sec: float = 0.0
    fsa_temperature: float = 0.1
    pm_max_spectral_radius: float = 0.0
    mod_selection_ratios: dict[int, float] = field(default_factory=dict)
    mod_selection_targets: dict[int, float] = field(default_factory=dict)
    last_samples: list[tuple[str, str]] = field(default_factory=list)
    last_canary_step: int = 0
    last_batch_preview: str = ""
    last_eval: Optional[dict] = None
    memory_alloc_gb: float = 0.0
    memory_reserved_gb: float = 0.0
    memory_total_gb: float = 0.0
    memory_peak_gb: float = 0.0
    warn: str = ""


class TrainingDashboard:
    def __init__(
        self,
        total_steps: int,
        loss_window: int = 200,
        refresh_per_second: int = 4,
        console: Optional[Console] = None,
    ) -> None:
        self.total_steps = total_steps
        self.loss_history: deque[float] = deque(maxlen=loss_window)
        self.tokens_history: deque[int] = deque(maxlen=loss_window)
        self.refresh_per_second = refresh_per_second
        self.console = console or Console()
        self.snapshot = TrainSnapshot(total_steps=total_steps)
        self.start_time = time.time()

    def update(self, snap: TrainSnapshot) -> None:
        if snap.loss == snap.loss:
            self.loss_history.append(snap.loss)
        self.tokens_history.append(snap.tokens_per_sec)
        self.snapshot = snap

    def _header(self) -> RenderableType:
        s = self.snapshot
        lr_str = f"{s.lr:.2e}"
        sps_str = f"{s.tokens_per_sec:,.0f}"
        eta_str = self._fmt_seconds(s.eta_sec)
        head = Text()
        head.append("AETHER-200M PRETRAIN", style="bold magenta")
        head.append("    ")
        head.append(f"step {s.step:,}/{s.total_steps:,}", style="bold white")
        head.append("    ")
        head.append(f"lr={lr_str}", style="cyan")
        head.append("    ")
        head.append(f"seq_len={s.seq_len}", style="cyan")
        head.append("    ")
        head.append(f"tps={sps_str}", style="green")
        head.append("    ")
        head.append(f"vram={s.memory_reserved_gb:.1f}/{s.memory_total_gb:.1f}GB", style="yellow")
        head.append("    ")
        head.append(f"eta={eta_str}", style="yellow")
        return head

    def _progress(self) -> RenderableType:
        s = self.snapshot
        pct = s.step / max(1, s.total_steps)
        prog = Progress(
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(bar_width=None, complete_style="green", finished_style="bold green"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TextColumn("->"),
            TimeRemainingColumn(),
            expand=True,
        )
        prog.add_task("train", total=s.total_steps, completed=s.step)
        return prog

    @staticmethod
    def _fmt_seconds(s: float) -> str:
        if not s or s < 0 or s != s:
            return "--"
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        if h > 0:
            return f"{h}h{m}m"
        if m > 0:
            return f"{m}m{sec}s"
        return f"{sec}s"

    @staticmethod
    def _sparkline(values: list[float], width: int = 40, lo: float | None = None, hi: float | None = None) -> str:
        if not values:
            return " " * width
        if lo is None:
            lo = min(values)
        if hi is None:
            hi = max(values)
        rng = hi - lo if hi > lo else 1.0
        bars = " ▁▂▃▄▅▆▇█"
        if len(values) > width:
            step = len(values) / width
            sampled = [values[int(i * step)] for i in range(width)]
        else:
            sampled = list(values) + [values[-1]] * (width - len(values))
        out = []
        for v in sampled:
            idx = int(((v - lo) / rng) * (len(bars) - 1))
            idx = max(0, min(len(bars) - 1, idx))
            out.append(bars[idx])
        return "".join(out)

    def _loss_panel(self) -> RenderableType:
        s = self.snapshot
        tbl = Table.grid(expand=True)
        tbl.add_column(justify="left")
        tbl.add_column(justify="right")
        spark = self._sparkline(list(self.loss_history), width=40)
        tbl.add_row(
            f"train loss (last {len(self.loss_history)}): [bold]{s.loss:.4f}[/bold]",
            spark,
        )
        tbl.add_row(f"task loss:        [cyan]{s.task_loss:.4f}[/cyan]", "")
        tbl.add_row(
            f"mod bal loss:     [yellow]{s.mod_loss:.4f}[/yellow]",
            f"fsa temp: [magenta]{s.fsa_temperature:.3f}[/magenta]",
        )
        tbl.add_row(
            f"fsa ortho loss:   [yellow]{s.fsa_ortho:.4f}[/yellow]",
            f"pm kbal loss: [yellow]{s.polymamba_kbal:.4f}[/yellow]",
        )
        tbl.add_row(
            f"grad norm:        [red]{s.grad_norm:.3f}[/red]",
            f"step time: [dim]{s.step_time_ms:.0f}ms[/dim]",
        )
        tbl.add_row(
            f"vram: [yellow]{s.memory_reserved_gb:.1f}/{s.memory_total_gb:.1f}GB[/yellow] "
            f"([dim]active {s.memory_alloc_gb:.1f}, peak {s.memory_peak_gb:.1f}[/dim])",
            "",
        )
        if s.warn:
            tbl.add_row(f"[bold red]WARN:[/bold red] {s.warn}", "")
        return Panel(tbl, title="Train", border_style="green")

    def _health_panel(self) -> RenderableType:
        s = self.snapshot
        tbl = Table.grid(expand=True)
        tbl.add_column(justify="left")
        tbl.add_column(justify="right")
        warn_pm = " [bold red]!! > 1.0[/bold red]" if s.pm_max_spectral_radius > 1.0 else ""
        tbl.add_row(f"PolyMamba max spectral radius: [bold]{s.pm_max_spectral_radius:.4f}[/bold]{warn_pm}", "")
        if s.mod_selection_ratios:
            lines = []
            for k, v in sorted(s.mod_selection_ratios.items()):
                target = s.mod_selection_targets.get(k, 0.0)
                dev = abs(v - target)
                color = "red" if dev > 0.1 else "green"
                lines.append(f"layer {k:>2}: kept={v:.2f}  target={target:.2f}")
            tbl.add_row("MoD route ratios:", "\n".join(lines))
        else:
            tbl.add_row("MoD route ratios: [dim](none yet)[/dim]", "")
        return Panel(tbl, title="Component Health", border_style="cyan")

    def _samples_panel(self) -> RenderableType:
        s = self.snapshot
        if not s.last_samples and not s.last_batch_preview:
            body = Text("waiting for first batch...", style="dim italic")
            return Panel(body, title="Model Output Monitor", border_style="magenta")
        body = Text()
        if s.last_batch_preview:
            body.append("Training data (from batch):\n", style="dim italic")
            preview = s.last_batch_preview
            if len(preview) > 300:
                preview = preview[:297] + "..."
            body.append(preview + "\n\n", style="dim")
        if s.last_samples:
            body.append(f"Canary generation (step {s.last_canary_step}):\n\n", style="dim italic")
            for prompt, continuation in s.last_samples:
                body.append(prompt, style="bold cyan")
                body.append(" → ", style="dim")
                cont = continuation if len(continuation) <= 80 else continuation[:77] + "..."
                if cont.strip():
                    body.append(cont, style="white")
                else:
                    body.append("(empty)", style="dim italic")
                body.append("\n\n")
        return Panel(body, title="Model Output Monitor", border_style="magenta")

    def _eval_panel(self) -> RenderableType:
        s = self.snapshot
        if not s.last_eval:
            body = Text("(no eval yet)", style="dim italic")
            return Panel(body, title="Q/A Eval", border_style="blue")
        ev = s.last_eval
        lines: list[str] = []
        lines.append(
            f"step {ev.get('step', '?')}  hellaswag {ev.get('hellaswag_acc', 0):.2%}  "
            f"arc_easy {ev.get('arc_easy_acc', 0):.2%}  ppl {ev.get('ppl', 0):.2f}"
        )
        for task in ("hellaswag", "arc_easy"):
            details = ev.get(f"{task}_details", []) or []
            for d in details[:4]:
                status = "✓" if d.get("ok") else "✗"
                ctx = d.get("ctx", "")[:80]
                pred = d.get("choices", ["", "", "", ""])[d.get("pred", 0)][:60]
                gold = d.get("choices", ["", "", "", ""])[d.get("label", 0)][:60]
                lines.append(f"  {status} {ctx} → pred: {pred} | gold: {gold}")
        return Panel(Text("\n".join(lines)), title="Q/A Eval", border_style="blue")

    def render(self) -> RenderableType:
        return Group(
            self._header(),
            self._progress(),
            self._loss_panel(),
            self._samples_panel(),
            self._eval_panel(),
            self._health_panel(),
        )

    def __enter__(self) -> "TrainingDashboard":
        self._live = Live(
            self.render(),
            console=self.console,
            refresh_per_second=self.refresh_per_second,
            transient=False,
            screen=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._live.__exit__(exc_type, exc_val, exc_tb)

    def refresh(self) -> None:
        self._live.update(self.render())