from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether.utils.numerics import to_fp32


class PolyMamba2(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        n_orders: int = 2,
        state_dim: int = 16,
        max_radius: float = 0.95,
    ) -> None:
        super().__init__()
        if n_orders not in (2, 4):
            raise ValueError(f"n_orders must be 2 or 4, got {n_orders}")

        self.d_model = d_model
        self.d_inner = d_inner
        self.n_orders = n_orders
        self.state_dim = state_dim
        self.max_radius = max_radius

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.gate = nn.Linear(d_model, n_orders, bias=False)

        self.A_log = nn.Parameter(torch.zeros(n_orders, state_dim))
        self.B = nn.Parameter(torch.zeros(n_orders, state_dim, d_inner))
        self.C = nn.Parameter(torch.zeros(n_orders, d_inner, state_dim))
        self.D = nn.Parameter(torch.zeros(n_orders, d_inner))

        self.structure_M = self._init_structure_matrices()
        self.dt_floor, self.dt_ceil = self._compute_dt_bounds()
        self._init_parameters()

        self._balance_loss: torch.Tensor = torch.zeros(())

    def _init_structure_matrices(self) -> nn.ParameterList:
        N = self.state_dim
        matrices = []
        for order in range(self.n_orders):
            if order == 0:
                M = torch.eye(N)
            else:
                M = torch.zeros(N, N)
                for i in range(N):
                    M[i, i] = 1.0
                    if i + 1 < N:
                        w = 0.4
                        M[i, i + 1] = w
                        M[i + 1, i] = w
            M = M / (torch.linalg.eigvalsh(M).abs().max() * 1.10 + 1e-8)
            matrices.append(nn.Parameter(M, requires_grad=False))
        return nn.ParameterList(matrices)

    def _compute_dt_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-order (floor, ceil) for dt that keep rho(A) <= max_radius.

        A = I - dt*M has eigenvalues 1 - dt*lambda_i(M), so capping the spectral
        radius needs BOTH ends: dt >= (1 - max_radius)/lambda_min for the small
        eigenvalue and dt <= (1 + max_radius)/lambda_max for the large one. Only
        the lower bound existed, which left rho(A) free to exceed 1 whenever dt
        grew past 2/lambda_max.
        """
        floors = []
        ceils = []
        for k in range(self.n_orders):
            eig = torch.linalg.eigvalsh(self.structure_M[k].detach().float())
            floors.append((1.0 - self.max_radius) / eig.min().clamp_min(1e-4))
            ceils.append((1.0 + self.max_radius) / eig.abs().max().clamp_min(1e-4))
        return torch.stack(floors), torch.stack(ceils)

    def _init_parameters(self) -> None:
        lo = self.dt_floor.clamp_min(0.005)
        hi = torch.minimum(self.dt_ceil, (lo * 4.0).clamp_min(lo * 1.001))
        dt = torch.exp(
            torch.rand(self.n_orders, self.state_dim) * (hi.log() - lo.log()).unsqueeze(-1)
            + lo.log().unsqueeze(-1)
        )
        with torch.no_grad():
            self.A_log.copy_(torch.log(torch.expm1(dt)))

        nn.init.normal_(self.B, mean=0.0, std=1.0 / math.sqrt(self.d_inner))
        nn.init.normal_(self.C, mean=0.0, std=1.0 / math.sqrt(self.d_inner))
        nn.init.zeros_(self.D)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def _A(self) -> torch.Tensor:
        dt = F.softplus(self.A_log)
        floor = self.dt_floor.to(dt.device).unsqueeze(-1)
        ceil = self.dt_ceil.to(dt.device).unsqueeze(-1)
        dt = dt.clamp_min(floor).clamp_max(ceil)
        A = torch.zeros(self.n_orders, self.state_dim, self.state_dim, device=self.A_log.device)
        for k in range(self.n_orders):
            A[k] = torch.eye(self.state_dim, device=self.A_log.device) - dt[k].unsqueeze(-1) * self.structure_M[k]
        return A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        A = self._A()
        gate_logits = self.gate(x)
        gate = F.softmax(gate_logits.float(), dim=-1).to(x.dtype)

        p = gate.float().mean(dim=(0, 1))
        entropy = -(p * torch.log(p.clamp_min(1e-4))).sum()
        margin = 0.3
        max_entropy = math.log(self.n_orders)
        self._balance_loss = F.relu(torch.full_like(entropy, max_entropy) - entropy - margin)

        x_proj = self.in_proj(x)
        x_val, x_gate = x_proj.chunk(2, dim=-1)
        x_inner = F.silu(x_gate) * x_val

        x_inner_f32 = x_inner.float()
        outputs = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            for k in range(self.n_orders):
                A_k = A[k].to(torch.float32)
                B_k = self.B[k].to(torch.float32)
                C_k = self.C[k].to(torch.float32)
                D_k = self.D[k].to(torch.float32)

                u = torch.einsum("bld,sd->bsl", x_inner_f32, B_k)
                h = _parallel_ssm_scan(A_k, u)
                y_seq = torch.einsum("bls,ds->bld", h, C_k) + x_inner_f32 * D_k
                outputs.append(y_seq.to(x.dtype))

        stacked = torch.stack(outputs, dim=-2)
        gate_expanded = gate.unsqueeze(-1)
        y = (stacked * gate_expanded).sum(dim=-2)
        return self.out_proj(y)

    def balance_loss(self) -> torch.Tensor:
        return self._balance_loss

    def spectral_radius(self) -> torch.Tensor:
        A = self._A().detach().to(torch.float32).cpu()
        radii = torch.linalg.eigvals(A).abs().max(dim=-1).values
        return radii


def _parallel_ssm_scan(A: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Doubling scan for the linear SSM h_t = A h_{t-1} + u_t.

    Because the transition matrix is constant across time, the block scan
    identity sum_{d<2^i} A^d u_{t-d} = h_t^{(i)} + A^{2^i} h_{t-2^i}^{(i)}
    only ever needs the (N, N) transition powers, never a per-position
    (bsz, seq_len, N, N) product tensor. That keeps the scan to a handful of
    small vector ops per doubling step instead of ~10 full-size tensor ops,
    and avoids materialising the per-position transition products.

    A: (N, N)   u: (bsz, N, seq_len)   -> h: (bsz, seq_len, N)
    """
    seq_len = u.shape[-1]
    h = u.transpose(-1, -2)
    power_t = A.transpose(-1, -2)
    d = 1
    while d < seq_len:
        shifted = F.pad(h, (0, 0, d, 0))[:, :seq_len]
        h = h + torch.matmul(shifted, power_t)
        power_t = torch.matmul(power_t, power_t)
        d *= 2
    return h
