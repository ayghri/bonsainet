"""
Copyright (c) 2025 Ayoub Ghriss and contributors
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.
"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, Optional, Literal
import torch

from bonsainet.specs import SpecCoupler
from bonsainet.specs import BlockGroupSpec


Tensor = torch.Tensor
Optimizer = torch.optim.Optimizer


@dataclass
class EMAController:
    """EMA of gradients for s(x) using g <- rho*g + (1-rho)*direction"""

    rho: float
    _ema: Dict[BlockGroupSpec, Tensor] = field(
        default_factory=dict, init=False, repr=False
    )

    def get(self, spec):
        return self._ema[spec]

    @torch.no_grad()
    def update_all(self, directions: Dict[BlockGroupSpec, Tensor]):
        for p, direct in directions.items():
            self.update_single(p, direct)

    @torch.no_grad()
    def update_single(self, spec: BlockGroupSpec, direction: Optional[Tensor]):
        if direction is None:
            return
        if spec not in self._ema:
            self._ema[spec] = torch.zeros_like(spec.param.data)
        if self.rho == 0:
            self._ema[spec] = direction
            return
        self._ema[spec].mul_(self.rho).add_(direction, alpha=1 - self.rho)


@dataclass
class AlphaController:
    default: float
    _alphas: Dict[BlockGroupSpec, Tensor] = field(
        default_factory=dict, init=False
    )

    def get(self, spec):
        return self._alphas.get(spec, self.default)

    def set(self, spec, alpha):
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        self._alphas[spec] = alpha.to(spec.param)


@dataclass
class LambdaController:
    device: torch.device
    beta: float = 1.0
    mode: Literal["constant", "RM"] = "constant"
    gamma: float = 0.75
    t0: int = 100
    cap: float = 10.0

    _momentums: Dict[SpecCoupler, Tensor] = field(init=False, repr=False)
    _t: Dict[SpecCoupler, int] = field(init=False, repr=False)

    def get(self, group: SpecCoupler):
        return self._momentums[group]

    def __post_init__(self):
        assert self.t0 > 0
        self._momentums = {}
        self._t = defaultdict(lambda: self.t0)

    def beta_t(self, t: int) -> float:
        if self.mode == "constant":
            return self.beta
        return self.beta / ((t + self.t0) ** self.gamma)

    def reset_time(self):
        for p, t in self._t.items():
            self._t[p] = 0

    def update_all(self, group_psi: Dict[SpecCoupler, Tensor]):
        """Update lambda via a fixed-point iteration towards a target score s_hat."""
        for p, psi in group_psi.items():
            self.update_single(p, psi)

    def update_single(self, group: SpecCoupler, psi: Tensor):
        """Update lambda via a fixed-point iteration towards a target score s_hat."""
        psi = psi.to(self.device)
        b = self.beta_t(self._t[group])
        self._t[group] += 1
        if group not in self._momentums:
            self._momentums[group] = torch.zeros_like(psi, device=self.device)
        self._momentums[group].mul_(1 - b).add_(psi, alpha=b)
        self._momentums[group].clamp_(max=self.cap)
