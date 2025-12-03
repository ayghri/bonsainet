"""
Copyright (c) 2025 Ayoub Ghriss and contributors
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Union
import torch
import torch.nn as nn

from abc import abstractmethod

from bonsainet.specs import SpecCoupler
from bonsainet.controllers import EMAController
from bonsainet.controllers import LambdaController
from bonsainet.controllers import AlphaController
from bonsainet.specs import BlockGroupSpec

Tensor = torch.Tensor
Parameter = nn.Parameter
Optimizer = torch.optim.Optimizer

# Optimizer Interface (Strategy Pattern)


class OptimizerProxy:
    """
    Abstracts the differences between SGD, Adam, etc.
    Extracts:
        - direction: The momentum/gradient used for the EMA controller.
        - learning rate
        - conditioner: when applicable
    """

    @staticmethod
    def get_proxy(optimizer: Optimizer) -> "OptimizerProxy":
        name = optimizer.__class__.__name__
        if name == "Adam":
            return AdamProxy(optimizer)
        elif name == "SGD":
            return SGDProxy(optimizer)
        else:
            raise ValueError(
                f"Optimizer {name} not explicitly supported. Assuming SGD-like behavior."
            )

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def get_info(
        self,
        param: Parameter,
        #   group_cfg: Dict[str, Any]
    ) -> Tuple[Tensor, Union[float, Tensor], Optional[Tensor]]:
        """Returns (momentum_buffer, effective_step_size)"""
        raise NotImplementedError


class SGDProxy(OptimizerProxy):
    @abstractmethod
    def get_info(
        self,
        param: Parameter,
        # group_cfg: Dict[str, Any],
    ) -> Tuple[Tensor, Union[float, Tensor], Optional[Tensor]]:
        # pass
        state = self.optimizer.state.get(param, {})
        if "momentum_buffer" in state:
            momentum = state["momentum_buffer"]
        else:
            momentum = param.grad

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if id(p) == id(param):
                    lr = group["lr"]

        return momentum, lr, None


class AdamProxy(OptimizerProxy):
    def get_info(
        self,
        param: Parameter,
        # group_cfg: Dict[str, Any],
    ) -> Tuple[Tensor, Union[float, Tensor], Optional[Tensor]]:
        beta1 = None
        beta2 = 0.0
        eps = 1e-12
        lr = 0.001
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if id(p) == id(param):
                    beta1, beta2 = group["betas"]
                    eps = group["eps"]
                    lr = group["lr"]

                    break
        assert beta1 is not None, (
            f"param with id{id(param)} not found in optimizer "
        )

        state = self.optimizer.state[param]
        step = state.get("step", 1)
        # Direction: Exp Avg (First Moment)
        if isinstance(step, Tensor):
            step = step.item()

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        momentum = state["exp_avg"] / bias_correction1
        denom = (state["exp_avg_sq"].sqrt() / (bias_correction2**0.5)).add(eps)

        return momentum, lr, denom


class AdamWProxy(OptimizerProxy):
    def get_info(
        self,
        param: Parameter,
        # group_cfg: Dict[str, Any],
    ) -> Tuple[Tensor, Union[float, Tensor], Optional[Tensor]]:
        beta1 = None
        beta2 = 0.0
        weight_decay = 0.0
        eps = 1e-8
        lr = 0.001
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if id(p) == id(param):
                    beta1, beta2 = group["betas"]
                    eps = group["eps"]
                    lr = group["lr"]
                    weight_decay = group["weight_decay"]

                    break
        assert beta1 is not None, (
            f"param with id{id(param)} not found in optimizer "
        )

        state = self.optimizer.state[param]
        step = state.get("step", 1)
        # Direction: Exp Avg (First Moment)
        if isinstance(step, Tensor):
            step = step.item()

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        momentum = state["exp_avg"] / bias_correction1
        denom = (state["exp_avg_sq"].sqrt() / (bias_correction2**0.5)).add(eps)
        if weight_decay != 0.0:
            momentum = momentum + weight_decay * param.data * denom

        return momentum, lr, denom


@dataclass
class ASTRASparsifier:
    groups: List[SpecCoupler]
    lambdas: LambdaController
    ema_grad: EMAController
    alphas: AlphaController
    optimizer: Optimizer
    eps: float = 1e-7

    _proxy: OptimizerProxy = field(init=False)

    def __post_init__(self):
        self._proxy = OptimizerProxy.get_proxy(self.optimizer)
        # Map params to specs for O(1) lookup
        self._param_to_spec = {}
        for g in self.groups:
            for s in g.specs:
                self._param_to_spec[s.param] = s

    @property
    def specs(self):
        return [s for g in self.groups for s in g.specs]

    @torch.no_grad()
    def step(self, sparsify: bool = True):
        """
        Call this AFTER optimizer.step().
        1. Updates EMA of gradients (using optimizer state).
        2. Computes Gradient Bar (Score).
        3. Updates Lambda (Threshold).
        4. Applies Soft Thresholding.
        """

        # 1. Gather Data & Update EMA
        # We iterate per parameter group to respect LR schedules
        param_updates: Dict[BlockGroupSpec, Dict] = {}

        for group_cfg in self.optimizer.param_groups:
            for p in group_cfg["params"]:
                if p not in self._param_to_spec:
                    continue

                spec = self._param_to_spec[p]

                # Get optimizer specific stats
                direction, step_size, conditioner = self._proxy.get_info(p)

                # Update EMA Controller
                self.ema_grad.update_single(spec, direction)

                param_updates[spec] = {
                    "step_size": step_size,
                    "ema": self.ema_grad.get(spec),
                    "conditioner": conditioner,
                }

        if not sparsify:
            return

        # Compute Scores (Grad Bar) & Update Lambdas
        for group in self.groups:
            grad_bar_values = {}
            step_sizes = {}

            for sp in group.specs:
                if sp not in param_updates:
                    continue

                data = param_updates[sp]
                alpha = self.alphas.get(sp)

                # Score = EMA_Grad - Alpha * Weights
                # This represents the "force" pushing the weight away from zero
                v = data["ema"] - alpha * sp.param.data
                grad_bar_values[sp] = v
                step_sizes[sp.param] = data["step_size"]

            # Calculate Threshold (Lambda)
            # The GroupCoupler handles the logic of "Coupled" vs "Independent"
            # inside kth_largest via the reduction logic we discussed previously.
            kappa = group.kappa

            # Returns the target threshold value (psi)
            psi = group.kth_largest(grad_bar_values, kappa)

            # Update the controller (Integral control / Momentum on lambda)
            self.lambdas.update_single(group, psi)

            # 4. Apply Proximal Step (Soft Thresholding)
            # threshold = lambda + eps
            current_lambda = self.lambdas.get(group).add(self.eps)

            # The GroupCoupler distributes this threshold back to the specs
            # If coupled, it uses the joint scores; if independent, individual scores.
            group.soft_threshold(current_lambda, learning_rates=step_sizes)


@dataclass
class IHTSparsifier:
    groups: List[SpecCoupler]
    optimizer: Optional[Optimizer]

    _masks: Dict[Parameter, Tensor] = field(default_factory=dict, init=False)
    _hooks: List = field(default_factory=list, init=False)

    @torch.no_grad()
    def step(self):
        """
        Performs Hard Thresholding (Projection).
        Call after optimizer.step().
        """
        for group in self.groups:
            # This relies on GroupCoupler.hard_threshold()
            # which calculates norms, finds top-k, and masks weights.
            group.hard_threshold()

    def freeze_support(self):
        """
        Locks the current sparsity pattern.
        1. Hard thresholds one last time.
        2. Registers hooks to zero out gradients for pruned weights.
        """
        self.step()  # Final prune

        self._masks.clear()

        for group in self.groups:
            for sp in group.specs:
                p = sp.param
                # Create binary mask
                mask = (p.data.abs() > 0).float()
                self._masks[p] = mask

                # Zero out current data (cleanup)
                p.data.mul_(mask)

                # Register hook to keep it zeroed
                h = p.register_post_accumulate_grad_hook(
                    lambda p, m=mask: p.grad.mul_(m)
                    if p.grad is not None
                    else None
                )
                self._hooks.append(h)

        # Clean optimizer state (Momentum) for pruned weights
        if self.optimizer:
            self._clean_optimizer_state()

    def _clean_optimizer_state(self):
        """Zeros out momentum buffers for pruned weights."""
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p in self._masks:
                        state = self.optimizer.state.get(p, {})
                        for key in ["momentum_buffer", "exp_avg", "exp_avg_sq"]:
                            if key in state and state[key] is not None:
                                state[key].mul_(self._masks[p])

    def unfreeze(self):
        """Removes hooks to allow dense training again."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._masks.clear()


# @dataclass
# class IHTSparsifier:
#     groups: List[GroupCoupler]
#     kappa: Optional[int] = None
#     sparsity: Optional[float] = None
#     device: Optional[torch.device] = None

#     _masks: dict = field(default_factory=dict, init=False, repr=False)

#     _hook_handles: List = field(default_factory=list, init=False, repr=False)

#     def step(self, sparsify=True, *args, **kwargs):
#         if not sparsify:
#             return
#         for group in self.groups:
#             group.hard_threshold()

#     def freeze_support(self, optimizer: Optimizer):
#         # hard threshold
#         self.step()
#         # get the masks
#         for group in self.groups:
#             for sp in group.specs:
#                 p = sp.param
#                 mask = p.data.abs() >= 1e-12
#                 p.data.mul_(mask.to(p))
#                 self._masks[sp.param] = mask
#                 # register hooks to zero-out gradients after each backward
#                 self._hook_handles.append(
#                     p.register_post_accumulate_grad_hook(
#                         # m=mask to avoid all params using the same mask... 🤬
#                         lambda p, m=mask: hooks.mask_post_accumulate_hook(
#                             p, m.to(p)
#                         )
#                     )
#                 )

#         if isinstance(optimizer, torch.optim.SGD):
#             # zero out the momentum outside the support
#             for param_g in optimizer.param_groups:
#                 if param_g["momentum"] == 0.0:
#                     continue
#                 for p in param_g["params"]:
#                     if p in self._masks:
#                         st = optimizer.state.get(p, {})
#                         momentum = st.get("momentum_buffer", None)
#                         if momentum is not None:
#                             momentum.mul_(self._masks[p].to(momentum))

#         else:
#             raise ValueError(
#                 "IHT.freeze_support only supports SGD (for now), "
#                 f"got {optimizer.__class__}"
#             )


# @dataclass
# class ASTRAOperator:
#     groups: List[GroupCoupler]
#     lambdas: LambdaController
#     ema_grad: EMAController
#     alphas: AlphaController
#     device: Optional[torch.device] = None
#     eps: float = 1e-7

#     _cached_directions: Dict[Parameter, Tensor] = field(
#         default_factory=dict, init=False, repr=False
#     )
#     _cached_learning_rates: Dict[Parameter, Tensor] = field(
#         default_factory=dict, init=False, repr=False
#     )
#     _hook_handles: Dict[Optimizer, list] = field(
#         default_factory=dict, init=False, repr=False
#     )

#     _updated = False

#     @property
#     def params(self) -> set[Parameter]:
#         ps = set()
#         for group in self.groups:
#             for p in group.params:
#                 ps.add(p)
#         return ps

#     @property
#     def specs(self):
#         ps = set()
#         for group in self.groups:
#             for s in group.specs:
#                 ps.add(s)
#         return ps

#     def _pre_step_hook(self, optimizer, *args, **kwargs):
#         self._cached_directions.update(self.gather_gradients(optimizer))
#         self._updated = True

#     def _post_step_hook(self, optimizer, *args, **kwargs):
#         params_for_opt = {
#             p for group in optimizer.param_groups for p in group["params"]
#         }
#         directions_for_opt = {
#             p: d
#             for p, d in self._cached_directions.items()
#             if p in params_for_opt
#         }
#         new_directions, new_lrs = self.gather_info(
#             directions_for_opt, optimizer
#         )
#         self._cached_directions.update(new_directions)
#         self._cached_learning_rates.update(new_lrs)
#         self._updated = True

#     def attach_optimizer(self, optimizer: Optimizer):
#         if optimizer in self._hook_handles:
#             warnings.warn("Optimizer already attached to ASTRA.")
#             return self
#         pre_hook = optimizer.register_step_pre_hook(self._pre_step_hook)
#         post_hook = optimizer.register_step_post_hook(self._post_step_hook)
#         self._hook_handles[optimizer] = [pre_hook, post_hook]
#         return self

#     def detach_optimizer(self, optimizer: Optimizer):
#         if optimizer not in self._hook_handles:
#             warnings.warn("Optimizer was not attached to ASTRA before")
#             return
#         for handle in self._hook_handles.pop(optimizer):
#             handle.remove()
#         return self

#     def detach_all_optimizers(self):
#         for optimizer in list(self._hook_handles.keys()):
#             self.detach_optimizer(optimizer)

#     @torch.no_grad()
#     def step(self, sparsify: bool = True):
#         if not self._updated:
#             print("Sparsifier states not updated. did you attach optimizer?")
#         if not self._cached_directions:
#             return

#         directions = self._cached_directions
#         learning_rates = self._cached_learning_rates
#         self._cached_directions = {}
#         self._cached_learning_rates = {}

#         for sp in self.specs:
#             self.ema_grad.update_single(sp, directions.get(sp.param, None))

#         for group in self.groups:
#             grad_bar_values = {}
#             lrs = {}
#             for sp in group.specs:
#                 lrs[sp] = learning_rates[sp.param]
#                 alpha = self.alphas.get(sp)
#                 ema = self.ema_grad.get(sp)
#                 v = ema - alpha * sp.param.data
#                 grad_bar_values[sp] = v

#             kappa = group.kappa
#             psi = group.kth_largest(grad_bar_values, kappa)
#             self.lambdas.update_single(group, psi)
#             threshold = self.lambdas.get(group).add(self.eps)
#             if sparsify:
#                 group.soft_threshold(threshold, learning_rates=lrs)

#         self._updated = False

#     def gather_gradients(self, optimizer: Optimizer) -> Dict[Parameter, Tensor]:
#         directions = {}
#         param_set = {s.param for g in self.groups for s in g.specs}
#         for group in optimizer.param_groups:
#             for p in group["params"]:
#                 if p not in param_set or p.grad is None:
#                     continue
#                 directions[p] = p.grad.detach()
#         return directions

#     def gather_info(
#         self, directions: Dict[Parameter, Tensor], optimizer: Optimizer
#     ) -> Tuple[Dict[Parameter, Tensor], Dict[Parameter, Tensor]]:
#         raise NotImplementedError(
#             "Subclasses must implement gather_update_info()"
#         )


# class SASTRA(BaseASTRASparsifier):
#     """ASTRA sparsifier specialized for SGD (with/without momentum).

#     - Direction per param: momentum_buffer if available (when use_momentum), else raw grad.
#     - Learning rate per param: scalar base lr from the param group.
#     """

#     def gather_info(
#         self, directions: Dict[Parameter, Tensor], optimizer: Optimizer
#     ) -> Tuple[Dict[Parameter, Tensor], Dict[Parameter, Tensor]]:
#         lrs: Dict[Parameter, Tensor] = {}
#         param_set = self.params

#         for param_g in optimizer.param_groups:
#             base_lr = param_g.get("lr", 1.0)
#             for p in param_g["params"]:
#                 if p not in param_set:
#                     continue
#                 st = optimizer.state.get(p, {})
#                 momentum = st.get("momentum_buffer", None)
#                 if momentum is not None:
#                     directions[p] = st["momentum_buffer"].detach()
#                 lrs[p] = torch.as_tensor(
#                     base_lr, device=p.device, dtype=p.dtype
#                 )

#         return directions, lrs


# class AdASTRA(BaseASTRASparsifier):
#     """ASTRA sparsifier specialized for Adam/AdamW.

#     - Direction per param: exp_avg (first moment EMA).
#     - Learning rate per param: elementwise step_size/(sqrt(exp_avg_sq)+eps) with bias correction.
#     """

#     def gather_info(
#         self, directions: Dict[Parameter, Tensor], optimizer: Optimizer
#     ) -> Tuple[Dict[Parameter, Tensor], Dict[Parameter, Tensor]]:
#         lrs: Dict[Parameter, Tensor] = {}
#         param_set = {s.param for g in self.groups for s in g.specs}

#         for group in optimizer.param_groups:
#             base_lr = group.get("lr", 1.0)
#             betas = group.get("betas", (0.9, 0.999))
#             eps = group.get("eps", 1e-8)
#             for p in group["params"]:
#                 if p not in param_set:
#                     continue
#                 st = optimizer.state.get(p, {})

#                 exp_avg = st.get("exp_avg", None)
#                 if exp_avg is not None:
#                     directions[p] = exp_avg.detach()

#                 # Elementwise learning rate tensor
#                 exp_avg_sq = st.get("exp_avg_sq", None)
#                 step_val = st.get("step", 0)
#                 step_t = (
#                     int(step_val.item())
#                     if isinstance(step_val, torch.Tensor)
#                     else int(step_val)
#                 )
#                 step_t = max(1, step_t)
#                 b1, b2 = betas
#                 bc1 = 1 - (b1**step_t)
#                 bc2 = 1 - (b2**step_t)
#                 base_lr_t = torch.as_tensor(
#                     base_lr, device=p.device, dtype=p.dtype
#                 )
#                 if exp_avg_sq is not None:
#                     denom = exp_avg_sq.detach().sqrt().add_(eps)
#                     step_size = base_lr_t * (bc2**0.5) / bc1
#                     lrs[p] = step_size / denom
#                 else:
#                     lrs[p] = base_lr_t

#         return directions, lrs
