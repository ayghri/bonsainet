"""
Copyright (c) 2025 Ayoub Ghriss and contributors
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Set, Literal

from torch import Tensor
from torch.nn import Parameter
import torch

from math import prod as _prod

from bonsainet.linalg import kth_largest


def _unsqueeze_odd_dims(tensor: Tensor) -> Tensor:
    """(s1, s2, s3,...) -> (s1, 1, s2, 1, s3, 1,...)."""
    for i in range(1, 2 * tensor.ndim, 2):
        tensor = tensor.unsqueeze(i)
    return tensor


def _merge_odd_dims(tensor: Tensor) -> Tensor:
    """(s0, s1, s2, s3, s4, s5,...) -> (s0, s2, s4,..., s1*s3*s5...)"""
    ndim = tensor.ndim
    permutation = list(range(0, ndim, 2)) + list(range(1, ndim, 2))
    even_shape = [tensor.shape[i] for i in range(0, ndim, 2)]
    return tensor.permute(permutation).reshape(*even_shape, -1)


def _normalize_order(
    order: Optional[Tuple[int, ...]], dim: int
) -> Tuple[int, ...]:
    """Validates and returns a permutation for a tensor of a given dimension."""
    if order is None or len(order) == 0:
        return tuple(range(dim))
    o = list(order)
    if set(o) != set(range(dim)):
        raise ValueError(
            f"order must be a permutation of 0..{dim - 1}, got {order}"
        )
    return tuple(o)


@dataclass
class BlockGroupSpec:
    """
    Specification for for N-D sparsification
      param: torch.nn.Parameter with shape s = (s1,...,sm)
      block_size: (b1,...,bm) with si % bi == 0, block grid B=(si//bi)_i
      if bi=-1 -> bi=si
      group_size: (g1,...,gm) with Bi % gi == 0, group grid G = (Gi = Bi//gi)_i
      if gi = -1 -> gi = Bi
    """

    param: Parameter
    block_size: Tuple[int, ...]
    group_size: Tuple[int, ...]
    name: Optional[str] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.param.shape)

    @property
    def numel(self) -> int:
        return self.param.numel()

    @property
    def ndim(self) -> int:
        return len(self.group_grid_size)

    @property
    def _unsqueezed_block_grid_size(self) -> Tuple[int, ...]:
        return tuple(si // bi for si, bi in zip(self.shape, self.block_size))

    @property
    def block_grid_size(self) -> Tuple[int, ...]:
        shape = tuple(s for s in self._unsqueezed_block_grid_size if s > 1)
        if len(shape) == 0:
            return (1,)
        return shape

    @property
    def num_blocks(self) -> int:
        return _prod(self._unsqueezed_block_grid_size)

    @property
    def block_numel(self) -> int:
        return _prod(self.block_size)

    @property
    def _unsqueezed_group_grid_size(self) -> Tuple[int, ...]:
        return tuple(
            Bi // gi
            for Bi, gi in zip(self._unsqueezed_block_grid_size, self.group_size)
        )

    @property
    def group_grid_size(self) -> Tuple[int, ...]:
        shape = tuple(s for s in self._unsqueezed_group_grid_size if s > 1)
        if len(shape) == 0:
            shape = (1,)
        return shape

    @property
    def num_groups(self) -> int:
        return _prod(self._unsqueezed_group_grid_size)

    @property
    def group_numel(self) -> int:
        return _prod(self.group_size)  # type: ignore

    def nnz(self, eps=1e-8) -> int:
        return int((self.param.data.abs() < eps).sum().item())

    def kappa(self, sparsity):
        return int(self.num_groups * (1 - sparsity))

    def __post_init__(self):
        if len(self.block_size) == 0:  # if block size empty, default to 1
            self.block_size = tuple([1 for i in range(self.param.ndim)])

        if len(self.block_size) != self.param.ndim:
            raise ValueError(
                f"{self.name} block has len {len(self.block_size)}:{self.block_size} "
                f"but tensor is {self.param.ndim}D:{self.param.shape}"
            )
        self.block_size = tuple(
            [
                bi if bi > 0 else self.shape[i]  # -1 means use the entire dim
                for i, bi in enumerate(self.block_size)
            ]
        )
        for i, (si, bi) in enumerate(zip(self.shape, self.block_size)):
            if si % bi != 0:
                raise ValueError(
                    f"dim {i}: size {si} not divisible by block_size[{i}]={bi}"
                )

        if len(self.group_size) == 0:
            self.group_size = tuple(-1 for _ in self.block_grid_size)

        if len(self.group_size) != len(self.block_grid_size):
            raise ValueError(
                f"group size {self.group_size} has len {len(self.group_size)} "
                f"but block_grid_size = {self.block_grid_size}D"
            )
        # A group dim = -1 ->  group spans the entire dimension
        self.group_size = tuple(
            [
                self.block_grid_size[i] if gi == -1 else gi
                for i, gi in enumerate(self.group_size)
            ]
        )

        for i, (Bi, gi) in enumerate(
            zip(self._unsqueezed_block_grid_size, self.group_size)
        ):
            if Bi % gi != 0:
                raise ValueError(
                    f"dim {i}: block_grid[{i}]={Bi} "
                    f"not divisible by group_size[{i}]={gi}"
                )

    def element_to_block(self, x: Tensor, squeeze: bool = True) -> Tensor:
        """Reshapes a tensor from its original shape to a blocked view.
        (s1, s2, ...) -> (B1, b1, B2, b2, ...) where Bi = si/bi
        """
        assert x.shape == self.shape
        shape = []
        for Bi, bi in zip(self._unsqueezed_block_grid_size, self.block_size):
            shape.extend([Bi, bi])
        inter_values = x.view(*shape)
        if squeeze:
            inter_values = inter_values.squeeze()
        return inter_values

        return

    def block_to_element(self, block_values: Tensor) -> Tensor:
        """Broadcast a tensor from a blocked shape back to its original shape.
        (B1, B2, ...) -> (s1, s2, ...) with si = bi*Bi
        """
        assert tuple(block_values.shape) == self.block_grid_size
        inter_values = block_values.view(self._unsqueezed_block_grid_size)
        for i, gi in enumerate(self.block_size):
            inter_values = inter_values.unsqueeze(2 * i + 1)
            inter_values = inter_values.repeat_interleave(gi, dim=2 * i + 1)

        return inter_values.reshape(*self.shape)

    def block_to_group(self, block_values: Tensor, squeeze=True) -> Tensor:
        """
        Reshapes a tensor of block values into a grouped view.
        block_values: shape (B1, B2,...,Bm)
        Returns tensor with shape (G1,g1,G2,g2...,Gm,gm), where Gi=Bi/gi
        """
        assert block_values.shape == self.block_grid_size
        shape = []
        for Gi, gi in zip(self._unsqueezed_group_grid_size, self.group_size):
            shape.extend([Gi, gi])
        inter_values = block_values.view(*shape)
        if squeeze:
            inter_values = inter_values.squeeze()
        return inter_values

    def group_to_block(self, group_values, squeeze=True) -> Tensor:
        """
        Broadcast a tensor of group values back to a block view.
        group_values: shape(G1,...,Gm)
        Returns tensor of shape (B1,...,Bm) with Bi=Gi*gi
        """
        assert tuple(group_values.shape) == self.group_grid_size
        inter_values = group_values.view(self._unsqueezed_group_grid_size)
        for i, gi in enumerate(self.group_size):  # type: ignore
            inter_values = inter_values.unsqueeze(2 * i + 1)
            inter_values = inter_values.repeat_interleave(gi, dim=2 * i + 1)
        inter_values = inter_values.view(self._unsqueezed_block_grid_size)
        if squeeze:
            inter_values = inter_values.view(self.block_grid_size)
        return inter_values

    def block_lp_enerpy(self, values) -> Tensor:
        return self.block_norms(values).pow(2)

    def block_norms(
        self,
        values: Tensor,
        # scale: bool = False,
    ) -> Tensor:
        """Return per-block norms with shape == block_size (L2 default)."""

        # else:
        # s = v.abs().pow(ord).sum(dim=reduce_dims).pow(1.0 / ord)

        # if scale:
        #     s = s / _pow(self.block_numel, 1.0 / ord)
        v = self.element_to_block(values, squeeze=False)
        reduce_dims = tuple(range(1, 2 * self.param.ndim, 2))
        s = torch.linalg.vector_norm(v, ord=2, dim=reduce_dims)
        return s.view(self.block_grid_size)

    @torch.no_grad()
    def hard_threshold(
        self,
        group_thresholds: Optional[Tensor],
        kappa: Optional[int] = None,
        sparsity: Optional[float] = None,
        scale=False,
    ):
        """
        Zeros out blocks in-place based on group-level thresholds.
        """
        if group_thresholds is None:
            if kappa is None:
                if sparsity is None:
                    raise ValueError(
                        "Either group_thresholds or kappa or sparsity should be provided"
                    )
                else:
                    kappa = int((1 - sparsity) * self.group_numel)
            # Shape: (B1,...)
            block_norms = self.block_norms(
                self.param.data
                #    , scale=scale
            )
            # Shape: (G1, g1...)
            group_norms = self.block_to_group(block_norms, squeeze=False)
            group_norms = _merge_odd_dims(group_norms)  # (G1,...Gm, prod(g_i))
            group_norms = group_norms.squeeze()
            group_thresholds = kth_largest(group_norms, k=kappa, dim=-1)

        assert group_thresholds.shape == self.group_grid_size

        block_thresholds = self.group_to_block(group_thresholds)
        block_scores = self.block_norms(
            self.param.data,
            #  scale=scale
        )
        block_mask = (block_scores >= block_thresholds).to(self.param.dtype)
        block_mask = block_mask.view(self._unsqueezed_block_grid_size)
        block_mask = _unsqueeze_odd_dims(block_mask)
        b_view = self.element_to_block(
            self.param.data, squeeze=False
        )  # (B1,bi,...,Bm,bm)
        b_view.mul_(block_mask)

    @torch.no_grad()
    def soft_threshold(self, group_lambdas, eta_t, eps=1e-12, scale=False):
        """
        Applies soft thresholding (proximal operator for L1) to blocks in-place.
        group_lamdas: shape (G1,G2,...,Gm) = self.group_grid_size
        eta_t:
        """
        assert tuple(group_lambdas.shape) == self.group_grid_size

        # if scale:
        # when using block size scale in the regularization:
        #  \sum_B |b|^.5 ||w_b||
        # the goal is to equally threshold individual elements in the block
        # since large blocks have smaller lambda / ||w_b||
        #  the prox operator is
        # [1- (lambda *|b|^.5)/ ||w_b|| ]_+ w_b

        block_lambdas = self.group_to_block(group_lambdas)
        if scale:
            block_lambdas = block_lambdas * (self.block_numel**0.5)
        block_norms = self.block_norms(self.param.data) + eps

        # Normalize eta_t to block space:
        # - if scalar: broadcast
        # - if shape == blocked_shape: use as-is
        # - if shape == element shape: convert to block by mean over within-block dims
        if torch.is_tensor(eta_t):
            if tuple(eta_t.shape) == self.block_grid_size:
                eta_block = eta_t
            elif tuple(eta_t.shape) == tuple(self.shape):
                e_view = self.element_to_block(eta_t, squeeze=False)
                reduce_dims = tuple(range(1, 2 * self.param.ndim, 2))
                eta_block = e_view.mean(dim=reduce_dims)
            else:
                # attempt to treat as scalar-like (e.g., 0-d tensor)
                if eta_t.dim() == 0:
                    eta_block = eta_t
                else:
                    raise ValueError(
                        f"eta_t shape {tuple(eta_t.shape)} not compatible with "
                        f"blocked {self.block_grid_size} or element {self.shape}"
                    )
        else:
            eta_block = torch.as_tensor(eta_t).to(group_lambdas)

        prox_factor = 1 - eta_block * block_lambdas / block_norms
        prox_factor.clamp_(min=0.0)
        # Shape (Bi,1,...)
        prox_factor = prox_factor.view(self._unsqueezed_block_grid_size)
        prox_factor = _unsqueeze_odd_dims(prox_factor)
        # (B1,bi,...,Bm,bm)
        b_view = self.element_to_block(self.param.data, squeeze=False)
        b_view.mul_(prox_factor)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(shape={self.shape}, "
            f"block={self.block_size}, group={self.group_size}, "
            f"B={self.block_grid_size}, G={self.group_grid_size}, "
            f"name={self.name}"
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return int(
            str(hash(self.param)) + str(self.block_numel) + str(self.num_groups)
        )


# @dataclass
# class SpecCoupler:
#     specs: List[BlockGroupSpec]
#     sparsity: float
#     name: Optional[str] = ""

#     _spec_to_kappa: Dict[BlockGroupSpec, int] = field(init=False)


@dataclass
class SpecCoupler:
    """
    Couples multiple BlockGroupSpec instances.

    - Orders (dimension permutations over the *bin grid*) live here.

    Within each aligned bin, we union groups from all specs, select top-κ,
    and hard-threshold parameters in-place (or return masks).
    """

    specs: List[BlockGroupSpec]
    sparsity: float
    orders: List[Tuple[int, ...]]
    mode: Literal["concat", "sum"] = "concat"
    _reverse_orders: List[Tuple[int, ...]] = field(init=False)
    _ref_order: Tuple[int] = field(init=False)
    _ref_group_grid_size: Tuple[int] = field(init=False)

    @property
    def num_blocks(self) -> int:
        return sum([s.group_numel for s in self.specs])

    @property
    def params(self) -> Set[Parameter]:
        """Expose underlying parameters for optimizer integration."""
        return {s.param for s in self.specs}

    @property
    def numel(self) -> int:
        return sum([s.numel for s in self.specs])

    @property
    def kappa(self):
        return int((1 - self.sparsity) * self.num_blocks)

    # def kappa(self, sparsity: float):
    #     return int((1 - sparsity) * self.num_blocks)

    def __post_init__(self):
        if self.orders is None or len(self.orders) == 0:
            self.orders = [tuple(range(s.ndim)) for s in self.specs]
        if len(self.orders) != len(self.specs):
            raise ValueError("orders must match number of specs.")

        self.orders = [
            _normalize_order(o, s.ndim) for o, s in zip(self.orders, self.specs)
        ]

        # ref_gpermute: Tuple[int, ...] = None  # type: ignore
        # Ensure that all specs have compatible grouped shapes after permutation
        self._ref_order = self.orders[0]  # type: ignore
        self._ref_group_grid_size = ref_permute = tuple(  # type: ignore
            self.specs[0].group_grid_size[i] for i in self._ref_order
        )
        for s, o in zip(self.specs[1:], self.orders[1:]):
            Gi = s.group_grid_size
            gperm = tuple(Gi[i] for i in o)
            if gperm != ref_permute:
                raise ValueError(
                    "Incompatible grouped shapes "
                    f"after order: {gperm} vs {ref_permute} "
                    f"(spec {s.name or '<unnamed>'})"
                )

        # Store the reverse permutation for each order
        self._reverse_orders = [
            tuple(o.index(i) for i in range(len(o))) for o in self.orders
        ]
        ref_spec = self.specs[0]
        ref_ord = self._ref_order
        if self.mode == "sum":
            for other_s, other_o in zip(self.specs[1:], self.orders[1:]):
                assert tuple(
                    other_s.group_size[o_o] for o_o in other_o
                ) == tuple(ref_spec.group_size[o_r] for o_r in ref_ord)

    def coupled_norm(
        self,
        element_values: Dict[BlockGroupSpec, Tensor],
    ) -> Tensor:
        if self.mode == "concat":
            grouped_scores = []
            for s, o in zip(self.specs, self.orders):
                e_vals = element_values[s]
                block_norms = s.block_norms(e_vals)  # Shape: (B1, B2, ...)
                group_norms = s.block_to_group(block_norms, squeeze=False)
                # (G1, ..., prod(g_i))
                group_norms = _merge_odd_dims(group_norms)
                group_norms = group_norms.view(*s.group_grid_size, -1)
                # Permute and append scores for this spec
                grouped_scores.append(group_norms.permute(o + (len(o),)))

            # Concatenate scores from all specs along the last dimension
            if len(grouped_scores) > 1:
                grouped_scores = torch.cat(grouped_scores, dim=-1)
            else:
                grouped_scores = grouped_scores[0]
            return grouped_scores
        else:
            grouped_scores = torch.zeros(self._ref_group_grid_size).to(
                element_values[self.specs[0]]
            )

            for s, o in zip(self.specs, self.orders):
                e_vals = element_values[s]
                # Shape: (B1, B2, ...)
                block_norms = s.block_lp_enerpy(e_vals)
                # Shape: (G1, g1,....)
                group_norms = s.block_to_group(block_norms)
                permutation = []
                for o_o in o:
                    permutation.append(2 * o_o)
                    permutation.append(2 * o_o + 1)
                group_norms = group_norms.permute(permutation)
                grouped_scores += group_norms

            grouped_scores.pow(0.5)

            grouped_scores = _merge_odd_dims(grouped_scores)

        return grouped_scores

    def kth_largest(
        self,
        element_values: Dict[BlockGroupSpec, Tensor],
        kappa=None,
        # scale=False,
    ):
        """
        Calculates the k-th largest score across all groups from all specs.
        This is used to determine the threshold for pruning.
        """
        assert len(element_values) == len(self.specs)
        grouped_scores = self.coupled_norm(
            element_values=element_values,
            #   scale=scale
        )

        if kappa is not None:
            return kth_largest(grouped_scores, k=kappa, dim=-1)
        return kth_largest(grouped_scores, k=self.kappa, dim=-1)

    @torch.no_grad()
    def hard_threshold(self, kappa: Optional[int] = None, scale=False):
        """Compute kappa-largest block_norm among coupled groups from
        all specs then sends the threshold to specs to hard-threshold in-place.
        Note that the threshold is across coupled-groups, so some parameters
        might be pruned more than others (it's expected).
        """

        if kappa is None:
            kappa = self.kappa
        group_thresholds = self.kth_largest(
            {s: s.param.data for s in self.specs},
            kappa,
            # scale=scale
        )
        for ro, s in zip(self._reverse_orders, self.specs):
            # Permute thresholds back to original order
            s.hard_threshold(
                group_thresholds=group_thresholds.permute(ro), scale=scale
            )
        return group_thresholds

    def soft_threshold(
        self,
        coupled_lambdas: Tensor,
        learning_rates: Dict[BlockGroupSpec, Tensor],
    ):
        """
        Performs soft thresholding on all coupled parameters.
        """
        for ro, s in zip(self._reverse_orders, self.specs):
            s.soft_threshold(
                coupled_lambdas.permute(ro), eta_t=learning_rates[s]
            )

    def __hash__(self):
        key = (
            tuple(
                (id(s.param), s.block_size, s.group_size) for s in self.specs
            ),
            tuple(self.orders),
        )
        return hash(key)

    def __repr__(self):
        return f"GroupCoupler: orders={self.orders}\n\t" + "\n\t".join(
            [str(s) for s in self.specs]
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    U = torch.nn.Parameter(torch.randn(4, 8, 2, 2, device="cuda"))
    V = torch.nn.Parameter(torch.randn(8, 16, 2, 2, device="cuda"))

    spec_u = BlockGroupSpec(
        U, block_size=(2, 2, 2, 2), group_size=(1, 1), name="U"
    )
    spec_v = BlockGroupSpec(
        V, block_size=(2, 2, 2, 2), group_size=(1, 4), name="V"
    )
    print(spec_u)
    print(spec_v)
    coupled = SpecCoupler(
        [spec_u, spec_v], orders=[(0, 1), (1, 0)], sparsity=0.25
    )
    masks = coupled.hard_threshold(kappa=2)
    print(masks.squeeze())
    spec_u.block_norms(U).squeeze()
    spec_v.block_norms(V).squeeze()

    U = torch.nn.Parameter(torch.randn(4, 8, device="cuda"))
    V = torch.nn.Parameter(torch.randn(8, 16, device="cuda"))
    print(U)
    print(V)

    spec_u = BlockGroupSpec(U, block_size=(2, 2), group_size=(1, 1), name="U")
    spec_v = BlockGroupSpec(V, block_size=(2, 2), group_size=(1, 4), name="V")
    print(spec_u)
    print(spec_v)

    coupled = SpecCoupler(
        [spec_u, spec_v], orders=[(0, 1), (1, 0)], sparsity=0.5
    )

    masks = coupled.hard_threshold(kappa=2)
    print(masks.squeeze())
    print(U)
    print(V)
