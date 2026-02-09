"""
KAN (Kolmogorov-Arnold Network) model wrapper for density prediction.

Uses the fast-kan package for spline-based layers. This implementation is
experimental and is not wired into the public training API yet.
"""

from typing import List, Optional, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fastkan import FastKAN
    FASTKAN_AVAILABLE = True
except ImportError:
    FASTKAN_AVAILABLE = False
    FastKAN = None


BaseActivation = Union[str, Callable[[torch.Tensor], torch.Tensor]]


def _resolve_base_activation(activation: BaseActivation) -> Callable[[torch.Tensor], torch.Tensor]:
    if callable(activation):
        return activation

    activation_map = {
        "silu": F.silu,
        "relu": F.relu,
        "tanh": torch.tanh,
        "gelu": F.gelu,
    }

    if activation not in activation_map:
        raise ValueError(
            f"Unknown base activation '{activation}'. "
            f"Choose from: {sorted(activation_map.keys())}"
        )

    return activation_map[activation]


class KANRegressor(nn.Module):
    """KAN-based regressor for chemical density prediction."""

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation: BaseActivation = "silu",
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()

        if not FASTKAN_AVAILABLE:
            raise ImportError("fast-kan is not installed. Install with: pip install fast-kan")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [8, 8]
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        self.spline_weight_init_scale = spline_weight_init_scale

        layers_hidden = [self.input_dim] + self.hidden_dims + [self.output_dim]
        base_act_fn = _resolve_base_activation(self.base_activation)

        self.network = FastKAN(
            layers_hidden=layers_hidden,
            grid_min=self.grid_min,
            grid_max=self.grid_max,
            num_grids=self.num_grids,
            use_base_update=self.use_base_update,
            base_activation=base_act_fn,
            spline_weight_init_scale=self.spline_weight_init_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        return (
            f"KANRegressor(dims={dims}, num_grids={self.num_grids}, "
            f"base_activation={self.base_activation}, params={self.get_num_parameters():,})"
        )


__all__ = ["KANRegressor", "FASTKAN_AVAILABLE"]
