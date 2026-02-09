"""
SIREN (Sinusoidal Representation Network) for density prediction.

Experimental architecture based on sine activations with specialized
initialization. Not wired into the public training API yet.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import math


class SineLayer(nn.Module):
    """Linear layer followed by sine activation with SIREN initialization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1 / self.in_features
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """SIREN regressor for chemical density prediction."""

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        use_bias: bool = True,
        outermost_linear: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [32, 32]
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.use_bias = use_bias
        self.outermost_linear = outermost_linear

        layers = []
        dims = [self.input_dim] + self.hidden_dims

        for i in range(len(dims) - 1):
            layers.append(
                SineLayer(
                    dims[i],
                    dims[i + 1],
                    bias=self.use_bias,
                    is_first=(i == 0),
                    omega_0=self.first_omega_0 if i == 0 else self.hidden_omega_0,
                )
            )

        if self.outermost_linear:
            final_linear = nn.Linear(dims[-1], self.output_dim, bias=self.use_bias)
            with torch.no_grad():
                bound = math.sqrt(6 / dims[-1]) / self.hidden_omega_0
                final_linear.weight.uniform_(-bound, bound)
                if final_linear.bias is not None:
                    final_linear.bias.uniform_(-bound, bound)
            layers.append(final_linear)
        else:
            layers.append(
                SineLayer(
                    dims[-1],
                    self.output_dim,
                    bias=self.use_bias,
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        return (
            f"SIREN(dims={dims}, first_omega_0={self.first_omega_0}, "
            f"hidden_omega_0={self.hidden_omega_0}, params={self.get_num_parameters():,})"
        )


__all__ = ["SIREN", "SineLayer"]
