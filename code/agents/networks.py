from __future__ import annotations

import torch
import torch.nn as nn

class MLPQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_sizes=(128, 128),
        negative_slope: float = 0.01,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        layers = []
        last = state_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            last = h

        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        # Kaiming init for LeakyReLU, changed over time
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
