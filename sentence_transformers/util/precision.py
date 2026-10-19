from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _Float32Linear(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        with torch.autocast(device_type=input.device.type, enabled=False):
            return F.linear(input.float(), self.weight.float(), self.bias.float() if self.bias is not None else None)
