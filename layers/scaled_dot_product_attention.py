import numpy as np
import torch
from torch import nn



class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scalar = np.sqrt(self.d_k)

        # Q * K^T / sqrt(d_k)
        attention_weight = torch.matmul(q, torch.transpose(k, 1, 2)) / scalar

        if mask is not None:
            if mask.dim() != attention_weight.dim():
                raise ValueError(f"Mask's dimention ({mask.dim()}) shold be equal to \
                                 attention weight's dimention ({attention_weight.dim()})")
            attention_weight = attention_weight.data.masked_fill(mask, torch.finfo(torch.float).max)
            
        # Softmax(Q * K^T / sqrt(d_k))
        attention_weight = nn.functional.softmax(attention_weight, dim=2)
        return torch.matmul(attention_weight, v)