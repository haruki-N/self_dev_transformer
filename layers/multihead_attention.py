import torch
from torch import nn
from layers.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_key = d_model // num_heads
        self.d_value = d_model // num_heads

        self.W_key = nn.Parameter(torch.Tensor(num_heads, d_model, self.d_key))
        self.W_query = nn.Parameter(torch.Tensor(num_heads, d_model, self.d_key))
        self.W_value = nn.Parameter(torch.Tensor(num_heads, d_model, self.d_value))

        self.scaled_dot_attention = ScaledDotProductAttention(self.d_key)

        self.linear = nn.Linear(num_heads * self.d_value, d_model)

    def forward(self, query, key, value, mask_3d) -> torch.Tensor:
        batch_size, seq_len = query.size(0), query.size(1)

        # Repeat Q, K, V inputs #heads times
        q = q.repeat(self.num_heads, 1, 1, 1)   # heads, batch_size, seq_len, d_model
        k = k.repeat(self.num_heads, 1, 1, 1)
        v = v.repeat(self.num_heads, 1, 1, 1)

        # Apply Linear before scaled dot product
        q = torch.einsum("hijk,hkl -> hijl", q, self.W_query)
        k = torch.einsum("hijk,hkl -> hijl", k, self.W_key)
        v = torch.einsum("hijk,hkl -> hijl", v, self.W_value)

        # Split heads
        q = q.view(self.num_heads * batch_size, seq_len, self.d_k)
        k = k.view(self.num_heads * batch_size, seq_len, self.d_k)
        v = v.view(self.num_heads * batch_size, seq_len, self.d_v)

        if mask_3d is not None:
            mask_3d = mask_3d.repeat(self.num_heads, 1, 1)

        # Scaled Dot Product Attention
        attention_output = self.scaled_dot_product_attention(q, k, v, mask_3d)  # (head*batch_size, seq_len, d_model)

        attention_output = torch.chunk(attention_output, self.num_heads, dim=0)
        attention_output = torch.cat(attention_output, dim=2)
        
        # Final linear
        attention_output = self.linear(attention_output)

        return attention_output
