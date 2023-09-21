import torch
from encoder import TransformerEncoder
from decoder import TransformerDecoder
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 max_len: int,
                 pad_idx: int=0,
                 d_model: int=512,
                 N: int=6,
                 d_ff: int=2048,
                 heads_num: int=8,
                 dropout_rate: float=0.1,
                 layer_norm_eps: float=1e-5,
                 device: torch.device=torch.device("cpu")) -> None:
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.heads_num = heads_num
        self.d_ff = d_ff
        self.N = N
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.pad_idx = pad_idx
        self.device = device

        self.encoder = TransformerEncoder(
            src_vocab_size, max_len, pad_idx, d_model, N, d_ff, heads_num, dropout_rate, layer_norm_eps, device
        )

        self.decoder = TransformerDecoder(tgt_vocab_size, max_len, pad_idx, d_model, N, d_ff, heads_num, dropout_rate, layer_norm_eps, device)

        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Params:
            src: list of word idxs (batch_size, max_len)
            tgt: list of word idxs (batch_size, max_len)
        """

        # mask
        pad_mask_src = self._pad_mask(src)
        src = self.encoder(src, pad_mask_src)

        mask_self_attn = torch.logical_or(
            self._subsequent_mask(tgt), self._pad_mask(tgt)
        )
        dec_output = self.decoder(tgt, src, pad_mask_src, mask_self_attn)

        return self.linear(dec_output)
    
    def _pad_mask(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        mask = x.eq(self.pad_idx)
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, seq_len, 1)
        return mask.to(self.device)
    
    def _subsequent_mask(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        max_len = x.size(1)

        return (torch.tril(torch.ones(batch_size, max_len, max_len)).eq(0).to(self.device))
