"""
kvcodec/codec_seq.py
====================
Sequence-axis KV codec.

I-frames: anchor tokens at fixed stride + BOS
P-frames: INT8 quantized residuals relative to nearest causal anchor
Optional: MLP predictor reduces residual magnitude before quantization
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import SeqCodecConfig


@dataclass
class CompressedSeqLayer:
    anchors_k:      torch.Tensor   # [n_anchors, H, D] fp16
    anchors_v:      torch.Tensor
    residuals_k:    torch.Tensor   # [T, H, D] int8
    residuals_v:    torch.Tensor
    scales_k:       torch.Tensor   # [T, H] fp32
    scales_v:       torch.Tensor
    anchor_map:     torch.Tensor   # [T] int64 → index into anchors
    anchor_positions: List[int]
    seq_len: int
    n_heads: int
    head_dim: int
    norm_k: Optional[torch.Tensor] = None   # [T, H] fp32 if normalized
    norm_v: Optional[torch.Tensor] = None
    predictor_used: bool = False

    def bytes_compressed(self) -> int:
        return (
            (self.anchors_k.numel() + self.anchors_v.numel()) * 2 +   # fp16
            (self.residuals_k.numel() + self.residuals_v.numel()) * 1 + # int8
            (self.scales_k.numel() + self.scales_v.numel()) * 4 +      # fp32
            self.anchor_map.numel() * 4                                  # int64
        )

    def bytes_original(self) -> int:
        return self.seq_len * self.n_heads * self.head_dim * 2 * 2  # K+V fp16

    def compression_ratio(self) -> float:
        c = self.bytes_compressed()
        return self.bytes_original() / c if c > 0 else 0.0


class SeqCodec:
    """Sequence-axis P-frame KV codec."""

    def __init__(self, config: SeqCodecConfig = None):
        self.cfg = config or SeqCodecConfig()

    def _anchor_positions(self, seq_len: int) -> List[int]:
        anchors = set(range(0, seq_len, self.cfg.anchor_stride))
        if self.cfg.anchor_at_bos:
            anchors.add(0)
        return sorted(anchors)

    def _nearest_causal_anchor(self, pos: int, anchors: List[int]) -> int:
        candidates = [a for a in anchors if a <= pos]
        return candidates[-1] if candidates else anchors[0]

    def _quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-vector symmetric INT8 quantization. x: [..., D]"""
        scales = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        x_q = (x / scales * 127).round().clamp(-128, 127).to(torch.int8)
        return x_q, scales.squeeze(-1).float()

    def _dequantize(self, x_q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return x_q.float() * (scales.unsqueeze(-1) / 127)

    def compress(
        self,
        keys: torch.Tensor,      # [H, T, D]
        values: torch.Tensor,
        predictor=None,
    ) -> CompressedSeqLayer:
        H, T, D = keys.shape
        k = keys.permute(1, 0, 2).float()     # [T, H, D]
        v = values.permute(1, 0, 2).float()

        anchor_pos = self._anchor_positions(T)
        anchor_map = torch.tensor(
            [anchor_pos.index(self._nearest_causal_anchor(t, anchor_pos))
             for t in range(T)],
            dtype=torch.long
        )

        # Optional L2 normalisation
        norm_k = norm_v = None
        if self.cfg.normalize_residuals:
            norm_k = k.norm(dim=-1).clamp(min=1e-6)   # [T, H]
            norm_v = v.norm(dim=-1).clamp(min=1e-6)
            k = k / norm_k.unsqueeze(-1)
            v = v / norm_v.unsqueeze(-1)

        anchors_k = k[torch.tensor(anchor_pos)].half()   # [A, H, D]
        anchors_v = v[torch.tensor(anchor_pos)].half()

        anchor_exp_k = anchors_k.float()[anchor_map]     # [T, H, D]
        anchor_exp_v = anchors_v.float()[anchor_map]

        if predictor is not None:
            nearest_pos_t = torch.tensor(
                [anchor_pos[anchor_map[t].item()] for t in range(T)],
                dtype=torch.float
            )
            delta = torch.arange(T, dtype=torch.float) - nearest_pos_t
            pred_k, pred_v = predictor.predict(anchor_exp_k, anchor_exp_v, delta)
            res_k = k - pred_k
            res_v = v - pred_v
        else:
            res_k = k - anchor_exp_k
            res_v = v - anchor_exp_v

        # Clip before quantization
        for res in [res_k, res_v]:
            std = res.std().clamp(min=1e-6)
            res.clamp_(-self.cfg.max_residual_clip * std,
                        self.cfg.max_residual_clip * std)

        res_k_q, scales_k = self._quantize(res_k)
        res_v_q, scales_v = self._quantize(res_v)

        return CompressedSeqLayer(
            anchors_k=anchors_k, anchors_v=anchors_v,
            residuals_k=res_k_q, residuals_v=res_v_q,
            scales_k=scales_k, scales_v=scales_v,
            anchor_map=anchor_map, anchor_positions=anchor_pos,
            seq_len=T, n_heads=H, head_dim=D,
            norm_k=norm_k, norm_v=norm_v,
            predictor_used=predictor is not None,
        )

    def decompress(
        self,
        c: CompressedSeqLayer,
        predictor=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor_exp_k = c.anchors_k.float()[c.anchor_map]
        anchor_exp_v = c.anchors_v.float()[c.anchor_map]

        res_k = self._dequantize(c.residuals_k, c.scales_k)
        res_v = self._dequantize(c.residuals_v, c.scales_v)

        if predictor is not None and c.predictor_used:
            nearest_pos_t = torch.tensor(
                [c.anchor_positions[c.anchor_map[t].item()] for t in range(c.seq_len)],
                dtype=torch.float
            )
            delta = torch.arange(c.seq_len, dtype=torch.float) - nearest_pos_t
            pred_k, pred_v = predictor.predict(anchor_exp_k, anchor_exp_v, delta)
            k = pred_k + res_k
            v = pred_v + res_v
        else:
            k = anchor_exp_k + res_k
            v = anchor_exp_v + res_v

        if c.norm_k is not None:
            k = k * c.norm_k.unsqueeze(-1)
            v = v * c.norm_v.unsqueeze(-1)

        return k.permute(1, 0, 2), v.permute(1, 0, 2)   # [H, T, D]

    def metrics(
        self,
        keys_orig: torch.Tensor,
        values_orig: torch.Tensor,
        c: CompressedSeqLayer,
        predictor=None,
    ) -> Dict:
        k_rec, v_rec = self.decompress(c, predictor)
        k_o = keys_orig.float();  v_o = values_orig.float()
        k_r = k_rec.float();      v_r = v_rec.float()
        D = c.head_dim
        return {
            'strategy':          'seq_only',
            'anchor_stride':     self.cfg.anchor_stride,
            'cosine_sim_k':      F.cosine_similarity(k_o.reshape(-1,D), k_r.reshape(-1,D), dim=-1).mean().item(),
            'cosine_sim_v':      F.cosine_similarity(v_o.reshape(-1,D), v_r.reshape(-1,D), dim=-1).mean().item(),
            'mse_k':             F.mse_loss(k_r, k_o).item(),
            'mse_v':             F.mse_loss(v_r, v_o).item(),
            'compression_ratio': c.compression_ratio(),
            'bytes_saved_pct':   (1 - 1/c.compression_ratio()) * 100,
            'predictor_used':    c.predictor_used,
        }
