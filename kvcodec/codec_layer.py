"""
kvcodec/codec_layer.py
======================
Layer-axis KV codec.

I-frames: anchor layers stored at full precision
P-frames: INT8 quantized residuals relative to nearest anchor layer
Operates across the full stack of layers simultaneously.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import LayerCodecConfig


@dataclass
class CompressedLayerStack:
    """Compressed representation of ALL layers (the full KV stack)."""
    anchors_k:       torch.Tensor    # [n_anchors, H, T, D] fp16
    anchors_v:       torch.Tensor
    residuals_k:     torch.Tensor    # [n_layers, H, T, D] int8
    residuals_v:     torch.Tensor
    scales_k:        torch.Tensor    # [n_layers, H, T] fp32
    scales_v:        torch.Tensor
    layer_anchor_map: torch.Tensor   # [n_layers] int64 → anchor index
    anchor_layers:   List[int]
    n_layers: int
    n_heads:  int
    seq_len:  int
    head_dim: int
    norm_k:   Optional[torch.Tensor] = None   # [n_layers, H, T] if normalised
    norm_v:   Optional[torch.Tensor] = None
    predictor_used: bool = False

    def bytes_compressed(self) -> int:
        return (
            (self.anchors_k.numel() + self.anchors_v.numel()) * 2 +
            (self.residuals_k.numel() + self.residuals_v.numel()) * 1 +
            (self.scales_k.numel() + self.scales_v.numel()) * 4 +
            self.layer_anchor_map.numel() * 4
        )

    def bytes_original(self) -> int:
        return self.n_layers * self.n_heads * self.seq_len * self.head_dim * 2 * 2

    def compression_ratio(self) -> float:
        c = self.bytes_compressed()
        return self.bytes_original() / c if c > 0 else 0.0


class LayerCodec:
    """Layer-axis cross-layer delta KV codec."""

    def __init__(self, config: LayerCodecConfig = None):
        self.cfg = config or LayerCodecConfig()

    def _resolve_anchor_layers(self, n_layers: int) -> List[int]:
        if self.cfg.anchor_layers is not None:
            return sorted(self.cfg.anchor_layers)
        stride = self.cfg.anchor_layer_stride
        anchors = set(range(0, n_layers, stride))
        anchors.add(0)
        return sorted(anchors)

    def _nearest_anchor_layer(self, layer: int, anchors: List[int]) -> int:
        candidates = [a for a in anchors if a <= layer]
        return candidates[-1] if candidates else anchors[0]

    def _quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-vector INT8 quantization. x: [..., D]"""
        scales = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        x_q = (x / scales * 127).round().clamp(-128, 127).to(torch.int8)
        return x_q, scales.squeeze(-1).float()

    def _dequantize(self, x_q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return x_q.float() * (scales.unsqueeze(-1) / 127)

    def compress(
        self,
        all_keys: List[torch.Tensor],    # list of [H, T, D] per layer
        all_values: List[torch.Tensor],
        predictor=None,
    ) -> CompressedLayerStack:
        """
        Compress the full layer stack.
        all_keys: list of n_layers tensors, each [H, T, D]
        """
        n_layers = len(all_keys)
        H, T, D = all_keys[0].shape

        # Stack: [L, H, T, D]
        k_stack = torch.stack([k.float() for k in all_keys])
        v_stack = torch.stack([v.float() for v in all_values])

        anchor_layers = self._resolve_anchor_layers(n_layers)
        layer_anchor_map = torch.tensor(
            [anchor_layers.index(self._nearest_anchor_layer(l, anchor_layers))
             for l in range(n_layers)],
            dtype=torch.long
        )

        # Optional L2 normalisation per-vector
        norm_k = norm_v = None
        if self.cfg.normalize_residuals:
            norm_k = k_stack.norm(dim=-1).clamp(min=1e-6)   # [L, H, T]
            norm_v = v_stack.norm(dim=-1).clamp(min=1e-6)
            k_stack = k_stack / norm_k.unsqueeze(-1)
            v_stack = v_stack / norm_v.unsqueeze(-1)

        anchor_idx_t = torch.tensor(anchor_layers, dtype=torch.long)
        anchors_k = k_stack[anchor_idx_t].half()   # [A, H, T, D]
        anchors_v = v_stack[anchor_idx_t].half()

        anchor_exp_k = anchors_k.float()[layer_anchor_map]   # [L, H, T, D]
        anchor_exp_v = anchors_v.float()[layer_anchor_map]

        if predictor is not None:
            nearest_layer_t = torch.tensor(
                [anchor_layers[layer_anchor_map[l].item()] for l in range(n_layers)],
                dtype=torch.float
            )
            delta = torch.arange(n_layers, dtype=torch.float) - nearest_layer_t
            pred_k, pred_v = predictor.predict_layer(
                anchor_exp_k, anchor_exp_v, delta
            )
            res_k = k_stack - pred_k
            res_v = v_stack - pred_v
        else:
            res_k = k_stack - anchor_exp_k
            res_v = v_stack - anchor_exp_v

        for res in [res_k, res_v]:
            std = res.std().clamp(min=1e-6)
            res.clamp_(-3.0 * std, 3.0 * std)

        res_k_q, scales_k = self._quantize(res_k)
        res_v_q, scales_v = self._quantize(res_v)

        return CompressedLayerStack(
            anchors_k=anchors_k, anchors_v=anchors_v,
            residuals_k=res_k_q, residuals_v=res_v_q,
            scales_k=scales_k, scales_v=scales_v,
            layer_anchor_map=layer_anchor_map,
            anchor_layers=anchor_layers,
            n_layers=n_layers, n_heads=H, seq_len=T, head_dim=D,
            norm_k=norm_k, norm_v=norm_v,
            predictor_used=predictor is not None,
        )

    def decompress(
        self,
        c: CompressedLayerStack,
        predictor=None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Returns list of [H, T, D] tensors per layer."""
        anchor_exp_k = c.anchors_k.float()[c.layer_anchor_map]
        anchor_exp_v = c.anchors_v.float()[c.layer_anchor_map]

        res_k = self._dequantize(c.residuals_k, c.scales_k)
        res_v = self._dequantize(c.residuals_v, c.scales_v)

        if predictor is not None and c.predictor_used:
            nearest_layer_t = torch.tensor(
                [c.anchor_layers[c.layer_anchor_map[l].item()]
                 for l in range(c.n_layers)],
                dtype=torch.float
            )
            delta = torch.arange(c.n_layers, dtype=torch.float) - nearest_layer_t
            pred_k, pred_v = predictor.predict_layer(
                anchor_exp_k, anchor_exp_v, delta
            )
            k_stack = pred_k + res_k
            v_stack = pred_v + res_v
        else:
            k_stack = anchor_exp_k + res_k
            v_stack = anchor_exp_v + res_v

        if c.norm_k is not None:
            k_stack = k_stack * c.norm_k.unsqueeze(-1)
            v_stack = v_stack * c.norm_v.unsqueeze(-1)

        return (
            [k_stack[l] for l in range(c.n_layers)],
            [v_stack[l] for l in range(c.n_layers)],
        )

    def metrics(
        self,
        all_keys_orig: List[torch.Tensor],
        all_values_orig: List[torch.Tensor],
        c: CompressedLayerStack,
        predictor=None,
    ) -> Dict:
        k_recs, v_recs = self.decompress(c, predictor)
        D = c.head_dim
        cos_k = cos_v = mse_k = mse_v = 0.0
        n = len(all_keys_orig)
        for l in range(n):
            k_o = all_keys_orig[l].float()
            v_o = all_values_orig[l].float()
            k_r = k_recs[l].float()
            v_r = v_recs[l].float()
            cos_k += F.cosine_similarity(k_o.reshape(-1,D), k_r.reshape(-1,D), dim=-1).mean().item()
            cos_v += F.cosine_similarity(v_o.reshape(-1,D), v_r.reshape(-1,D), dim=-1).mean().item()
            mse_k += F.mse_loss(k_r, k_o).item()
            mse_v += F.mse_loss(v_r, v_o).item()
        return {
            'strategy':           'layer_only',
            'anchor_layer_stride': self.cfg.anchor_layer_stride,
            'cosine_sim_k':       cos_k / n,
            'cosine_sim_v':       cos_v / n,
            'mse_k':              mse_k / n,
            'mse_v':              mse_v / n,
            'compression_ratio':  c.compression_ratio(),
            'bytes_saved_pct':    (1 - 1/c.compression_ratio()) * 100,
            'predictor_used':     c.predictor_used,
        }
