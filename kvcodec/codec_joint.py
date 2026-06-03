"""
kvcodec/codec_joint.py
======================
2D joint KV codec — the full codec-complete framework.

Operates on both axes simultaneously:
  Layer axis:    I-frame anchor layers, P-frame delta across depth
  Sequence axis: I-frame anchor tokens, P-frame delta across positions

Each KV vector is encoded relative to a 2D anchor point (anchor_layer, anchor_pos).
The residual is smaller than either 1D codec alone because both sources
of redundancy are removed before quantization.

This is the novel contribution. No prior work applies joint 2D compression.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import JointCodecConfig


@dataclass
class CompressedJoint:
    """
    Full 2D compressed representation.
    Anchor tensors stored at (anchor_layer, anchor_pos) intersections.
    All other positions encoded as residuals from nearest 2D anchor.
    """
    # Anchors: at intersections of anchor layers and anchor positions
    anchors_k:       torch.Tensor    # [n_anchor_layers, n_anchor_pos, H, D] fp16
    anchors_v:       torch.Tensor
    # Residuals for all (layer, pos) pairs
    residuals_k:     torch.Tensor    # [L, T, H, D] int8
    residuals_v:     torch.Tensor
    scales_k:        torch.Tensor    # [L, T, H] fp32
    scales_v:        torch.Tensor
    # Maps
    layer_anchor_map: torch.Tensor   # [L] → index into anchor_layers
    pos_anchor_map:   torch.Tensor   # [T] → index into anchor_positions
    anchor_layers:   List[int]
    anchor_positions: List[int]
    n_layers: int
    n_heads:  int
    seq_len:  int
    head_dim: int
    norm_k:   Optional[torch.Tensor] = None   # [L, T, H] if normalised
    norm_v:   Optional[torch.Tensor] = None
    predictor_used: bool = False

    def bytes_compressed(self) -> int:
        return (
            (self.anchors_k.numel() + self.anchors_v.numel()) * 2 +
            (self.residuals_k.numel() + self.residuals_v.numel()) * 1 +
            (self.scales_k.numel() + self.scales_v.numel()) * 4 +
            (self.layer_anchor_map.numel() + self.pos_anchor_map.numel()) * 4
        )

    def bytes_original(self) -> int:
        return self.n_layers * self.seq_len * self.n_heads * self.head_dim * 2 * 2

    def compression_ratio(self) -> float:
        c = self.bytes_compressed()
        return self.bytes_original() / c if c > 0 else 0.0


class JointCodec:
    """2D joint KV codec — operates on both layer and sequence axes."""

    def __init__(self, config: JointCodecConfig = None):
        self.cfg = config or JointCodecConfig()

    def _layer_anchors(self, n_layers: int) -> List[int]:
        anchors = set(range(0, n_layers, self.cfg.layer_anchor_stride))
        anchors.add(0)
        return sorted(anchors)

    def _seq_anchors(self, seq_len: int) -> List[int]:
        anchors = set(range(0, seq_len, self.cfg.seq_anchor_stride))
        anchors.add(0)
        return sorted(anchors)

    def _nearest_prior(self, pos: int, anchors: List[int]) -> int:
        candidates = [a for a in anchors if a <= pos]
        return candidates[-1] if candidates else anchors[0]

    def _quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    ) -> CompressedJoint:
        n_layers = len(all_keys)
        H, T, D = all_keys[0].shape

        # Stack: [L, H, T, D] → rearrange to [L, T, H, D] for joint indexing
        k = torch.stack([kk.float() for kk in all_keys]).permute(0, 2, 1, 3)  # [L,T,H,D]
        v = torch.stack([vv.float() for vv in all_values]).permute(0, 2, 1, 3)

        anchor_layers    = self._layer_anchors(n_layers)
        anchor_positions = self._seq_anchors(T)

        layer_anchor_map = torch.tensor(
            [anchor_layers.index(self._nearest_prior(l, anchor_layers))
             for l in range(n_layers)], dtype=torch.long
        )
        pos_anchor_map = torch.tensor(
            [anchor_positions.index(self._nearest_prior(p, anchor_positions))
             for p in range(T)], dtype=torch.long
        )

        # Normalise
        norm_k = norm_v = None
        if self.cfg.normalize_residuals:
            norm_k = k.norm(dim=-1).clamp(min=1e-6)   # [L, T, H]
            norm_v = v.norm(dim=-1).clamp(min=1e-6)
            k = k / norm_k.unsqueeze(-1)
            v = v / norm_v.unsqueeze(-1)

        # Extract 2D anchor grid: [A_l, A_p, H, D]
        al_t = torch.tensor(anchor_layers, dtype=torch.long)
        ap_t = torch.tensor(anchor_positions, dtype=torch.long)
        anchors_k = k[al_t][:, ap_t].half()   # [A_l, A_p, H, D]
        anchors_v = v[al_t][:, ap_t].half()

        # Expand anchor grid to full [L, T, H, D]
        anchor_exp_k = anchors_k.float()[layer_anchor_map][:, pos_anchor_map]
        anchor_exp_v = anchors_v.float()[layer_anchor_map][:, pos_anchor_map]

        if predictor is not None:
            nearest_l = torch.tensor(
                [anchor_layers[layer_anchor_map[l].item()] for l in range(n_layers)],
                dtype=torch.float
            )
            nearest_p = torch.tensor(
                [anchor_positions[pos_anchor_map[p].item()] for p in range(T)],
                dtype=torch.float
            )
            delta_l = (torch.arange(n_layers, dtype=torch.float) - nearest_l)
            delta_p = (torch.arange(T, dtype=torch.float) - nearest_p)
            pred_k, pred_v = predictor.predict_joint(
                anchor_exp_k, anchor_exp_v, delta_l, delta_p
            )
            res_k = k - pred_k
            res_v = v - pred_v
        else:
            res_k = k - anchor_exp_k   # [L, T, H, D]
            res_v = v - anchor_exp_v

        for res in [res_k, res_v]:
            std = res.std().clamp(min=1e-6)
            res.clamp_(-3.0 * std, 3.0 * std)

        res_k_q, scales_k = self._quantize(res_k)
        res_v_q, scales_v = self._quantize(res_v)

        return CompressedJoint(
            anchors_k=anchors_k, anchors_v=anchors_v,
            residuals_k=res_k_q, residuals_v=res_v_q,
            scales_k=scales_k, scales_v=scales_v,
            layer_anchor_map=layer_anchor_map, pos_anchor_map=pos_anchor_map,
            anchor_layers=anchor_layers, anchor_positions=anchor_positions,
            n_layers=n_layers, n_heads=H, seq_len=T, head_dim=D,
            norm_k=norm_k, norm_v=norm_v,
            predictor_used=predictor is not None,
        )

    def decompress(
        self,
        c: CompressedJoint,
        predictor=None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        anchor_exp_k = c.anchors_k.float()[c.layer_anchor_map][:, c.pos_anchor_map]
        anchor_exp_v = c.anchors_v.float()[c.layer_anchor_map][:, c.pos_anchor_map]

        res_k = self._dequantize(c.residuals_k, c.scales_k)
        res_v = self._dequantize(c.residuals_v, c.scales_v)

        if predictor is not None and c.predictor_used:
            nearest_l = torch.tensor(
                [c.anchor_layers[c.layer_anchor_map[l].item()]
                 for l in range(c.n_layers)], dtype=torch.float
            )
            nearest_p = torch.tensor(
                [c.anchor_positions[c.pos_anchor_map[p].item()]
                 for p in range(c.seq_len)], dtype=torch.float
            )
            delta_l = torch.arange(c.n_layers, dtype=torch.float) - nearest_l
            delta_p = torch.arange(c.seq_len, dtype=torch.float) - nearest_p
            pred_k, pred_v = predictor.predict_joint(
                anchor_exp_k, anchor_exp_v, delta_l, delta_p
            )
            k = pred_k + res_k
            v = pred_v + res_v
        else:
            k = anchor_exp_k + res_k   # [L, T, H, D]
            v = anchor_exp_v + res_v

        if c.norm_k is not None:
            k = k * c.norm_k.unsqueeze(-1)
            v = v * c.norm_v.unsqueeze(-1)

        # Return list of [H, T, D] per layer
        k = k.permute(0, 2, 1, 3)   # [L, H, T, D]
        v = v.permute(0, 2, 1, 3)
        return [k[l] for l in range(c.n_layers)], [v[l] for l in range(c.n_layers)]

    def metrics(
        self,
        all_keys_orig: List[torch.Tensor],
        all_values_orig: List[torch.Tensor],
        c: CompressedJoint,
        predictor=None,
    ) -> Dict:
        k_recs, v_recs = self.decompress(c, predictor)
        D = c.head_dim
        cos_k = cos_v = mse_k = mse_v = 0.0
        n = len(all_keys_orig)
        for l in range(n):
            k_o = all_keys_orig[l].float();  v_o = all_values_orig[l].float()
            k_r = k_recs[l].float();         v_r = v_recs[l].float()
            cos_k += F.cosine_similarity(k_o.reshape(-1,D), k_r.reshape(-1,D), dim=-1).mean().item()
            cos_v += F.cosine_similarity(v_o.reshape(-1,D), v_r.reshape(-1,D), dim=-1).mean().item()
            mse_k += F.mse_loss(k_r, k_o).item()
            mse_v += F.mse_loss(v_r, v_o).item()
        return {
            'strategy':            'joint_2d',
            'seq_anchor_stride':   self.cfg.seq_anchor_stride,
            'layer_anchor_stride': self.cfg.layer_anchor_stride,
            'cosine_sim_k':        cos_k / n,
            'cosine_sim_v':        cos_v / n,
            'mse_k':               mse_k / n,
            'mse_v':               mse_v / n,
            'compression_ratio':   c.compression_ratio(),
            'bytes_saved_pct':     (1 - 1/c.compression_ratio()) * 100,
            'predictor_used':      c.predictor_used,
        }
