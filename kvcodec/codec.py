"""
kvcodec/codec.py
================
Sequence-axis KV cache codec.

Architecture:
  - I-frames: anchor tokens stored at full precision (float16)
  - P-frames: INT8 quantized residuals relative to nearest anchor
  - Optional predictor: reduces residual magnitude before quantization

Operates per-layer, per-head independently.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

from ._residual import _clip_per_vector


@dataclass
class CodecConfig:
    """Configuration for the KV sequence codec."""
    anchor_stride: int = 16          # I-frame every N tokens
    anchor_at_bos: bool = True       # always anchor position 0
    quantize_bits: int = 8           # residual quantization (8 or 4)
    normalize_before_delta: bool = True   # L2-norm before delta (improves quant)
    predictor: Optional[object] = None   # KVPredictor instance, or None
    max_residual_clip: float = 3.0   # clip residuals at ±N std before quantization

    def anchor_positions(self, seq_len: int) -> List[int]:
        """Return sorted list of I-frame positions for a given sequence length."""
        anchors = set(range(0, seq_len, self.anchor_stride))
        if self.anchor_at_bos:
            anchors.add(0)
        return sorted(anchors)

    def nearest_anchor(self, pos: int, anchors: List[int]) -> int:
        """Return the index of the nearest anchor at or before pos."""
        # Causal: only look back — don't use future anchors
        candidates = [a for a in anchors if a <= pos]
        return candidates[-1] if candidates else anchors[0]


@dataclass
class CompressedKVLayer:
    """
    Compressed representation of KV tensors for one layer.

    Stores:
      anchors_k / anchors_v  : float16 [n_anchors, n_heads, head_dim]
      residuals_k / residuals_v: int8   [seq_len, n_heads, head_dim]
      scales_k / scales_v    : float32 [seq_len, n_heads]  — per-vector quant scale
      anchor_map             : int32   [seq_len]  — which anchor each position uses
      anchor_positions       : List[int]
      seq_len, n_heads, head_dim: shape info
    """
    anchors_k: torch.Tensor       # [n_anchors, n_heads, head_dim] fp16
    anchors_v: torch.Tensor
    residuals_k: torch.Tensor     # [seq_len, n_heads, head_dim] int8
    residuals_v: torch.Tensor
    scales_k: torch.Tensor        # [seq_len, n_heads] fp32
    scales_v: torch.Tensor
    anchor_map: torch.Tensor      # [seq_len] int32 → index into anchors
    anchor_positions: List[int]
    seq_len: int
    n_heads: int
    head_dim: int
    norm_factors_k: Optional[torch.Tensor] = None  # [seq_len, n_heads] if normalized
    norm_factors_v: Optional[torch.Tensor] = None

    def bytes_compressed(self) -> int:
        """Estimate compressed size in bytes."""
        anchor_bytes = (self.anchors_k.numel() + self.anchors_v.numel()) * 2  # fp16
        resid_bytes  = (self.residuals_k.numel() + self.residuals_v.numel()) * 1  # int8
        scale_bytes  = (self.scales_k.numel() + self.scales_v.numel()) * 4  # fp32
        meta_bytes   = self.anchor_map.numel() * 4
        return anchor_bytes + resid_bytes + scale_bytes + meta_bytes

    def bytes_original(self) -> int:
        """Original size if stored as float16."""
        return self.seq_len * self.n_heads * self.head_dim * 2 * 2  # K + V, fp16

    def compression_ratio(self) -> float:
        orig = self.bytes_original()
        comp = self.bytes_compressed()
        return orig / comp if comp > 0 else 0.0


class KVCodec:
    """
    Sequence-axis KV cache codec.

    Usage:
        codec = KVCodec(CodecConfig(anchor_stride=16))
        compressed = codec.compress(keys, values)   # per-layer tensors
        k_rec, v_rec = codec.decompress(compressed)
        sim = codec.reconstruction_quality(keys, values, compressed)
    """

    def __init__(self, config: CodecConfig = None):
        self.config = config or CodecConfig()

    # ── Quantization helpers ──────────────────────────────────

    def _quantize_int8(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-vector symmetric INT8 quantization.
        x: [..., D]
        Returns:
            x_q: [..., D] int8
            scales: [...] float32  (max absolute value / 127)
        """
        scales = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)  # [..., 1]
        x_q = (x / scales * 127).round().clamp(-128, 127).to(torch.int8)
        return x_q, scales.squeeze(-1).float()

    def _dequantize_int8(
        self, x_q: torch.Tensor, scales: torch.Tensor
    ) -> torch.Tensor:
        """
        x_q: [..., D] int8
        scales: [...] float32
        Returns: [..., D] float32
        """
        return x_q.float() * (scales.unsqueeze(-1) / 127)

    # ── Compress ──────────────────────────────────────────────

    def compress(
        self,
        keys: torch.Tensor,    # [n_heads, seq_len, head_dim]
        values: torch.Tensor,  # [n_heads, seq_len, head_dim]
        layer_idx: int = 0,
    ) -> CompressedKVLayer:
        """
        Compress a single layer's KV tensors.
        keys/values: [n_heads, seq_len, head_dim]
        """
        n_heads, seq_len, head_dim = keys.shape
        cfg = self.config

        # Rearrange to [seq_len, n_heads, head_dim] for easier indexing
        k = keys.permute(1, 0, 2).float()    # [T, H, D]
        v = values.permute(1, 0, 2).float()

        anchor_positions = cfg.anchor_positions(seq_len)
        n_anchors = len(anchor_positions)

        # ── Optional: normalize per-vector before delta ───────
        norm_k = norm_v = None
        if cfg.normalize_before_delta:
            norm_k = k.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [T, H, 1]
            norm_v = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            k = k / norm_k
            v = v / norm_v
            norm_k = norm_k.squeeze(-1)  # [T, H]
            norm_v = norm_v.squeeze(-1)

        # ── Extract anchors ───────────────────────────────────
        anchor_idx = torch.tensor(anchor_positions, dtype=torch.long)
        anchors_k = k[anchor_idx].half()   # [n_anchors, H, D]
        anchors_v = v[anchor_idx].half()

        # ── Build anchor map (each position → anchor array index) ─
        anchor_map = torch.zeros(seq_len, dtype=torch.long)
        anchor_set_map = {pos: i for i, pos in enumerate(anchor_positions)}
        for t in range(seq_len):
            nearest = cfg.nearest_anchor(t, anchor_positions)
            anchor_map[t] = anchor_set_map[nearest]

        # ── Compute residuals ─────────────────────────────────
        # Expand anchors to full sequence via anchor_map
        anchor_expanded_k = anchors_k.float()[anchor_map]  # [T, H, D]
        anchor_expanded_v = anchors_v.float()[anchor_map]

        # Apply predictor if available
        if cfg.predictor is not None:
            anchor_positions_tensor = torch.tensor(anchor_positions, dtype=torch.long)
            pos_tensor = torch.arange(seq_len, dtype=torch.long)
            nearest_pos = torch.tensor(
                [anchor_positions[anchor_map[t].item()] for t in range(seq_len)],
                dtype=torch.long
            )
            delta_pos = (pos_tensor - nearest_pos).float()
            pred_k, pred_v = cfg.predictor.predict(
                anchor_expanded_k, anchor_expanded_v, delta_pos
            )
            residuals_k_raw = k - pred_k
            residuals_v_raw = v - pred_v
        else:
            residuals_k_raw = k - anchor_expanded_k  # [T, H, D]
            residuals_v_raw = v - anchor_expanded_v

        # Per-vector outlier clip before quantization (see _clip_per_vector):
        # threshold scales with each vector's own spread, not a global std.
        residuals_k_raw = _clip_per_vector(residuals_k_raw, cfg.max_residual_clip)
        residuals_v_raw = _clip_per_vector(residuals_v_raw, cfg.max_residual_clip)

        # Quantize residuals
        residuals_k_q, scales_k = self._quantize_int8(residuals_k_raw)
        residuals_v_q, scales_v = self._quantize_int8(residuals_v_raw)

        return CompressedKVLayer(
            anchors_k=anchors_k,
            anchors_v=anchors_v,
            residuals_k=residuals_k_q,
            residuals_v=residuals_v_q,
            scales_k=scales_k,
            scales_v=scales_v,
            anchor_map=anchor_map,
            anchor_positions=anchor_positions,
            seq_len=seq_len,
            n_heads=n_heads,
            head_dim=head_dim,
            norm_factors_k=norm_k,
            norm_factors_v=norm_v,
        )

    # ── Decompress ────────────────────────────────────────────

    def decompress(
        self,
        compressed: CompressedKVLayer,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct KV tensors from compressed representation.
        Returns:
            keys:   [n_heads, seq_len, head_dim] float32
            values: [n_heads, seq_len, head_dim] float32
        """
        cfg = self.config
        c = compressed

        # Expand anchors
        anchor_expanded_k = c.anchors_k.float()[c.anchor_map]  # [T, H, D]
        anchor_expanded_v = c.anchors_v.float()[c.anchor_map]

        # Dequantize residuals
        res_k = self._dequantize_int8(c.residuals_k, c.scales_k)  # [T, H, D]
        res_v = self._dequantize_int8(c.residuals_v, c.scales_v)

        # Apply predictor if available
        if cfg.predictor is not None:
            pos_tensor = torch.arange(c.seq_len, dtype=torch.long)
            nearest_pos = torch.tensor(
                [c.anchor_positions[c.anchor_map[t].item()] for t in range(c.seq_len)],
                dtype=torch.long
            )
            delta_pos = (pos_tensor - nearest_pos).float()
            pred_k, pred_v = cfg.predictor.predict(
                anchor_expanded_k, anchor_expanded_v, delta_pos
            )
            k_rec = pred_k + res_k
            v_rec = pred_v + res_v
        else:
            k_rec = anchor_expanded_k + res_k  # [T, H, D]
            v_rec = anchor_expanded_v + res_v

        # Undo normalization
        if c.norm_factors_k is not None:
            k_rec = k_rec * c.norm_factors_k.unsqueeze(-1)
            v_rec = v_rec * c.norm_factors_v.unsqueeze(-1)

        # Return [n_heads, seq_len, head_dim]
        return k_rec.permute(1, 0, 2), v_rec.permute(1, 0, 2)

    # ── Quality metrics ───────────────────────────────────────

    def reconstruction_quality(
        self,
        keys_orig: torch.Tensor,
        values_orig: torch.Tensor,
        compressed: CompressedKVLayer,
    ) -> Dict[str, float]:
        """
        Compute reconstruction quality metrics.
        Returns dict with cosine_sim_k, cosine_sim_v, mse_k, mse_v,
        compression_ratio, bytes_saved_pct.
        """
        k_rec, v_rec = self.decompress(compressed)

        # Move to same device/dtype for comparison
        k_orig = keys_orig.float()
        v_orig = values_orig.float()
        k_rec  = k_rec.float().to(k_orig.device)
        v_rec  = v_rec.float().to(v_orig.device)

        # Cosine similarity (per vector, then mean)
        cos_k = F.cosine_similarity(
            k_orig.reshape(-1, compressed.head_dim),
            k_rec.reshape(-1, compressed.head_dim),
            dim=-1
        ).mean().item()

        cos_v = F.cosine_similarity(
            v_orig.reshape(-1, compressed.head_dim),
            v_rec.reshape(-1, compressed.head_dim),
            dim=-1
        ).mean().item()

        mse_k = F.mse_loss(k_rec, k_orig).item()
        mse_v = F.mse_loss(v_rec, v_orig).item()

        cr = compressed.compression_ratio()
        saved_pct = (1 - 1/cr) * 100 if cr > 0 else 0

        return {
            'cosine_sim_k': cos_k,
            'cosine_sim_v': cos_v,
            'mse_k': mse_k,
            'mse_v': mse_v,
            'compression_ratio': cr,
            'bytes_saved_pct': saved_pct,
            'anchor_stride': self.config.anchor_stride,
            'seq_len': compressed.seq_len,
            'n_anchors': len(compressed.anchor_positions),
        }

    def sweep_anchor_strides(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        strides: List[int] = None,
    ) -> List[Dict]:
        """
        Run compression at multiple anchor strides.
        Returns list of quality dicts sorted by stride.
        Useful for finding the compression/quality tradeoff curve.
        """
        if strides is None:
            strides = [1, 2, 4, 8, 16, 32, 64]
        results = []
        for stride in strides:
            cfg = CodecConfig(
                anchor_stride=stride,
                normalize_before_delta=self.config.normalize_before_delta,
                quantize_bits=self.config.quantize_bits,
            )
            codec = KVCodec(cfg)
            compressed = codec.compress(keys, values)
            metrics = codec.reconstruction_quality(keys, values, compressed)
            results.append(metrics)
        return results
