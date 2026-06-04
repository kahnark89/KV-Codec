"""
kvcodec/config.py
=================
All configuration dataclasses, enums, and threshold constants.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ── Compressibility thresholds ────────────────────────────────

LAYER_AXIS_THRESHOLD  = 0.70   # surface[1,0] — adjacent layer similarity
SEQ_AXIS_THRESHOLD    = 0.50   # surface[0,1] — adjacent position similarity
JOINT_BONUS_THRESHOLD = 0.05   # extra sim on off-diagonal vs product prediction


class CodecStrategy(Enum):
    """Which compression axes are active."""
    NONE        = "none"         # neither axis compressible — quantization only
    SEQ_ONLY    = "seq_only"     # sequence axis only
    LAYER_ONLY  = "layer_only"   # layer axis only
    JOINT_2D    = "joint_2d"     # both axes, full 2D codec


@dataclass
class CompressibilityProfile:
    """
    Output of the detector. Describes the similarity structure
    of a specific model on a specific input distribution.
    """
    # Raw measurements
    layer_sim_adj: float      # surface[1,0]  — same pos, Δlayer=1
    layer_sim_mid: float      # surface[L//2, 0]
    layer_sim_decay: float    # how fast layer sim drops (fitted exp decay rate)

    seq_sim_adj: float        # surface[0,1]  — same layer, Δpos=1
    seq_sim_8: float          # surface[0,8]
    seq_sim_global: float     # mean similarity at large Δpos (global floor)

    joint_sim: float          # surface[1,1]  — off-diagonal
    joint_bonus: float        # joint_sim - layer_sim_adj * seq_sim_adj

    # Model metadata
    n_layers: int
    n_heads: int
    head_dim: int
    seq_len_measured: int
    model_id: str = ""

    # Derived recommendation
    @property
    def strategy(self) -> CodecStrategy:
        layer_ok = self.layer_sim_adj >= LAYER_AXIS_THRESHOLD
        seq_ok   = self.seq_sim_adj   >= SEQ_AXIS_THRESHOLD
        if layer_ok and seq_ok:
            # Both axes are individually compressible, but the heavier 2D
            # codec only pays off when the axes are genuinely *coupled* —
            # i.e. the off-diagonal (Δlayer=1, Δpos=1) similarity beats the
            # product of the two 1D similarities by JOINT_BONUS_THRESHOLD.
            # If they are separable, the better single axis wins.
            if self.joint_coupling:
                return CodecStrategy.JOINT_2D
            return self._better_single_axis()
        elif seq_ok:
            return CodecStrategy.SEQ_ONLY
        elif layer_ok:
            return CodecStrategy.LAYER_ONLY
        else:
            return CodecStrategy.NONE

    def _better_single_axis(self) -> CodecStrategy:
        """Pick the single-axis strategy with the higher estimated 1D ratio."""
        seq_ratio   = 2.0 + self.seq_sim_adj   * 4.0
        layer_ratio = 2.0 + self.layer_sim_adj * 3.0
        return (CodecStrategy.SEQ_ONLY if seq_ratio >= layer_ratio
                else CodecStrategy.LAYER_ONLY)

    @property
    def layer_compressible(self) -> bool:
        return self.layer_sim_adj >= LAYER_AXIS_THRESHOLD

    @property
    def seq_compressible(self) -> bool:
        return self.seq_sim_adj >= SEQ_AXIS_THRESHOLD

    @property
    def joint_coupling(self) -> bool:
        """True if joint 2D compression adds meaningful benefit over 1D."""
        return self.joint_bonus >= JOINT_BONUS_THRESHOLD

    def estimated_compression_ratio(self) -> float:
        """
        Rough compression ratio estimate based on similarity structure.
        Higher similarity → smaller residuals → better quantization.
        """
        if self.strategy == CodecStrategy.NONE:
            return 1.5   # quantization only, minimal benefit
        elif self.strategy == CodecStrategy.SEQ_ONLY:
            # Sequence axis: anchor every ~16 tokens, INT8 residuals
            # Effective ratio ~ 2–4x depending on similarity decay
            return 2.0 + self.seq_sim_adj * 4.0
        elif self.strategy == CodecStrategy.LAYER_ONLY:
            return 2.0 + self.layer_sim_adj * 3.0
        else:  # JOINT_2D
            return 4.0 + (self.layer_sim_adj + self.seq_sim_adj) * 3.0

    def summary(self) -> str:
        lines = [
            f"CompressibilityProfile — {self.model_id}",
            f"  Model:         {self.n_layers} layers × {self.n_heads} heads × {self.head_dim} dim",
            f"",
            f"  Layer axis:    sim(Δl=1) = {self.layer_sim_adj:.4f}  "
            f"{'✓ COMPRESSIBLE' if self.layer_compressible else '✗ weak'}  "
            f"(threshold {LAYER_AXIS_THRESHOLD})",
            f"  Seq axis:      sim(Δp=1) = {self.seq_sim_adj:.4f}  "
            f"{'✓ COMPRESSIBLE' if self.seq_compressible else '✗ weak'}  "
            f"(threshold {SEQ_AXIS_THRESHOLD})",
            f"  Joint bonus:   {self.joint_bonus:+.4f}  "
            f"{'✓ coupling present' if self.joint_coupling else '— separable'}",
            f"",
            f"  → Strategy:    {self.strategy.value.upper()}",
            f"  → Est. ratio:  {self.estimated_compression_ratio():.1f}x",
        ]
        return "\n".join(lines)


# ── Per-codec configs ─────────────────────────────────────────

@dataclass
class SeqCodecConfig:
    """Configuration for sequence-axis codec."""
    anchor_stride: int = 16           # I-frame every N tokens
    anchor_at_bos: bool = True
    quantize_bits: int = 8
    normalize_residuals: bool = True
    use_predictor: bool = False
    max_residual_clip: float = 3.0


@dataclass
class LayerCodecConfig:
    """Configuration for layer-axis codec."""
    anchor_layers: Optional[List[int]] = None  # None = auto (every N layers)
    anchor_layer_stride: int = 4
    quantize_bits: int = 8
    normalize_residuals: bool = True
    use_predictor: bool = False
    max_residual_clip: float = 3.0


@dataclass
class JointCodecConfig:
    """Configuration for 2D joint codec."""
    seq_anchor_stride: int = 16
    layer_anchor_stride: int = 4
    quantize_bits: int = 8
    normalize_residuals: bool = True
    use_predictor: bool = False
    coupling_mode: str = "product"    # "product" | "learned"
    max_residual_clip: float = 3.0
