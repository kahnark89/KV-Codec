"""
kvcodec/_residual.py
====================
Shared residual-processing helpers used by all codecs.
"""

import torch


def _clip_per_vector(res: torch.Tensor, k: float) -> torch.Tensor:
    """
    Per-vector outlier clip.

    Clamps each KV vector's residual elements to ``±k · std`` where ``std`` is
    that vector's own spread along the head-dim (the last axis). Using a
    per-vector threshold instead of a single global std keeps outlier
    suppression while preserving high-energy vectors, which a global clamp
    would otherwise truncate (destroying signal and biasing the subsequent
    per-vector quantizer).
    """
    lim = k * res.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.minimum(torch.maximum(res, -lim), lim)
