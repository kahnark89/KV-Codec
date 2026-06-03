"""
kvcodec/detector.py
===================
Axis compressibility detector.

Runs a fast similarity sweep over a small sample of text,
measures the 2D similarity surface, and outputs a CompressibilityProfile
that drives codec strategy selection.

This is the novel adaptive piece — nothing in the literature does this.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple

from .config import CompressibilityProfile


# ── Default probe texts ───────────────────────────────────────
# Chosen to be representative of typical LLM input distributions.
# Short enough to run fast, diverse enough to average out anomalies.

PROBE_TEXTS = [
    "The transformer architecture processes tokens by computing attention weights "
    "across all positions simultaneously. Each layer applies a residual transformation "
    "to the representation, accumulating semantic information through depth.",

    "Machine learning inference requires careful management of memory bandwidth. "
    "During autoregressive generation, the key-value cache grows linearly with "
    "sequence length, creating a significant bottleneck in production deployments.",

    "Compression algorithms exploit statistical redundancy to reduce storage requirements. "
    "Video codecs use temporal prediction to encode differences between frames rather "
    "than storing each frame independently, achieving high compression ratios.",

    "The residual stream in deep neural networks accumulates small incremental updates "
    "through each layer. Adjacent layers exhibit high cosine similarity because the "
    "transformation applied at each step is small relative to the base representation.",

    "Natural language has hierarchical structure at multiple scales: phonemes, morphemes, "
    "words, phrases, sentences, paragraphs, and documents. Effective compression must "
    "respect this structure to maintain semantic coherence after reconstruction.",
]


class KVDetector:
    """
    Measures KV cache similarity structure and determines compression strategy.

    Fast path: ~2-5 seconds on CPU with a 125M model.
    Full path:  ~10-15 seconds on GPU with a 7B model.
    """

    def __init__(
        self,
        probe_texts: Optional[List[str]] = None,
        max_seq_len: int = 128,
        max_delta_pos: int = 32,
        device: str = 'cpu',
        verbose: bool = True,
    ):
        self.probe_texts  = probe_texts or PROBE_TEXTS
        self.max_seq_len  = max_seq_len
        self.max_delta_pos = max_delta_pos
        self.device       = device
        self.verbose      = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[detector] {msg}")

    # ── KV extraction ─────────────────────────────────────────

    def _extract_kv(
        self,
        model,
        tokenizer,
        text: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Run one forward pass, return stacked KV tensors.
        Returns:
            keys:   [n_layers, n_heads, seq_len, head_dim]
            values: same
            seq_len: actual token count
        """
        enc = tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
        )
        input_ids = enc['input_ids'].to(self.device)
        T = input_ids.shape[1]

        with torch.no_grad():
            out = model(input_ids, use_cache=True,
                        output_attentions=False, output_hidden_states=False)

        pkv = out.past_key_values

        # Handle all known cache formats
        if hasattr(pkv, 'layers') and hasattr(pkv.layers[0], 'keys'):
            key_list = [pkv.layers[l].keys.squeeze(0).cpu().float()
                        for l in range(len(pkv.layers))]
            val_list = [pkv.layers[l].values.squeeze(0).cpu().float()
                        for l in range(len(pkv.layers))]
        elif hasattr(pkv, 'key_cache'):
            key_list = [pkv.key_cache[l].squeeze(0).cpu().float()
                        for l in range(len(pkv.key_cache))]
            val_list = [pkv.value_cache[l].squeeze(0).cpu().float()
                        for l in range(len(pkv.value_cache))]
        else:
            key_list = [pkv[l][0].squeeze(0).cpu().float() for l in range(len(pkv))]
            val_list = [pkv[l][1].squeeze(0).cpu().float() for l in range(len(pkv))]

        keys   = torch.stack(key_list)    # [L, H, T, D]
        values = torch.stack(val_list)
        return keys, values, T

    # ── Similarity computations ───────────────────────────────

    def _head_avg_normalized(self, kv: torch.Tensor) -> torch.Tensor:
        """
        [L, H, T, D] → [L, T, D] head-averaged, L2-normalized.
        """
        k = kv.mean(dim=1)                           # [L, T, D]
        return F.normalize(k, p=2, dim=-1)

    def _layer_sim(self, k_norm: torch.Tensor, dl: int) -> float:
        """
        Mean cosine similarity between same position at layer offset dl.
        k_norm: [L, T, D]
        """
        L, T, D = k_norm.shape
        if dl >= L:
            return 0.0
        anchor = k_norm[:L-dl, :, :]   # [L-dl, T, D]
        target = k_norm[dl:,   :, :]
        return (anchor * target).sum(dim=-1).mean().item()

    def _seq_sim(self, k_norm: torch.Tensor, dp: int) -> float:
        """
        Mean cosine similarity between same layer at position offset dp.
        k_norm: [L, T, D]
        """
        L, T, D = k_norm.shape
        if dp >= T:
            return 0.0
        anchor = k_norm[:, :T-dp, :]
        target = k_norm[:, dp:,   :]
        return (anchor * target).sum(dim=-1).mean().item()

    def _joint_sim(self, k_norm: torch.Tensor, dl: int, dp: int) -> float:
        """
        Mean cosine similarity at joint offset (dl, dp).
        k_norm: [L, T, D]
        """
        L, T, D = k_norm.shape
        if dl >= L or dp >= T:
            return 0.0
        anchor = k_norm[:L-dl, :T-dp, :]
        target = k_norm[dl:,   dp:,   :]
        return (anchor * target).sum(dim=-1).mean().item()

    def _fit_decay_rate(self, profile: List[float]) -> float:
        """
        Fit an exponential decay: sim(d) ≈ sim(1) * exp(-rate * (d-1))
        Returns the decay rate (higher = faster decay = worse compression).
        """
        if len(profile) < 3:
            return 1.0
        import math
        vals = [max(v, 1e-6) for v in profile[1:5] if v > 1e-6]
        if len(vals) < 2:
            return 1.0
        # Linear regression on log values
        log_vals = [math.log(v) for v in vals]
        n = len(log_vals)
        xs = list(range(1, n+1))
        x_mean = sum(xs) / n
        y_mean = sum(log_vals) / n
        denom = sum((x - x_mean)**2 for x in xs)
        if denom < 1e-8:
            return 0.0
        slope = sum((xs[i] - x_mean) * (log_vals[i] - y_mean) for i in range(n)) / denom
        return -slope  # positive = decaying

    # ── Main detection ────────────────────────────────────────

    def detect(
        self,
        model,
        tokenizer,
        model_id: str = "",
    ) -> CompressibilityProfile:
        """
        Run the full compressibility detection sweep.
        Returns a CompressibilityProfile with strategy recommendation.
        """
        self._log(f"Starting detection sweep ({len(self.probe_texts)} probe texts)...")

        # Accumulators
        layer_sims_adj   = []
        layer_sims_mid   = []
        seq_sims_adj     = []
        seq_sims_8       = []
        seq_sims_global  = []
        joint_sims       = []
        layer_profiles_list = []
        seq_profiles_list   = []

        n_layers = n_heads = head_dim = seq_len_used = 0

        for i, text in enumerate(self.probe_texts):
            self._log(f"  probe {i+1}/{len(self.probe_texts)}...")
            keys, values, T = self._extract_kv(model, tokenizer, text)

            L = keys.shape[0]
            H = keys.shape[1]
            D = keys.shape[3]
            n_layers  = L
            n_heads   = H
            head_dim  = D
            seq_len_used = T

            k_norm = self._head_avg_normalized(keys)   # [L, T, D]

            # Layer axis: measure at dl = 1, 2, 4, L//2
            l_adj = self._layer_sim(k_norm, 1)
            l_mid = self._layer_sim(k_norm, max(1, L // 2))
            layer_profile = [self._layer_sim(k_norm, dl)
                             for dl in range(0, min(L, 8))]

            # Sequence axis: measure at dp = 1, 4, 8, 16, large
            s_adj    = self._seq_sim(k_norm, 1)
            s_8      = self._seq_sim(k_norm, min(8, T-1))
            s_global = self._seq_sim(k_norm, min(T//2, 32))
            seq_profile = [self._seq_sim(k_norm, dp)
                           for dp in range(0, min(T, self.max_delta_pos))]

            # Joint: off-diagonal (1,1)
            j_11 = self._joint_sim(k_norm, 1, 1)

            layer_sims_adj.append(l_adj)
            layer_sims_mid.append(l_mid)
            seq_sims_adj.append(s_adj)
            seq_sims_8.append(s_8)
            seq_sims_global.append(s_global)
            joint_sims.append(j_11)
            layer_profiles_list.append(layer_profile)
            seq_profiles_list.append(seq_profile)

            self._log(
                f"    layer_sim[1]={l_adj:.4f}  "
                f"seq_sim[1]={s_adj:.4f}  "
                f"joint[1,1]={j_11:.4f}"
            )

        # Average across probes
        def mean(lst): return sum(lst) / len(lst) if lst else 0.0

        l_adj_avg   = mean(layer_sims_adj)
        l_mid_avg   = mean(layer_sims_mid)
        s_adj_avg   = mean(seq_sims_adj)
        s_8_avg     = mean(seq_sims_8)
        s_global_avg = mean(seq_sims_global)
        j_11_avg    = mean(joint_sims)

        # Average profiles for decay fitting
        max_len_l = min(len(p) for p in layer_profiles_list)
        max_len_s = min(len(p) for p in seq_profiles_list)
        avg_layer_profile = [
            mean([p[i] for p in layer_profiles_list]) for i in range(max_len_l)
        ]
        avg_seq_profile = [
            mean([p[i] for p in seq_profiles_list]) for i in range(max_len_s)
        ]

        layer_decay = self._fit_decay_rate(avg_layer_profile)
        # Joint bonus: how much more similar is (1,1) vs product of (1,0)*(0,1)
        joint_bonus = j_11_avg - (l_adj_avg * s_adj_avg)

        profile = CompressibilityProfile(
            layer_sim_adj   = l_adj_avg,
            layer_sim_mid   = l_mid_avg,
            layer_sim_decay = layer_decay,
            seq_sim_adj     = s_adj_avg,
            seq_sim_8       = s_8_avg,
            seq_sim_global  = s_global_avg,
            joint_sim       = j_11_avg,
            joint_bonus     = joint_bonus,
            n_layers        = n_layers,
            n_heads         = n_heads,
            head_dim        = head_dim,
            seq_len_measured= seq_len_used,
            model_id        = model_id,
        )

        self._log("\n" + profile.summary())
        return profile

    def detect_from_kv(
        self,
        all_keys: List[List[torch.Tensor]],
        all_values: List[List[torch.Tensor]],
        model_id: str = "",
    ) -> CompressibilityProfile:
        """
        Alternative entry point if you already have extracted KV tensors.
        all_keys: list (samples) of list (layers) of [H, T, D] tensors
        """
        layer_sims_adj = []
        seq_sims_adj   = []
        seq_sims_8     = []
        seq_sims_global= []
        joint_sims     = []

        n_layers = n_heads = head_dim = seq_len_used = 0

        for sample_keys in all_keys:
            # Stack layers: [L, H, T, D]
            keys = torch.stack([k.float() for k in sample_keys])
            L, H, T, D = keys.shape
            n_layers = L; n_heads = H; head_dim = D; seq_len_used = T

            k_norm = self._head_avg_normalized(keys)

            layer_sims_adj.append(self._layer_sim(k_norm, 1))
            seq_sims_adj.append(self._seq_sim(k_norm, 1))
            seq_sims_8.append(self._seq_sim(k_norm, min(8, T-1)))
            seq_sims_global.append(self._seq_sim(k_norm, min(T//2, 32)))
            joint_sims.append(self._joint_sim(k_norm, 1, 1))

        def mean(lst): return sum(lst) / len(lst) if lst else 0.0

        l_adj = mean(layer_sims_adj)
        s_adj = mean(seq_sims_adj)
        j_11  = mean(joint_sims)

        profile = CompressibilityProfile(
            layer_sim_adj   = l_adj,
            layer_sim_mid   = l_adj * 0.5,  # rough estimate without L//2 data
            layer_sim_decay = 0.0,
            seq_sim_adj     = s_adj,
            seq_sim_8       = mean(seq_sims_8),
            seq_sim_global  = mean(seq_sims_global),
            joint_sim       = j_11,
            joint_bonus     = j_11 - l_adj * s_adj,
            n_layers        = n_layers,
            n_heads         = n_heads,
            head_dim        = head_dim,
            seq_len_measured= seq_len_used,
            model_id        = model_id,
        )

        if self.verbose:
            print(profile.summary())
        return profile
