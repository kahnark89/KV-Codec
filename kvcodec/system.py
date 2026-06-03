"""
kvcodec/system.py
=================
Top-level orchestrator. Single entry point for the full pipeline:
  detect → select → compress → benchmark → report
"""

import torch
from typing import List, Optional

from .detector  import KVDetector
from .selector  import select_codec, sweep_strategies
from .config    import CompressibilityProfile, CodecStrategy
from .predictor import KVPredictor, PredictorTrainer


class KVSystem:
    """
    Full KV codec pipeline for a given model.

    Usage:
        system = KVSystem(model, tokenizer, device='cuda')
        profile = system.detect()
        results = system.benchmark()
        system.print_report()
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cpu',
        model_id: str = "",
        verbose: bool = True,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device
        self.model_id  = model_id or getattr(model.config, '_name_or_path', 'unknown')
        self.verbose   = verbose

        self.profile: Optional[CompressibilityProfile] = None
        self.codec    = None
        self._kv_cache: dict = {}   # layer_idx → (keys, values) last extracted

    def _log(self, msg):
        if self.verbose:
            print(f"[KVSystem] {msg}")

    # ── Detection ─────────────────────────────────────────────

    def detect(
        self,
        probe_texts: List[str] = None,
        max_seq_len: int = 128,
    ) -> CompressibilityProfile:
        """
        Run the compressibility detector.
        Measures both axes, returns CompressibilityProfile with strategy.
        """
        self._log("Running compressibility detection...")
        detector = KVDetector(
            probe_texts=probe_texts,
            max_seq_len=max_seq_len,
            device=self.device,
            verbose=self.verbose,
        )
        self.profile = detector.detect(
            self.model, self.tokenizer, model_id=self.model_id
        )
        self._log(f"Detection complete. Strategy: {self.profile.strategy.value}")
        return self.profile

    # ── Codec selection ───────────────────────────────────────

    def select(
        self,
        force_strategy: CodecStrategy = None,
        use_predictor: bool = False,
    ):
        """Instantiate the codec recommended by the profile."""
        if self.profile is None:
            raise RuntimeError("Run detect() before select().")
        self.codec = select_codec(
            self.profile,
            force_strategy=force_strategy,
            use_predictor=use_predictor,
        )
        return self.codec

    # ── KV extraction ─────────────────────────────────────────

    def extract_kv(self, text: str, max_seq_len: int = 256):
        """
        Run one forward pass, return extracted KV tensors.
        Returns:
            all_keys:   list of [H, T, D] per layer
            all_values: same
            seq_len:    actual token count
        """
        enc = self.tokenizer(
            text, return_tensors='pt',
            max_length=max_seq_len, truncation=True, padding=False
        )
        input_ids = enc['input_ids'].to(self.device)
        T = input_ids.shape[1]

        with torch.no_grad():
            out = self.model(input_ids, use_cache=True,
                             output_attentions=False, output_hidden_states=False)

        pkv = out.past_key_values

        if hasattr(pkv, 'layers') and hasattr(pkv.layers[0], 'keys'):
            kl = [pkv.layers[l].keys.squeeze(0).cpu().float()  for l in range(len(pkv.layers))]
            vl = [pkv.layers[l].values.squeeze(0).cpu().float() for l in range(len(pkv.layers))]
        elif hasattr(pkv, 'key_cache'):
            kl = [pkv.key_cache[l].squeeze(0).cpu().float()  for l in range(len(pkv.key_cache))]
            vl = [pkv.value_cache[l].squeeze(0).cpu().float() for l in range(len(pkv.value_cache))]
        else:
            kl = [pkv[l][0].squeeze(0).cpu().float() for l in range(len(pkv))]
            vl = [pkv[l][1].squeeze(0).cpu().float() for l in range(len(pkv))]

        return kl, vl, T

    # ── Benchmark ─────────────────────────────────────────────

    def benchmark(
        self,
        texts: List[str] = None,
        max_seq_len: int = 256,
        strides_seq:   List[int] = None,
        strides_layer: List[int] = None,
    ) -> List[dict]:
        """
        Run the full strategy sweep across multiple texts.
        Returns list of metric dicts sorted by compression ratio.
        """
        if self.profile is None:
            self.detect()

        if texts is None:
            from .detector import PROBE_TEXTS
            texts = PROBE_TEXTS[:3]

        self._log(f"Benchmarking on {len(texts)} texts...")

        all_results = []
        for i, text in enumerate(texts):
            self._log(f"  text {i+1}/{len(texts)}")
            all_keys, all_values, T = self.extract_kv(text, max_seq_len)
            results = sweep_strategies(
                self.profile, all_keys, all_values,
                strides_seq=strides_seq,
                strides_layer=strides_layer,
            )
            all_results.append(results)

        # Average across texts
        if not all_results:
            return []

        strategies = [r['strategy'] for r in all_results[0]]
        averaged = []
        for strat in strategies:
            strat_results = [
                next((r for r in sample if r['strategy'] == strat), None)
                for sample in all_results
            ]
            strat_results = [r for r in strat_results if r is not None]
            if not strat_results:
                continue
            avg = {}
            for k in strat_results[0]:
                vals = [r[k] for r in strat_results if isinstance(r.get(k), (int, float))]
                avg[k] = sum(vals) / len(vals) if vals else strat_results[0].get(k)
            averaged.append(avg)

        self._benchmark_results = sorted(
            averaged, key=lambda x: x.get('compression_ratio', 0), reverse=True
        )
        return self._benchmark_results

    # ── Report ────────────────────────────────────────────────

    def print_report(self):
        """Print a full summary of detection results and benchmark metrics."""
        if self.profile is None:
            print("No profile yet — run detect() first.")
            return

        print("\n" + "="*65)
        print(self.profile.summary())

        if hasattr(self, '_benchmark_results') and self._benchmark_results:
            print("\n" + "-"*65)
            print(f"{'Strategy':<22} {'CosSim K':>9} {'CosSim V':>9} "
                  f"{'MSE K':>10} {'Ratio':>7} {'Saved':>7}")
            print("-"*65)
            for r in self._benchmark_results:
                print(
                    f"{r.get('strategy','?'):<22} "
                    f"{r.get('cosine_sim_k',0):>9.4f} "
                    f"{r.get('cosine_sim_v',0):>9.4f} "
                    f"{r.get('mse_k',0):>10.6f} "
                    f"{r.get('compression_ratio',0):>6.1f}x "
                    f"{r.get('bytes_saved_pct',0):>6.1f}%"
                )
            print("="*65)

            # Highlight the recommended operating point
            # Target: cosine_sim_k > 0.90 with highest compression ratio
            viable = [r for r in self._benchmark_results
                      if r.get('cosine_sim_k', 0) > 0.90]
            if viable:
                best = viable[0]
                print(f"\n  ★ Recommended: {best['strategy']}")
                print(f"    cosine_sim_k = {best['cosine_sim_k']:.4f}  "
                      f"ratio = {best['compression_ratio']:.1f}x  "
                      f"saved = {best['bytes_saved_pct']:.1f}%")
        print()
