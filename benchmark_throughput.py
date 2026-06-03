"""
benchmark_throughput.py
=======================
Measures whether KV-Codec compression gains translate to real throughput
improvements. Reports baseline generation speed, raw KV cache size,
compression ratio + quality, codec latency overhead, and a net verdict.

Usage:
    python benchmark_throughput.py
    python benchmark_throughput.py --model facebook/opt-350m --max-new-tokens 200
    python benchmark_throughput.py --model meta-llama/Llama-2-7b-hf --runs 5
"""

import argparse
import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvcodec import KVSystem
from kvcodec.selector import select_codec, sweep_strategies


# ── helpers ───────────────────────────────────────────────────────────────────

def _hr(char="─", width=65):
    print(char * width)

def _peak_mem_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    return 0.0

def _reset_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def _kv_bytes(past_key_values) -> int:
    """Return total bytes across all KV tensors regardless of cache format."""
    tensors = []
    pkv = past_key_values
    if hasattr(pkv, 'layers') and hasattr(pkv.layers[0], 'keys'):
        for l in pkv.layers:
            tensors += [l.keys, l.values]
    elif hasattr(pkv, 'key_cache'):
        for l in range(len(pkv.key_cache)):
            tensors += [pkv.key_cache[l], pkv.value_cache[l]]
    else:
        for layer in pkv:
            tensors += list(layer)
    return sum(t.nelement() * t.element_size() for t in tensors)


# ── main benchmark ─────────────────────────────────────────────────────────────

def run(
    model_name: str = "facebook/opt-125m",
    prompt: str = "Transformer models use a key-value cache to store attention states "
                  "across decoding steps, allowing each new token to attend to all prior tokens.",
    max_new_tokens: int = 100,
    n_runs: int = 3,
    kv_seq_len: int = 256,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nKV-Codec Throughput Benchmark")
    _hr("=")
    print(f"  Model:           {model_name}")
    print(f"  Device:          {device}")
    print(f"  Prompt tokens:   (auto)")
    print(f"  Max new tokens:  {max_new_tokens}")
    print(f"  Runs (averaged): {n_runs}")
    _hr("=")

    # ── load ──────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]
    print(f"       Prompt tokens: {prompt_tokens}")

    # ── baseline generation ───────────────────────────────────────────────────
    print(f"\n[2/5] Baseline generation  ({n_runs} runs, no compression)")
    _hr()

    gen_times = []
    mem_peaks = []

    for i in range(n_runs):
        _reset_mem()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        t1 = time.perf_counter()
        gen_times.append(t1 - t0)
        mem_peaks.append(_peak_mem_gb())

    avg_time   = sum(gen_times) / n_runs
    avg_mem_gb = sum(mem_peaks) / n_runs
    tps        = max_new_tokens / avg_time

    print(f"  Avg generation time : {avg_time*1000:.1f} ms")
    print(f"  Throughput          : {tps:.1f} tokens/sec")
    if device == "cuda":
        print(f"  Peak GPU memory     : {avg_mem_gb:.3f} GB")
    else:
        print(f"  Peak GPU memory     : n/a (CPU run)")

    # ── raw KV cache size ─────────────────────────────────────────────────────
    print(f"\n[3/5] Raw KV cache size")
    _hr()

    enc = tokenizer(
        prompt, return_tensors="pt",
        max_length=kv_seq_len, truncation=True,
    )
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        fwd = model(input_ids, use_cache=True,
                    output_attentions=False, output_hidden_states=False)

    raw_bytes = _kv_bytes(fwd.past_key_values)
    raw_mb    = raw_bytes / 1024 ** 2

    # Per-token size (useful for projecting to longer contexts)
    t_len           = input_ids.shape[1]
    bytes_per_token = raw_bytes / t_len

    print(f"  Sequence length     : {t_len} tokens")
    print(f"  Raw KV cache size   : {raw_mb:.2f} MB  ({raw_bytes:,} bytes)")
    print(f"  Per-token KV cost   : {bytes_per_token/1024:.2f} KB/token")
    print(f"  Projected @ 4K tok  : {bytes_per_token * 4096 / 1024**2:.1f} MB")
    print(f"  Projected @ 32K tok : {bytes_per_token * 32768 / 1024**2:.1f} MB")

    del fwd
    _reset_mem()

    # ── compressibility + quality ─────────────────────────────────────────────
    print(f"\n[4/5] Compressibility detection & codec quality")
    _hr()

    system = KVSystem(model, tokenizer, device=device, verbose=False)
    profile = system.detect()

    print(f"  Recommended strategy: {profile.strategy.name}")
    print(f"  Layer similarity    : {profile.layer_sim_adj:.4f}  "
          f"({'compressible' if profile.layer_compressible else 'not compressible'})")
    print(f"  Seq similarity      : {profile.seq_sim_adj:.4f}  "
          f"({'compressible' if profile.seq_compressible else 'not compressible'})")

    results = system.benchmark(max_seq_len=kv_seq_len)

    viable = [r for r in results if r.get("cosine_sim_k", 0) > 0.90]

    if not viable:
        print("\n  No viable compression strategy found (cosine_sim_k < 0.90 for all).")
        best = None
    else:
        best = viable[0]
        print(f"\n  Best viable strategy: {best['strategy']}")
        print(f"  Compression ratio   : {best['compression_ratio']:.2f}x")
        print(f"  Bytes saved         : {best.get('bytes_saved_pct', 0):.1f}%")
        print(f"  Cosine sim K        : {best['cosine_sim_k']:.4f}")
        print(f"  Cosine sim V        : {best.get('cosine_sim_v', 0):.4f}")
        print(f"  MSE K               : {best.get('mse_k', 0):.6f}")

    # ── codec latency overhead ────────────────────────────────────────────────
    print(f"\n[5/5] Codec latency overhead")
    _hr()

    all_keys, all_values, _ = system.extract_kv(prompt, max_seq_len=kv_seq_len)

    if best is None:
        print("  Skipped — no viable codec found.")
        compress_ms = decompress_ms = 0.0
    else:
        codec = select_codec(profile)

        # Warm-up
        _ = codec.compress(all_keys, all_values) if hasattr(codec, 'compress') else None

        # Time compress
        OVERHEAD_RUNS = 10
        t0 = time.perf_counter()
        for _ in range(OVERHEAD_RUNS):
            compressed = codec.compress(all_keys, all_values)
        compress_ms = (time.perf_counter() - t0) / OVERHEAD_RUNS * 1000

        # Time decompress
        t0 = time.perf_counter()
        for _ in range(OVERHEAD_RUNS):
            _ = codec.decompress(compressed)
        decompress_ms = (time.perf_counter() - t0) / OVERHEAD_RUNS * 1000

        total_overhead_ms = compress_ms + decompress_ms
        overhead_pct      = (total_overhead_ms / (avg_time * 1000)) * 100

        print(f"  Compress time       : {compress_ms:.2f} ms")
        print(f"  Decompress time     : {decompress_ms:.2f} ms")
        print(f"  Total overhead      : {total_overhead_ms:.2f} ms")
        print(f"  Overhead vs baseline: {overhead_pct:.1f}% of {avg_time*1000:.1f} ms generation")

    # ── net verdict ───────────────────────────────────────────────────────────
    _hr("=")
    print("NET VERDICT")
    _hr("=")

    if best is None:
        print("  RESULT : INCOMPRESSIBLE")
        print("  REASON : No strategy achieves cosine_sim_k > 0.90.")
        print("           This model's KV cache is not redundant enough to compress safely.")

    else:
        ratio        = best["compression_ratio"]
        saved_pct    = best.get("bytes_saved_pct", 0)
        cosine_k     = best["cosine_sim_k"]
        total_ms     = compress_ms + decompress_ms
        overhead_pct = (total_ms / (avg_time * 1000)) * 100

        compressed_mb = raw_mb / ratio
        saved_mb      = raw_mb - compressed_mb

        print(f"  Compression ratio   : {ratio:.2f}x  ({saved_pct:.1f}% smaller)")
        print(f"  Reconstruction fidelity: cosine_sim_k = {cosine_k:.4f}")
        print(f"  Codec overhead      : {total_ms:.2f} ms  ({overhead_pct:.1f}% of generation)")
        print(f"  Memory freed        : {saved_mb:.2f} MB per {t_len}-token forward pass")
        print()

        if overhead_pct < 5 and ratio >= 2.0 and cosine_k >= 0.95:
            verdict = "STRONGLY BENEFICIAL"
            reason  = (f"High fidelity ({cosine_k:.4f}), {ratio:.1f}x compression, "
                       f"minimal overhead ({overhead_pct:.1f}%).")
        elif overhead_pct < 15 and ratio >= 1.5 and cosine_k >= 0.90:
            verdict = "BENEFICIAL"
            reason  = (f"Good compression ({ratio:.1f}x) with acceptable overhead "
                       f"({overhead_pct:.1f}%). Gains scale with sequence length.")
        elif overhead_pct < 30 and ratio >= 1.2:
            verdict = "MARGINAL — context-length dependent"
            reason  = (f"Overhead ({overhead_pct:.1f}%) is significant at this sequence "
                       f"length ({t_len} tok). Re-run with longer contexts — savings "
                       f"grow linearly while overhead stays roughly fixed.")
        else:
            verdict = "NOT BENEFICIAL at this sequence length"
            reason  = (f"Codec overhead ({overhead_pct:.1f}%) exceeds memory savings "
                       f"at {t_len} tokens. Consider longer contexts or larger batch sizes.")

        print(f"  RESULT  : {verdict}")
        print(f"  REASON  : {reason}")
        print()
        print(f"  NOTE: KV memory scales as O(n_layers × seq_len × n_heads × head_dim).")
        print(f"        At {t_len} tokens: {raw_mb:.1f} MB raw → {compressed_mb:.1f} MB compressed.")
        print(f"        At 4K  tokens: ~{bytes_per_token*4096/1024**2:.0f} MB raw → "
              f"~{bytes_per_token*4096/ratio/1024**2:.0f} MB compressed.")
        print(f"        At 32K tokens: ~{bytes_per_token*32768/1024**2:.0f} MB raw → "
              f"~{bytes_per_token*32768/ratio/1024**2:.0f} MB compressed.")

    _hr("=")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV-Codec throughput benchmark")
    parser.add_argument("--model",          default="facebook/opt-125m")
    parser.add_argument("--prompt",         default=None)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--runs",           type=int, default=3)
    parser.add_argument("--kv-seq-len",     type=int, default=256)
    args = parser.parse_args()

    kwargs = dict(
        model_name     = args.model,
        max_new_tokens = args.max_new_tokens,
        n_runs         = args.runs,
        kv_seq_len     = args.kv_seq_len,
    )
    if args.prompt:
        kwargs["prompt"] = args.prompt

    run(**kwargs)
