#!/usr/bin/env python3
"""
test_system.py
==============
Full pipeline test for the kvcodec system.
Runs on opt-125m (already cached) — no GPU, no downloads needed.

Tests:
  1. Detection — both axes measured, strategy selected
  2. Seq codec  — compress/decompress, quality metrics
  3. Layer codec — compress/decompress, quality metrics
  4. Joint codec — compress/decompress, quality metrics
  5. Sweep      — Pareto frontier across all configs
  6. System API — end-to-end through KVSystem

Run:
    python test_system.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from kvcodec import (
    KVSystem, KVDetector,
    SeqCodec, LayerCodec, JointCodec,
    SeqCodecConfig, LayerCodecConfig, JointCodecConfig,
    CodecStrategy, select_codec,
)

MODEL_ID = 'facebook/opt-125m'
DEVICE   = 'cpu'

TEST_TEXTS = [
    "The transformer architecture uses self-attention to process sequences. "
    "Each layer applies a residual transformation, accumulating semantic information "
    "through depth. Key-value caches store intermediate computations.",
    "Compression algorithms exploit statistical redundancy. Video codecs use temporal "
    "prediction to encode differences between frames rather than full frames. "
    "The same principle applies to transformer key-value caches.",
    "Machine learning inference is memory bandwidth bound at the scales required "
    "for production deployment. Reducing the KV cache footprint directly translates "
    "to higher throughput without changes to model weights.",
]


def load_model():
    print(f"\nLoading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    print(f"Loaded. {sum(p.numel() for p in model.parameters())/1e6:.1f}M params\n")
    return model, tok


def extract_kv(model, tok, text, max_len=256):
    enc = tok(text, return_tensors='pt', max_length=max_len,
              truncation=True, padding=False)
    input_ids = enc['input_ids']
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
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
    return kl, vl


def hr(title=''):
    print(('\n' if title else '') + '─'*60 + (f'  {title}' if title else ''))


def test_detection(model, tok):
    hr('TEST 1: Axis Detection')
    detector = KVDetector(
        probe_texts=TEST_TEXTS,
        max_seq_len=128,
        device=DEVICE,
        verbose=True,
    )
    profile = detector.detect(model, tok, model_id=MODEL_ID)
    print(f"\nStrategy selected: {profile.strategy.value.upper()}")
    print(f"Est. compression:  {profile.estimated_compression_ratio():.1f}x")
    print(f"Layer compressible: {profile.layer_compressible}")
    print(f"Seq compressible:   {profile.seq_compressible}")
    print(f"Joint coupling:     {profile.joint_coupling}")
    return profile


def test_seq_codec(all_keys, all_values):
    hr('TEST 2: Sequence-Axis Codec')
    for stride in [4, 8, 16, 32]:
        codec = SeqCodec(SeqCodecConfig(anchor_stride=stride))
        layer_metrics = []
        for l in range(len(all_keys)):
            c = codec.compress(all_keys[l], all_values[l])
            m = codec.metrics(all_keys[l], all_values[l], c)
            layer_metrics.append(m)
        avg_cos_k = sum(m['cosine_sim_k'] for m in layer_metrics) / len(layer_metrics)
        avg_ratio = sum(m['compression_ratio'] for m in layer_metrics) / len(layer_metrics)
        print(f"  stride={stride:3d}  "
              f"cosine_sim_k={avg_cos_k:.4f}  "
              f"ratio={avg_ratio:.2f}x  "
              f"anchors={len(codec._anchor_positions(all_keys[0].shape[1]))}")


def test_layer_codec(all_keys, all_values):
    hr('TEST 3: Layer-Axis Codec')
    for stride in [2, 4]:
        codec = LayerCodec(LayerCodecConfig(anchor_layer_stride=stride))
        c = codec.compress(all_keys, all_values)
        m = codec.metrics(all_keys, all_values, c)
        print(f"  layer_stride={stride}  "
              f"cosine_sim_k={m['cosine_sim_k']:.4f}  "
              f"cosine_sim_v={m['cosine_sim_v']:.4f}  "
              f"ratio={m['compression_ratio']:.2f}x  "
              f"anchors={len(c.anchor_layers)}/{c.n_layers} layers")


def test_joint_codec(all_keys, all_values):
    hr('TEST 4: Joint 2D Codec')
    for sl, sp in [(2, 8), (4, 16), (4, 32)]:
        codec = JointCodec(JointCodecConfig(
            layer_anchor_stride=sl, seq_anchor_stride=sp
        ))
        c = codec.compress(all_keys, all_values)
        m = codec.metrics(all_keys, all_values, c)
        print(f"  layer_stride={sl} seq_stride={sp:2d}  "
              f"cosine_sim_k={m['cosine_sim_k']:.4f}  "
              f"ratio={m['compression_ratio']:.2f}x  "
              f"saved={m['bytes_saved_pct']:.1f}%")


def test_reconstruction_correctness(all_keys, all_values):
    hr('TEST 5: Reconstruction Correctness')
    # Check that compress → decompress → original for stride=1 (lossless except quant)
    codec = SeqCodec(SeqCodecConfig(anchor_stride=1, normalize_residuals=False))
    layer_0_k, layer_0_v = all_keys[0], all_values[0]
    c = codec.compress(layer_0_k, layer_0_v)
    k_rec, v_rec = codec.decompress(c)
    import torch.nn.functional as F
    D = layer_0_k.shape[-1]
    cos = F.cosine_similarity(
        layer_0_k.reshape(-1, D).float(),
        k_rec.reshape(-1, D).float(),
        dim=-1
    ).mean().item()
    print(f"  stride=1 (anchor every token): cosine_sim_k={cos:.6f}  "
          f"(expect ~1.0 — only quantization error)")
    assert cos > 0.995, f"Reconstruction error too high: {cos}"
    print("  ✓ Reconstruction correctness verified")


def test_system_api(model, tok):
    hr('TEST 6: KVSystem End-to-End API')
    system = KVSystem(model, tok, device=DEVICE, model_id=MODEL_ID, verbose=False)
    profile = system.detect(probe_texts=TEST_TEXTS, max_seq_len=128)
    results = system.benchmark(texts=TEST_TEXTS[:2], max_seq_len=128)
    system.print_report()
    print(f"  Benchmark produced {len(results)} strategy results")
    return results


def main():
    print("="*60)
    print("kvcodec — Full System Test")
    print(f"Model: {MODEL_ID}  Device: {DEVICE}")
    print("="*60)

    model, tok = load_model()

    # Extract KV from first test text for direct codec tests
    all_keys, all_values = extract_kv(model, tok, TEST_TEXTS[0], max_len=256)
    L, H, T, D = len(all_keys), all_keys[0].shape[0], all_keys[0].shape[1], all_keys[0].shape[2]
    print(f"Extracted KV: {L} layers × {H} heads × {T} tokens × {D} head_dim")

    # Run all tests
    profile = test_detection(model, tok)
    test_seq_codec(all_keys, all_values)
    test_layer_codec(all_keys, all_values)
    test_joint_codec(all_keys, all_values)
    test_reconstruction_correctness(all_keys, all_values)
    test_system_api(model, tok)

    print("\n" + "="*60)
    print("All tests passed.")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
