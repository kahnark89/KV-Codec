"""
kvcodec/selector.py
===================
Reads a CompressibilityProfile and instantiates the correct codec
with sensible default configs derived from the measured similarity values.

This is the routing layer that makes the system adaptive.
"""

from typing import Union

from .config import (
    CodecStrategy, CompressibilityProfile,
    SeqCodecConfig, LayerCodecConfig, JointCodecConfig,
)
from .codec_seq   import SeqCodec
from .codec_layer import LayerCodec
from .codec_joint import JointCodec


def _recommended_seq_stride(profile: CompressibilityProfile) -> int:
    """
    Derive anchor stride from sequence similarity decay.
    Faster decay → shorter stride (more anchors).
    Target: similarity at stride distance >= 0.70
    """
    s1  = profile.seq_sim_adj   # sim at Δp=1
    s8  = profile.seq_sim_8     # sim at Δp=8
    if s1 < 0.50:
        return 4    # very fast decay, short stride
    if s8 > 0.65:
        return 32   # slow decay, wide stride — fewer anchors
    if s8 > 0.55:
        return 16   # moderate decay
    return 8        # fast decay


def _recommended_layer_stride(profile: CompressibilityProfile) -> int:
    """
    Derive anchor layer stride from layer similarity decay.
    """
    s1 = profile.layer_sim_adj
    if s1 < 0.70:
        return 2    # just above threshold, short stride
    if s1 > 0.85:
        return 8    # very high sim, wide stride
    return 4        # standard


def select_codec(
    profile: CompressibilityProfile,
    force_strategy: CodecStrategy = None,
    quantize_bits: int = 8,
    use_predictor: bool = False,
) -> Union[SeqCodec, LayerCodec, JointCodec, None]:
    """
    Given a CompressibilityProfile, return the appropriate codec instance
    configured based on the measured similarity values.

    Returns None if strategy is NONE (no compression beneficial).
    """
    strategy = force_strategy or profile.strategy

    if strategy == CodecStrategy.NONE:
        print("[selector] No axis compressible — falling back to quantization only.")
        return None

    elif strategy == CodecStrategy.SEQ_ONLY:
        stride = _recommended_seq_stride(profile)
        print(f"[selector] Strategy: SEQ_ONLY  "
              f"(layer_sim={profile.layer_sim_adj:.3f} < {0.70}, "
              f"seq_sim={profile.seq_sim_adj:.3f})  "
              f"anchor_stride={stride}")
        return SeqCodec(SeqCodecConfig(
            anchor_stride=stride,
            quantize_bits=quantize_bits,
            use_predictor=use_predictor,
            normalize_residuals=True,
        ))

    elif strategy == CodecStrategy.LAYER_ONLY:
        stride = _recommended_layer_stride(profile)
        print(f"[selector] Strategy: LAYER_ONLY  "
              f"(layer_sim={profile.layer_sim_adj:.3f}, "
              f"seq_sim={profile.seq_sim_adj:.3f} < {0.50})  "
              f"layer_stride={stride}")
        return LayerCodec(LayerCodecConfig(
            anchor_layer_stride=stride,
            quantize_bits=quantize_bits,
            use_predictor=use_predictor,
            normalize_residuals=True,
        ))

    elif strategy == CodecStrategy.JOINT_2D:
        seq_stride   = _recommended_seq_stride(profile)
        layer_stride = _recommended_layer_stride(profile)
        coupling     = "learned" if (use_predictor and profile.joint_coupling) else "product"
        print(f"[selector] Strategy: JOINT_2D  "
              f"(layer_sim={profile.layer_sim_adj:.3f}, "
              f"seq_sim={profile.seq_sim_adj:.3f}, "
              f"joint_bonus={profile.joint_bonus:+.3f})  "
              f"seq_stride={seq_stride}  layer_stride={layer_stride}  "
              f"coupling={coupling}")
        return JointCodec(JointCodecConfig(
            seq_anchor_stride=seq_stride,
            layer_anchor_stride=layer_stride,
            quantize_bits=quantize_bits,
            use_predictor=use_predictor,
            normalize_residuals=True,
            coupling_mode=coupling,
        ))

    return None


def compress_cache(codec, all_keys, all_values, predictor=None):
    """
    Compress a full per-layer KV cache with any codec, hiding the
    per-codec input contract.

    SeqCodec operates on a single layer ([H, T, D]) at a time, so it is
    applied per layer and returns a *list* of compressed layers. LayerCodec
    and JointCodec consume the whole layer stack and return a single object.

    Returns (compressed, metrics) where ``metrics`` is a single averaged dict.
    """
    if isinstance(codec, SeqCodec):
        comps = [codec.compress(k, v, predictor)
                 for k, v in zip(all_keys, all_values)]
        ms = [codec.metrics(k, v, c, predictor)
              for k, v, c in zip(all_keys, all_values, comps)]
        return comps, _avg_metrics(ms)
    compressed = codec.compress(all_keys, all_values, predictor)
    return compressed, codec.metrics(all_keys, all_values, compressed, predictor)


def decompress_cache(codec, compressed, predictor=None):
    """
    Inverse of :func:`compress_cache`. Returns (all_keys, all_values) lists
    of per-layer [H, T, D] tensors regardless of codec type.
    """
    if isinstance(codec, SeqCodec):
        ks, vs = [], []
        for c in compressed:
            k, v = codec.decompress(c, predictor)
            ks.append(k)
            vs.append(v)
        return ks, vs
    return codec.decompress(compressed, predictor)


def sweep_strategies(
    profile: CompressibilityProfile,
    all_keys,
    all_values,
    strides_seq:   list = None,
    strides_layer: list = None,
) -> list:
    """
    Run all applicable strategies across a range of stride configs.
    Returns list of metric dicts sorted by compression ratio.
    Useful for finding the Pareto frontier of quality vs compression.
    """
    strides_seq   = strides_seq   or [4, 8, 16, 32, 64]
    strides_layer = strides_layer or [2, 4, 8]
    results = []

    # Always run seq if applicable
    if profile.seq_compressible:
        for s in strides_seq:
            codec = SeqCodec(SeqCodecConfig(anchor_stride=s))
            # Compress/measure each layer, average
            layer_results = []
            for l in range(len(all_keys)):
                c = codec.compress(all_keys[l], all_values[l])
                m = codec.metrics(all_keys[l], all_values[l], c)
                layer_results.append(m)
            avg = _avg_metrics(layer_results)
            avg['strategy'] = f'seq_stride_{s}'
            results.append(avg)

    # Run layer if applicable
    if profile.layer_compressible:
        for s in strides_layer:
            codec = LayerCodec(LayerCodecConfig(anchor_layer_stride=s))
            c = codec.compress(all_keys, all_values)
            m = codec.metrics(all_keys, all_values, c)
            m['strategy'] = f'layer_stride_{s}'
            results.append(m)

    # Run joint if both applicable
    if profile.layer_compressible and profile.seq_compressible:
        for sl in strides_layer:
            for sp in strides_seq:
                codec = JointCodec(JointCodecConfig(
                    seq_anchor_stride=sp, layer_anchor_stride=sl
                ))
                c = codec.compress(all_keys, all_values)
                m = codec.metrics(all_keys, all_values, c)
                m['strategy'] = f'joint_l{sl}_s{sp}'
                results.append(m)

    return sorted(results, key=lambda x: x.get('compression_ratio', 0), reverse=True)


def _avg_metrics(metric_list: list) -> dict:
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    avg = {}
    for k in keys:
        vals = [m[k] for m in metric_list if isinstance(m.get(k), (int, float))]
        avg[k] = sum(vals) / len(vals) if vals else metric_list[0].get(k)
    return avg
