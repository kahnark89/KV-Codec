# KV-Codec

Adaptive KV cache compression for transformer models. Automatically detects which compression axes are viable for a given model, selects the best codec, and reports quality vs. compression tradeoffs.

## How It Works

Transformer KV caches are redundant along two axes:

- **Sequence axis** — adjacent token positions have similar keys/values (temporal locality)
- **Layer axis** — adjacent layers produce correlated KV representations (depth locality)

KV-Codec measures both axes at runtime, then routes to the best strategy:

| Strategy | When selected | Typical ratio |
|---|---|---|
| `SEQ_ONLY` | Sequence axis compressible, layer axis not | 2–8× |
| `LAYER_ONLY` | Layer axis compressible, sequence axis not | 2–4× |
| `JOINT_2D` | Both axes compressible | 4–16× |
| `NONE` | Neither axis compressible | — |

All codecs use I-frame/P-frame style compression: anchor positions store full vectors, non-anchors store INT8 quantized residuals from the nearest anchor.

## Installation

```bash
pip install .
```

Or for development:

```bash
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, Transformers ≥ 4.35

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvcodec import KVSystem

model     = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

system  = KVSystem(model, tokenizer, device="cpu")
profile = system.detect()        # measure both axes
results = system.benchmark()     # sweep all viable strategies
system.print_report()            # print full summary + recommendation
```

### Use a specific codec directly

```python
from kvcodec import SeqCodec, SeqCodecConfig

codec      = SeqCodec(SeqCodecConfig(anchor_stride=8))
compressed = codec.compress(keys, values)   # keys/values: [H, T, D] tensors
k_rec, v_rec = codec.decompress(compressed)
metrics    = codec.metrics(keys, values, compressed)
# {'cosine_sim_k': 0.9973, 'compression_ratio': 4.2, 'bytes_saved_pct': 76.2, ...}
```

### Benchmark throughput impact

```bash
python benchmark_throughput.py
python benchmark_throughput.py --model facebook/opt-350m --max-new-tokens 200
python benchmark_throughput.py --model meta-llama/Llama-2-7b-hf --kv-seq-len 1024
```

## Codecs

### SeqCodec — Sequence axis
Anchors placed every `anchor_stride` tokens. Non-anchor positions store INT8 quantized residuals from the nearest anchor. Optional L2 normalization before residual computation.

```python
from kvcodec import SeqCodec, SeqCodecConfig

codec = SeqCodec(SeqCodecConfig(
    anchor_stride=16,       # 1 anchor per 16 tokens
    quantize_bits=8,
    normalize_residuals=True,
))
```

### LayerCodec — Layer axis
Anchor layers store full KV tensors. Non-anchor layers store residuals from the nearest anchor layer.

```python
from kvcodec import LayerCodec, LayerCodecConfig

codec = LayerCodec(LayerCodecConfig(
    anchor_layer_stride=4,  # 1 anchor per 4 layers
    quantize_bits=8,
))
```

### JointCodec — 2D joint (best compression)
Operates on both axes simultaneously. Anchors placed at `(anchor_layer, anchor_position)` grid intersections. Residuals computed from the nearest 2D anchor, giving smaller residuals and better quantization.

```python
from kvcodec import JointCodec, JointCodecConfig

codec = JointCodec(JointCodecConfig(
    seq_anchor_stride=16,
    layer_anchor_stride=4,
    quantize_bits=8,
))
```

### KVPredictor — Optional MLP predictor
A lightweight MLP (15–50K params) that predicts KV vectors from anchor + positional delta, reducing residuals before quantization. Train it on your target domain for best results.

```python
from kvcodec import KVPredictor, PredictorTrainer

predictor = KVPredictor(head_dim=64)
trainer   = PredictorTrainer(predictor, mode='joint')
trainer.train(all_keys, all_values, epochs=10)
trainer.save("predictor.pt")
```

## Adaptive Selection

```python
from kvcodec import KVDetector, select_codec

detector = KVDetector(max_seq_len=128)
profile  = detector.detect(model, tokenizer)

print(profile.strategy)           # CodecStrategy.JOINT_2D
print(profile.layer_sim_adj)      # 0.842
print(profile.seq_sim_adj)        # 0.671
print(profile.estimated_compression_ratio())  # 6.3

codec = select_codec(profile)     # returns the right codec, pre-configured
```

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `anchor_stride` | 16 | Token positions between seq anchors |
| `layer_anchor_stride` | 4 | Layers between layer anchors |
| `quantize_bits` | 8 | Residual quantization bits |
| `normalize_residuals` | True | L2-normalize before residual computation |
| `use_predictor` | False | Use MLP predictor to reduce residuals |

Global thresholds (in `kvcodec/config.py`):

| Constant | Value | Description |
|---|---|---|
| `LAYER_AXIS_THRESHOLD` | 0.70 | Min layer similarity to enable layer compression |
| `SEQ_AXIS_THRESHOLD` | 0.50 | Min seq similarity to enable seq compression |
| `JOINT_BONUS_THRESHOLD` | 0.05 | Min joint bonus to prefer JOINT_2D over single-axis |

## Testing

```bash
# Fast unit tests (no model required, ~seconds)
pytest tests/ -m "not slow"

# Include integration tests (downloads opt-125m, ~minutes)
pytest tests/ -m slow

# Full standalone test script
python test_system.py
```

## Project Structure

```
kvcodec/
├── config.py        # thresholds, enums, codec config dataclasses
├── detector.py      # adaptive axis detection
├── codec_seq.py     # sequence-axis codec
├── codec_layer.py   # layer-axis codec
├── codec_joint.py   # 2D joint codec
├── predictor.py     # MLP motion predictor + trainer
├── selector.py      # codec routing and Pareto sweep
└── system.py        # top-level orchestrator
benchmark_throughput.py   # wall-clock + memory impact benchmark
test_system.py            # full pipeline integration test
tests/                    # pytest suite
examples/                 # usage examples
```
