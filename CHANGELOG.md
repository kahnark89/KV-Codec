# Changelog

## [0.1.0] - 2026-06-03

### Added
- `KVDetector`: adaptive compressibility measurement across layer and sequence axes
- `SeqCodec`: sequence-axis I-frame/P-frame compression with INT8 quantization
- `LayerCodec`: layer-axis cross-layer delta compression
- `JointCodec`: 2D joint codec operating on both axes simultaneously
- `KVPredictor`: optional lightweight MLP (15–50K params) for motion vector prediction
- `PredictorTrainer`: full training loop with Adam + cosine annealing LR
- `select_codec()`: adaptive codec routing based on measured similarity profile
- `sweep_strategies()`: Pareto frontier sweep across all stride configurations
- `KVSystem`: top-level orchestrator (detect → select → compress → benchmark → report)
- `benchmark_throughput.py`: wall-clock and memory impact benchmarking script
