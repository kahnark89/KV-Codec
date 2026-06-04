"""
kvcodec
=======
Adaptive KV cache codec framework.

Public API:
    from kvcodec import KVSystem

    system = KVSystem(model, tokenizer)
    profile = system.detect()          # measure both axes
    results = system.benchmark()       # run all applicable codecs
    system.print_report()              # full summary
"""

from .config     import (CodecStrategy, CompressibilityProfile,
                          SeqCodecConfig, LayerCodecConfig, JointCodecConfig,
                          LAYER_AXIS_THRESHOLD, SEQ_AXIS_THRESHOLD)
from .detector   import KVDetector
from .codec_seq  import SeqCodec, CompressedSeqLayer
from .codec_layer import LayerCodec, CompressedLayerStack
from .codec_joint import JointCodec, CompressedJoint
from .selector   import (select_codec, sweep_strategies,
                          compress_cache, decompress_cache)
from .predictor  import KVPredictor, PredictorTrainer
from .system     import KVSystem

__all__ = [
    'KVSystem',
    'KVDetector',
    'SeqCodec', 'LayerCodec', 'JointCodec',
    'KVPredictor', 'PredictorTrainer',
    'select_codec', 'sweep_strategies',
    'compress_cache', 'decompress_cache',
    'CodecStrategy', 'CompressibilityProfile',
    'SeqCodecConfig', 'LayerCodecConfig', 'JointCodecConfig',
]
