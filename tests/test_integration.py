"""
Integration tests — load opt-125m and run the full pipeline.
Skipped unless -m slow is passed (or no marker filter).
"""
import pytest

from kvcodec import KVSystem, KVDetector, SeqCodec, SeqCodecConfig


MODEL_ID   = "facebook/opt-125m"
TEST_TEXT  = (
    "The transformer architecture uses self-attention to process sequences. "
    "Key-value caches store intermediate computations across decoding steps."
)


def _extract_kv(model, tok, text, max_len=128):
    # Reuse the production extractor so the test exercises the same code path
    # and handles every supported cache format (DynamicCache, legacy tuple, etc.).
    system = KVSystem(model, tok, device="cpu", verbose=False)
    kl, vl, _ = system.extract_kv(text, max_seq_len=max_len)
    return kl, vl


@pytest.mark.slow
class TestIntegration:
    def test_detector_returns_profile(self, real_model):
        model, tok = real_model
        detector = KVDetector(max_seq_len=64, verbose=False)
        profile = detector.detect(model, tok, model_id=MODEL_ID)
        assert profile is not None
        assert profile.n_layers > 0
        assert 0.0 <= profile.layer_sim_adj <= 1.0
        assert 0.0 <= profile.seq_sim_adj   <= 1.0

    def test_kvsystem_benchmark_returns_results(self, real_model):
        model, tok = real_model
        system = KVSystem(model, tok, device="cpu", verbose=False)
        system.detect(max_seq_len=64)
        results = system.benchmark(max_seq_len=64)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "cosine_sim_k" in results[0]
        assert "compression_ratio" in results[0]

    def test_seq_codec_on_real_kv(self, real_model):
        model, tok = real_model
        keys, values = _extract_kv(model, tok, TEST_TEXT)
        codec = SeqCodec(SeqCodecConfig(anchor_stride=8))
        c = codec.compress(keys[0], values[0])
        m = codec.metrics(keys[0], values[0], c)
        assert m["cosine_sim_k"] > 0.85
        assert m["compression_ratio"] > 1.0

    def test_reconstruction_correctness_real_kv(self, real_model):
        import torch.nn.functional as F
        model, tok = real_model
        keys, values = _extract_kv(model, tok, TEST_TEXT)
        codec = SeqCodec(SeqCodecConfig(anchor_stride=1, normalize_residuals=False))
        k, v = keys[0], values[0]
        c = codec.compress(k, v)
        k_rec, _ = codec.decompress(c)
        D = k.shape[-1]
        cos = F.cosine_similarity(
            k.reshape(-1, D).float(),
            k_rec.reshape(-1, D).float(),
            dim=-1,
        ).mean().item()
        assert cos > 0.995, f"stride=1 cosine sim too low on real KV: {cos:.4f}"
