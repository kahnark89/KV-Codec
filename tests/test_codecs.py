"""
Fast unit tests — run entirely on synthetic tensors, no model download needed.
"""
import pytest
import torch
import torch.nn.functional as F

from kvcodec import (
    SeqCodec, SeqCodecConfig,
    LayerCodec, LayerCodecConfig,
    JointCodec, JointCodecConfig,
    CompressibilityProfile, CodecStrategy,
)


# ── SeqCodec ──────────────────────────────────────────────────────────────────

class TestSeqCodec:
    def test_compress_returns_compressed_object(self, synthetic_kv):
        keys, values = synthetic_kv
        codec = SeqCodec(SeqCodecConfig(anchor_stride=8))
        c = codec.compress(keys[0], values[0])
        assert c is not None

    def test_decompress_restores_shape(self, synthetic_kv):
        keys, values = synthetic_kv
        H, T, D = keys[0].shape
        codec = SeqCodec(SeqCodecConfig(anchor_stride=8))
        c = codec.compress(keys[0], values[0])
        k_rec, v_rec = codec.decompress(c)
        assert k_rec.shape == (H, T, D)
        assert v_rec.shape == (H, T, D)

    def test_stride1_is_near_lossless(self, synthetic_kv):
        keys, values = synthetic_kv
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
        assert cos > 0.995, f"Stride-1 cosine sim too low: {cos:.4f}"

    @pytest.mark.parametrize("stride", [4, 8, 16, 32])
    def test_metrics_keys_present(self, synthetic_kv, stride):
        keys, values = synthetic_kv
        codec = SeqCodec(SeqCodecConfig(anchor_stride=stride))
        c = codec.compress(keys[0], values[0])
        m = codec.metrics(keys[0], values[0], c)
        for key in ("cosine_sim_k", "cosine_sim_v", "compression_ratio", "bytes_saved_pct"):
            assert key in m, f"Missing metric: {key}"

    def test_higher_stride_gives_higher_ratio(self, synthetic_kv):
        keys, values = synthetic_kv
        ratios = []
        for stride in [4, 8, 16, 32]:
            codec = SeqCodec(SeqCodecConfig(anchor_stride=stride))
            c = codec.compress(keys[0], values[0])
            m = codec.metrics(keys[0], values[0], c)
            ratios.append(m["compression_ratio"])
        assert ratios == sorted(ratios), "Compression ratio should increase with stride"


# ── LayerCodec ────────────────────────────────────────────────────────────────

class TestLayerCodec:
    def test_compress_decompress_shape(self, synthetic_kv):
        keys, values = synthetic_kv
        H, T, D = keys[0].shape
        codec = LayerCodec(LayerCodecConfig(anchor_layer_stride=2))
        c = codec.compress(keys, values)
        reconstructed = codec.decompress(c)
        assert len(reconstructed) == 2              # (keys_list, values_list)
        assert len(reconstructed[0]) == len(keys)
        assert reconstructed[0][0].shape == (H, T, D)

    def test_metrics_compression_ratio_gt1(self, synthetic_kv):
        keys, values = synthetic_kv
        codec = LayerCodec(LayerCodecConfig(anchor_layer_stride=4))
        c = codec.compress(keys, values)
        m = codec.metrics(keys, values, c)
        assert m["compression_ratio"] > 1.0

    @pytest.mark.parametrize("stride", [2, 4])
    def test_metrics_keys_present(self, synthetic_kv, stride):
        keys, values = synthetic_kv
        codec = LayerCodec(LayerCodecConfig(anchor_layer_stride=stride))
        c = codec.compress(keys, values)
        m = codec.metrics(keys, values, c)
        for key in ("cosine_sim_k", "cosine_sim_v", "compression_ratio"):
            assert key in m


# ── JointCodec ────────────────────────────────────────────────────────────────

class TestJointCodec:
    def test_compress_decompress_shape(self, synthetic_kv):
        keys, values = synthetic_kv
        H, T, D = keys[0].shape
        codec = JointCodec(JointCodecConfig(seq_anchor_stride=8, layer_anchor_stride=2))
        c = codec.compress(keys, values)
        reconstructed = codec.decompress(c)
        assert len(reconstructed[0]) == len(keys)
        assert reconstructed[0][0].shape == (H, T, D)

    def test_higher_compression_than_single_axis(self, synthetic_kv):
        keys, values = synthetic_kv
        seq_codec   = SeqCodec(SeqCodecConfig(anchor_stride=8))
        joint_codec = JointCodec(JointCodecConfig(seq_anchor_stride=8, layer_anchor_stride=2))

        seq_m   = seq_codec.metrics(keys[0], values[0],
                                    seq_codec.compress(keys[0], values[0]))
        joint_c = joint_codec.compress(keys, values)
        joint_m = joint_codec.metrics(keys, values, joint_c)

        assert joint_m["compression_ratio"] >= seq_m["compression_ratio"]

    @pytest.mark.parametrize("sl,sp", [(2, 8), (4, 16), (4, 32)])
    def test_metrics_present(self, synthetic_kv, sl, sp):
        keys, values = synthetic_kv
        codec = JointCodec(JointCodecConfig(layer_anchor_stride=sl, seq_anchor_stride=sp))
        c = codec.compress(keys, values)
        m = codec.metrics(keys, values, c)
        for key in ("cosine_sim_k", "cosine_sim_v", "compression_ratio", "bytes_saved_pct"):
            assert key in m


# ── CompressibilityProfile ────────────────────────────────────────────────────

class TestCompressibilityProfile:
    def _make_profile(self, layer_sim, seq_sim, joint_sim=None):
        joint = joint_sim if joint_sim is not None else layer_sim * seq_sim
        return CompressibilityProfile(
            layer_sim_adj=layer_sim,
            layer_sim_mid=layer_sim * 0.8,
            layer_sim_decay=0.1,
            seq_sim_adj=seq_sim,
            seq_sim_8=seq_sim * 0.8,
            seq_sim_global=seq_sim * 0.5,
            joint_sim=joint,
            joint_bonus=joint - layer_sim * seq_sim,
            n_layers=12, n_heads=4, head_dim=32, seq_len_measured=64,
        )

    def test_strategy_none(self):
        p = self._make_profile(layer_sim=0.3, seq_sim=0.2)
        assert p.strategy == CodecStrategy.NONE

    def test_strategy_seq_only(self):
        p = self._make_profile(layer_sim=0.3, seq_sim=0.6)
        assert p.strategy == CodecStrategy.SEQ_ONLY

    def test_strategy_layer_only(self):
        p = self._make_profile(layer_sim=0.8, seq_sim=0.2)
        assert p.strategy == CodecStrategy.LAYER_ONLY

    def test_strategy_joint(self):
        p = self._make_profile(layer_sim=0.8, seq_sim=0.6, joint_sim=0.9)
        assert p.strategy == CodecStrategy.JOINT_2D

    def test_strategy_separable_axes_pick_single(self):
        # Both axes individually compressible, but joint_sim equals the product
        # of the 1D sims → no coupling bonus → the better single axis wins,
        # not the heavier JOINT_2D codec.
        p = self._make_profile(layer_sim=0.8, seq_sim=0.6, joint_sim=0.8 * 0.6)
        assert p.joint_coupling is False
        assert p.strategy != CodecStrategy.JOINT_2D
        assert p.strategy in (CodecStrategy.SEQ_ONLY, CodecStrategy.LAYER_ONLY)

    def test_compression_ratio_gt1_when_compressible(self):
        p = self._make_profile(layer_sim=0.8, seq_sim=0.6)
        assert p.estimated_compression_ratio() > 1.0
