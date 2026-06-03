import pytest
import torch


@pytest.fixture(scope="session")
def synthetic_kv():
    """Small synthetic KV tensors — no model needed. Shape: [H, T, D]."""
    torch.manual_seed(42)
    L, H, T, D = 12, 4, 64, 32
    keys   = [torch.randn(H, T, D) for _ in range(L)]
    values = [torch.randn(H, T, D) for _ in range(L)]
    return keys, values


@pytest.fixture(scope="session")
def real_model():
    """Load opt-125m once per test session. Marked slow."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL_ID = "facebook/opt-125m"
    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    return model, tok
