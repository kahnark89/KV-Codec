"""
examples/hf_integration.py
===========================
Shows how to measure KV cache size during a real HuggingFace generation
loop and apply the best codec to the captured cache.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvcodec import KVSystem
from kvcodec.selector import select_codec

MODEL_ID  = "facebook/opt-125m"
PROMPT    = "Transformer models store key-value states for each attention layer."
MAX_TOKENS = 100

model     = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model.eval()

inputs = tokenizer(PROMPT, return_tensors="pt")

# ── 1. Detect compressibility ─────────────────────────────────────────────────
system  = KVSystem(model, tokenizer, device="cpu", verbose=False)
profile = system.detect()
print(f"Strategy: {profile.strategy.name}  "
      f"layer_sim={profile.layer_sim_adj:.3f}  "
      f"seq_sim={profile.seq_sim_adj:.3f}")

# ── 2. Run generation, capture KV cache ──────────────────────────────────────
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

# ── 3. Extract KV tensors ────────────────────────────────────────────────────
all_keys, all_values, seq_len = system.extract_kv(PROMPT)

raw_bytes = sum(k.nelement() * k.element_size() for k in all_keys) * 2
print(f"Raw KV cache: {raw_bytes / 1024:.1f} KB  ({seq_len} tokens)")

# ── 4. Compress ───────────────────────────────────────────────────────────────
codec = select_codec(profile)
if codec is None:
    print("No compression beneficial for this model.")
else:
    compressed   = codec.compress(all_keys, all_values)
    metrics      = codec.metrics(all_keys, all_values, compressed)
    k_rec, v_rec = codec.decompress(compressed)

    print(f"Compression ratio : {metrics['compression_ratio']:.2f}x")
    print(f"Bytes saved       : {metrics['bytes_saved_pct']:.1f}%")
    print(f"Cosine sim K      : {metrics['cosine_sim_k']:.4f}")
    print(f"Cosine sim V      : {metrics['cosine_sim_v']:.4f}")
