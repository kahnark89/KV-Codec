"""
examples/quickstart.py
======================
Minimal end-to-end usage of KV-Codec.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from kvcodec import KVSystem

MODEL_ID = "facebook/opt-125m"

model     = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

system  = KVSystem(model, tokenizer, device="cpu")
profile = system.detect()
results = system.benchmark()
system.print_report()
