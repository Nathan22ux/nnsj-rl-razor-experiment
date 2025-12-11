import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME

print(f"Checking model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Model embeddings size: {model.get_input_embeddings().weight.shape[0]}")
print(f"Tokenizer len: {len(tokenizer)}")

if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
    print("WARNING: Tokenizer has more tokens than model embeddings!")
else:
    print("Vocab sizes look OK.")
