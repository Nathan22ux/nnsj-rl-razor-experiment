import os
import sys

# Set PyTorch CUDA memory management BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

sys.path.insert(0, 'src')

import torch
# Set memory fraction to use only 90% of GPU
torch.cuda.set_per_process_memory_fraction(0.9, 0)

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data.load_data import load_dataset_byname
from src.trainingv1.train_sft_baseline import train_sft_baseline

print("="*70)
print("ULTRA-MINIMAL TEST FOR 4GB CUDA")
print("="*70)

# Clear any existing CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"GPU Memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Load TINY model (GPT-2 124M params = ~500MB)
print("\nLoading tiny model (GPT-2 124M - fits easily in 4GB)...")
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",  # Only 124M parameters
    torch_dtype=torch.float16,  # Half precision
    device_map="auto",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print(f"GPU Memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Load dataset
print("\nLoading dataset...")
dataset = load_dataset_byname("math")
print(f"Dataset size: {len(dataset)}")

# Run training with MINIMAL settings
print("\n" + "="*70)
print("Starting SFT training with MINIMAL settings:")
print("  - Model: GPT-2 (124M params)")
print("  - max_samples: 3")
print("  - batch_size: 1")
print("  - epochs: 1")
print("="*70)

trained_model, nt = train_sft_baseline(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    learning_rate=1e-5,
    batch_size=1,
    epochs=1,
    max_samples=3  # Only 3 samples!
)

print("\n" + "="*70)
print("✓ Training completed!")
print("="*70)
