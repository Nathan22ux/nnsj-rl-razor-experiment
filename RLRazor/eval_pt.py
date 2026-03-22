# eval_pt.py  (place in RLRazor/ directory)
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURE THESE ---
MODEL_PATH = "./results_rl/lr3e-06_mu1/model"   # or base model path
LIMIT = 100                                        # examples per benchmark
# -----------------------

sys.path.insert(0, "src")
from evaluation.evaluation import evaluate_benchmarks

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

scores = evaluate_benchmarks(model, tokenizer, limit=LIMIT, use_extended=False)
print(scores)
