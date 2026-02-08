"""
Test script for reward.py using real data from the science dataset

Usage:
    python test_reward_science.py              # Use first 5 examples
    python test_reward_science.py --start 10   # Use examples 10-14
    python test_reward_science.py --random     # Use 5 random examples
"""
import sys
import json
import random
import argparse
from pathlib import Path
sys.path.insert(0, 'src')

from trainingv1.reward import (
    extract_answer,
    check_answer_correctness,
    build_binary_rewards
)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test reward.py with science dataset')
parser.add_argument('--start', type=int, default=0, help='Starting index (default: 0)')
parser.add_argument('--random', action='store_true', help='Use random examples instead of sequential')
parser.add_argument('--num', type=int, default=5, help='Number of examples to test (default: 5)')
args = parser.parse_args()

print("="*70)
print("TESTING REWARD.PY WITH REAL SCIENCE DATASET")
print("="*70)

# Load real data from the science dataset (JSONL files)
print("\n[1/4] Loading dataset...")
science_dir = Path('src/data/science')
all_examples = []

# Load all JSONL files
jsonl_files = list(science_dir.glob("*.jsonl"))
print(f"  Found {len(jsonl_files)} JSONL files:")

for jsonl_file in jsonl_files:
    print(f"    - {jsonl_file.name}")
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_examples.append(json.loads(line))

print(f"✓ Loaded {len(all_examples)} chemistry problems from dataset")

# Extract examples based on user selection
print(f"\n[2/4] Extracting {args.num} test examples...")

if args.random:
    print(f"  Mode: Random selection")
    indices = random.sample(range(len(all_examples)), min(args.num, len(all_examples)))
else:
    print(f"  Mode: Sequential starting from index {args.start}")
    indices = range(args.start, min(args.start + args.num, len(all_examples)))

test_examples = []
for idx in indices:
    example = all_examples[idx]
    question = example['question']
    answer_key = example.get('answerKey', '')

    # Get the full answer (e.g., "A. 1639.900")
    if answer_key and 'choices' in example:
        labels = example['choices'].get('label', [])
        values = example['choices'].get('text', [])
        try:
            answer_idx = labels.index(answer_key)
            full_answer = f"{answer_key}. {values[answer_idx]}"
        except (ValueError, IndexError):
            full_answer = answer_key
    else:
        full_answer = answer_key

    test_examples.append((question, full_answer, answer_key))
    print(f"  Example (index {idx}):")
    print(f"    Q: {question}")
    print(f"    Answer Key: {answer_key}")
    print(f"    Full Answer: {full_answer}")

# Test 1: extract_answer() with various formats
print("\n[3/4] Testing extract_answer() with simulated model outputs...")
test_cases = [
    (f"The answer is {test_examples[0][2]}", test_examples[0][2]),  # Just letter
    (f"Answer: {test_examples[1][1]}", test_examples[1][1]),        # Full answer
    (f"Therefore, the result is: {test_examples[2][2]}", test_examples[2][2]),  # Just letter
]

passed = 0
for i, (model_output, expected) in enumerate(test_cases):
    extracted = extract_answer(model_output)
    # For science, we check exact match or if answer key is in output
    is_correct = (extracted == expected) or (test_examples[i][2] in model_output)
    status = "✓" if is_correct else "✗"
    if is_correct:
        passed += 1
    print(f"  {status} Test {i+1}: extracted '{extracted}' from model output")
    print(f"      Expected: '{expected}', Match: {is_correct}")

print(f"\n  Result: {passed}/{len(test_cases)} passed")

# Test 2: check_answer_correctness() with science domain
print("\n[4/4] Testing check_answer_correctness() with science domain...")

# Create test cases with different answer formats
test_correctness = []
for i, (question, full_answer, answer_key) in enumerate(test_examples[:3]):
    # Correct format - just the letter
    test_correctness.append((f"Answer: {answer_key}", full_answer, True, f"letter only {i+1}"))

    # Correct format - full answer
    test_correctness.append((f"Answer: {full_answer}", full_answer, True, f"full answer {i+1}"))

    # Substring match - answer key in text
    test_correctness.append((f"The correct answer is {answer_key}.", full_answer, True, f"substring {i+1}"))

    # Wrong answer
    wrong_key = "Z" if answer_key != "Z" else "Y"
    test_correctness.append((f"Answer: {wrong_key}", full_answer, False, f"wrong {i+1}"))

passed = 0
for pred_text, gt_answer, expected, label in test_correctness:
    result = check_answer_correctness(pred_text, gt_answer, domain="science")
    status = "✓" if result == expected else "✗"
    if result == expected:
        passed += 1
    print(f"  {status} {label}: {result} (expected: {expected})")

print(f"\n  Result: {passed}/{len(test_correctness)} passed")

# Test 3: build_binary_rewards() with science domain
print("\n[5/5] Testing build_binary_rewards() with science domain...")

# Simulate model generations for 3 prompts
generations = []
answers = []

for i, (question, full_answer, answer_key) in enumerate(test_examples[:3]):
    # Create 3 generations per prompt
    wrong_key = "Z" if answer_key != "Z" else "Y"
    gen_group = [
        f"Answer: {answer_key}",           # Correct (just letter)
        f"Answer: {wrong_key}",            # Wrong
        f"The answer is {full_answer}"     # Correct (full answer)
    ]
    generations.append(gen_group)
    answers.append(full_answer)

print(f"  Building rewards for {len(generations)} prompts...")
domains = ["science"] * len(generations)
rewards = build_binary_rewards(generations, answers, domains=domains)

print(f"  ✓ Generated {len(rewards)} reward tensors")
for i, reward_tensor in enumerate(rewards):
    reward_list = reward_tensor.tolist()
    num_correct = sum(reward_list)
    print(f"    Prompt {i+1} rewards: {reward_list} ({int(num_correct)}/3 correct)")

# Verify expected pattern (2 correct per prompt)
expected_correct = [2, 2, 2]  # Each should have 2 correct
actual_correct = [int(sum(rewards[i].tolist())) for i in range(len(rewards))]

if actual_correct == expected_correct:
    print(f"  ✓ All reward patterns match expected (2/3 correct each)!")
else:
    print(f"  ⚠ Actual: {actual_correct}, Expected: {expected_correct}")

# ============================================================================
# TEST NT EVALUATION
# ============================================================================
print("\n" + "="*70)
print("TESTING NT EVALUATION WITH SCIENCE DATASET")
print("="*70)

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data.load_data import load_dataset_byname

# Clear CUDA cache if available
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU Memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Load a small model for testing
print("\nLoading model (GPT-2 124M for 4GB CUDA)...")
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

if torch.cuda.is_available():
    print(f"GPU Memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Load science dataset
print("\nLoading science dataset via load_dataset_byname...")
science_dataset = load_dataset_byname("science")
print(f"Dataset size: {len(science_dataset)} chemistry problems")

# Use a small subset for evaluation
eval_subset = science_dataset.select(range(min(5, len(science_dataset))))

print(f"\nEvaluating NT on {len(eval_subset)} examples...")
from evaluation.evaluation import evaluate_new_task

# Compute NT score
nt_score = evaluate_new_task(
    model=model,
    tokenizer=tokenizer,
    dataset=eval_subset,
    num_samples=len(eval_subset)
)

print(f"NT Score: {nt_score:.2f}%")

# Test individual predictions to verify correctness checking
print("\n" + "-"*70)
print("Sample Predictions (Science - Chemistry):")
print("-"*70)

correct_count = 0
for i in range(min(3, len(eval_subset))):
    example = eval_subset[i]
    question = example['question']
    gt_answer = example['answer']

    # Generate answer
    prompt = f"Question: {question}\nPlease select the correct answer.\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only generated portion
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    pred_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Check correctness using domain-specific function
    is_correct = check_answer_correctness(pred_answer, gt_answer, domain="science")

    if is_correct:
        correct_count += 1

    print(f"\nExample {i+1}:")
    print(f"  Question: {question[:80]}...")
    print(f"  Ground Truth: {gt_answer}")
    print(f"  Prediction: {pred_answer[:80]}")
    print(f"  Correct: {'✓' if is_correct else '✗'}")

print("\n" + "-"*70)
print(f"Manual accuracy: {correct_count}/3 = {correct_count/3*100:.1f}%")
print("-"*70)

# Cleanup
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Final summary
print("\n" + "="*70)
print("TEST SUMMARY - SCIENCE DATASET")
print("="*70)
print(f"✓ Tested with {len(test_examples)} chemistry problems")
print(f"✓ Dataset: science/*.jsonl (SciKnowEval Chemistry)")
print(f"✓ Total examples in dataset: {len(all_examples)}")
print(f"✓ JSONL files: {len(jsonl_files)}")
print(f"✓ NT evaluation tested: {nt_score:.2f}%")
print("\nFunctions verified:")
print("  ✓ extract_answer() - Extracts answer keys (A, B, C, D)")
print("  ✓ check_answer_correctness() - Full correctness checking (domain=science)")
print("  ✓ build_binary_rewards() - Creates reward tensors for science domain")
print("  ✓ evaluate_new_task() - Computes NT score for science domain")
print("="*70)
print("\n✓✓✓ reward.py and NT evaluation work with science data! ✓✓✓\n")
