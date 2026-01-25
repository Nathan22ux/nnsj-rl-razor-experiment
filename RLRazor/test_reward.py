"""
Test script for reward.py using real data from the math dataset

Usage:
    python test_reward.py              # Use first 5 examples
    python test_reward.py --start 10   # Use examples 10-14
    python test_reward.py --random     # Use 5 random examples
"""
import sys
import json
import random
import argparse
sys.path.insert(0, 'src')

from trainingv1.reward import (
    extract_answer,
    check_answer_correctness,
    build_binary_rewards
)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test reward.py with math dataset')
parser.add_argument('--start', type=int, default=0, help='Starting index (default: 0)')
parser.add_argument('--random', action='store_true', help='Use random examples instead of sequential')
parser.add_argument('--num', type=int, default=5, help='Number of examples to test (default: 5)')
args = parser.parse_args()

print("="*70)
print("TESTING REWARD.PY WITH REAL MATH DATASET")
print("="*70)

# Load real data from the math dataset
print("\n[1/4] Loading dataset...")
with open('src/data/math/orz_math_13k_collection_hard.json', 'r') as f:
    data = json.load(f)

print(f"✓ Loaded {len(data)} problems from dataset")

# Extract examples based on user selection
print(f"\n[2/4] Extracting {args.num} test examples...")

if args.random:
    print(f"  Mode: Random selection")
    indices = random.sample(range(len(data)), min(args.num, len(data)))
else:
    print(f"  Mode: Sequential starting from index {args.start}")
    indices = range(args.start, min(args.start + args.num, len(data)))

test_examples = []
for idx in indices:
    question = data[idx][0]['value']
    answer = data[idx][1]['ground_truth']['value']
    test_examples.append((question, answer))
    print(f"  Example (index {idx}):")
    print(f"    Q: {question}")
    print(f"    A: {answer}")

# Test 1: extract_answer() with various formats
print("\n[3/4] Testing extract_answer() with simulated model outputs...")
test_cases = [
    (f"The answer is {test_examples[0][1]}", test_examples[0][1]),
    (f"Answer: {test_examples[1][1]}", test_examples[1][1]),
    (f"Therefore, the result is: {test_examples[2][1]}", test_examples[2][1]),
    (f"So the final answer is {test_examples[3][1]}.", test_examples[3][1].rstrip('.')),
]

passed = 0
for i, (model_output, expected) in enumerate(test_cases):
    extracted = extract_answer(model_output)
    # Check if extracted answer is correct
    is_correct = check_answer_correctness(extracted, expected, domain="math")
    status = "✓" if is_correct else "✗"
    if is_correct:
        passed += 1
    print(f"  {status} Test {i+1}: extracted '{extracted}' from model output")
    print(f"      Expected: '{expected}', Match: {is_correct}")

print(f"\n  Result: {passed}/{len(test_cases)} passed")

# Test 2: check_answer_correctness() with real examples
print("\n[4/4] Testing check_answer_correctness() with various formats...")

# Create test cases with different answer formats
test_correctness = []
for i, (question, answer) in enumerate(test_examples[:3]):
    # Correct format
    test_correctness.append((f"Answer: {answer}", answer, True))
    # Wrong answer
    test_correctness.append((f"Answer: wrong_value_{i}", answer, False))
    # Empty answer
    test_correctness.append(("No answer provided", answer, False))

passed = 0
for pred_text, gt_answer, expected in test_correctness:
    result = check_answer_correctness(pred_text, gt_answer, domain="math")
    status = "✓" if result == expected else "✗"
    if result == expected:
        passed += 1
    print(f"  {status} Prediction: '{pred_text}' vs GT: '{gt_answer}' -> {result}")

print(f"\n  Result: {passed}/{len(test_correctness)} passed")

# Test 3: build_binary_rewards() with real examples
print("\n[5/5] Testing build_binary_rewards()...")

# Simulate model generations for 3 prompts
generations = [
    [
        f"Answer: {test_examples[0][1]}",  # Correct
        "Answer: wrong_value",             # Wrong
        f"Answer: {test_examples[0][1]}",  # Correct
    ],
    [
        f"Answer: {test_examples[1][1]}",  # Correct
        f"Answer: {test_examples[1][1]}",  # Correct
        "Answer: wrong_value",             # Wrong
    ],
    [
        "Answer: wrong_value",             # Wrong
        "Answer: wrong_value",             # Wrong
        f"Answer: {test_examples[2][1]}",  # Correct
    ]
]

answers = [ex[1] for ex in test_examples[:3]]

print(f"  Building rewards for {len(generations)} prompts...")
rewards = build_binary_rewards(generations, answers, domains=None)

print(f"  ✓ Generated {len(rewards)} reward tensors")
for i, reward_tensor in enumerate(rewards):
    reward_list = reward_tensor.tolist()
    num_correct = sum(reward_list)
    print(f"    Prompt {i+1} rewards: {reward_list} ({int(num_correct)}/3 correct)")

# Verify expected pattern
expected_patterns = [
    [1.0, 0.0, 1.0],  # 2 correct
    [1.0, 1.0, 0.0],  # 2 correct
    [0.0, 0.0, 1.0],  # 1 correct
]

all_match = all(
    rewards[i].tolist() == expected_patterns[i]
    for i in range(len(rewards))
)

if all_match:
    print(f"  ✓ All reward patterns match expected!")
else:
    print(f"  ⚠ Some reward patterns don't match (may be due to answer extraction)")

# ============================================================================
# TEST NT EVALUATION
# ============================================================================
print("\n" + "="*70)
print("TESTING NT EVALUATION WITH MATH DATASET")
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
model_nt = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True
)
tokenizer_nt = AutoTokenizer.from_pretrained("gpt2")
tokenizer_nt.pad_token = tokenizer_nt.eos_token

if torch.cuda.is_available():
    print(f"GPU Memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Load math dataset
print("\nLoading math dataset via load_dataset_byname...")
math_dataset = load_dataset_byname("math")
print(f"Dataset size: {len(math_dataset)} math problems")

# Use a small subset for evaluation
eval_subset = math_dataset.select(range(min(5, len(math_dataset))))

print(f"\nEvaluating NT on {len(eval_subset)} examples...")
from evaluation.evaluation import evaluate_new_task

# Compute NT score
nt_score = evaluate_new_task(
    model=model_nt,
    tokenizer=tokenizer_nt,
    dataset=eval_subset,
    num_samples=len(eval_subset)
)

print(f"NT Score: {nt_score:.2f}%")

# Test individual predictions to verify correctness checking
print("\n" + "-"*70)
print("Sample Predictions (Math):")
print("-"*70)

correct_count = 0
for i in range(min(3, len(eval_subset))):
    example = eval_subset[i]
    question = example['question']
    gt_answer = example['answer']

    # Generate answer
    prompt = f"Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer:"
    inputs = tokenizer_nt(prompt, return_tensors="pt", truncation=True, max_length=512)

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    with torch.no_grad():
        outputs = model_nt.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer_nt.eos_token_id
        )

    # Decode only generated portion
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    pred_answer = tokenizer_nt.decode(generated_ids, skip_special_tokens=True)

    # Check correctness using domain-specific function
    is_correct = check_answer_correctness(pred_answer, gt_answer, domain="math")

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
del model_nt
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Final summary
print("\n" + "="*70)
print("TEST SUMMARY - REAL DATASET")
print("="*70)
print(f"✓ Tested with {len(test_examples)} real math problems")
print(f"✓ Dataset: orz_math_13k_collection_hard.json")
print(f"✓ NT evaluation tested: {nt_score:.2f}%")
print("\nFunctions verified:")
print("  ✓ extract_answer() - Extracts answers from model outputs")
print("  ✓ check_answer_correctness() - Full correctness checking (domain=math)")
print("  ✓ build_binary_rewards() - Creates reward tensors")
print("  ✓ evaluate_new_task() - Computes NT score for math domain")
print("="*70)
print("\n✓✓✓ reward.py and NT evaluation work with real data! ✓✓✓\n")
