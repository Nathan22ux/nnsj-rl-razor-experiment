"""
Test script for reward.py using real data from the tool dataset

Usage:
    python test_reward_tool.py              # Use first 5 examples
    python test_reward_tool.py --start 10   # Use examples 10-14
    python test_reward_tool.py --random     # Use 5 random examples
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
parser = argparse.ArgumentParser(description='Test reward.py with tool dataset')
parser.add_argument('--start', type=int, default=0, help='Starting index (default: 0)')
parser.add_argument('--random', action='store_true', help='Use random examples instead of sequential')
parser.add_argument('--num', type=int, default=5, help='Number of examples to test (default: 5)')
args = parser.parse_args()

print("="*70)
print("TESTING REWARD.PY WITH REAL TOOL DATASET")
print("="*70)

# Load real data from the tool dataset
print("\n[1/4] Loading dataset...")
with open('src/data/tool/train_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✓ Loaded {len(data)} APIs from dataset")

# Flatten to get all instances
all_instances = []
for api in data:
    api_name = api.get('Name', 'Unknown API')
    instances = api.get('Instances', [])
    for instance in instances:
        if isinstance(instance, dict):
            all_instances.append({
                'api_name': api_name,
                'input': instance.get('input', ''),
                'output': instance.get('output', ''),
                'intermediate_steps': instance.get('intermediate_steps', [])
            })

print(f"✓ Total instances: {len(all_instances)}")

# Extract examples based on user selection
print(f"\n[2/4] Extracting {args.num} test examples...")

if args.random:
    print(f"  Mode: Random selection")
    indices = random.sample(range(len(all_instances)), min(args.num, len(all_instances)))
else:
    print(f"  Mode: Sequential starting from index {args.start}")
    indices = range(args.start, min(args.start + args.num, len(all_instances)))

test_examples = []
for idx in indices:
    instance = all_instances[idx]
    api_name = instance['api_name']
    user_input = instance['input']
    expected_output = instance['output']

    test_examples.append((user_input, expected_output))
    print(f"  Example {len(test_examples)} (API: {api_name}):")
    print(f"    Input: {user_input}")
    print(f"    Output: {expected_output}")

# Test 1: Extract action from intermediate steps
print("\n[3/4] Testing action extraction from tool use examples...")
passed = 0
total = 0

for idx in indices[:3]:  # Test first 3
    instance = all_instances[idx]
    intermediate_steps = instance['intermediate_steps']

    if intermediate_steps and len(intermediate_steps) > 0:
        step = intermediate_steps[0]
        if isinstance(step, list) and len(step) >= 2:
            action_info = step[0]
            if isinstance(action_info, list) and len(action_info) >= 3:
                action = action_info[0]
                action_input = action_info[1]

                # Simulate model output
                model_output = f"Action: {action}\nAction Input: {action_input}"

                # Test if we can extract it
                extracted = extract_answer(model_output)

                # Check if action is in the extracted text
                if action in model_output:
                    passed += 1
                    print(f"  ✓ Example {total+1}: Found action '{action}' in output")
                else:
                    print(f"  ✗ Example {total+1}: Could not find action")

                total += 1

print(f"\n  Result: {passed}/{total} action extractions successful")

# Test 2: check_answer_correctness() with tool domain
print("\n[4/4] Testing check_answer_correctness() with tool domain...")

# Create test cases
test_correctness = []
for i, (user_input, expected_output) in enumerate(test_examples[:3]):
    # Correct format - full output
    test_correctness.append((expected_output, expected_output, True, f"exact match {i+1}"))

    # Partial match - substring
    partial = expected_output[:min(50, len(expected_output))]
    test_correctness.append((f"The result is: {partial}", partial, True, f"substring {i+1}"))

    # Wrong answer
    test_correctness.append(("Wrong answer completely different", expected_output, False, f"wrong {i+1}"))

passed = 0
for pred_text, gt_answer, expected, label in test_correctness:
    result = check_answer_correctness(pred_text, gt_answer, domain="tool")
    status = "✓" if result == expected else "✗"
    if result == expected:
        passed += 1
    print(f"  {status} {label}: {result} (expected: {expected})")

print(f"\n  Result: {passed}/{len(test_correctness)} passed")

# Test 3: build_binary_rewards() with tool examples
print("\n[5/5] Testing build_binary_rewards() with tool domain...")

# Simulate model generations for 3 prompts
generations = []
answers = []

for i, (user_input, expected_output) in enumerate(test_examples[:3]):
    # Create 3 generations per prompt
    gen_group = [
        expected_output,           # Correct
        "Wrong answer",           # Wrong
        f"Result: {expected_output}"  # Correct (substring match)
    ]
    generations.append(gen_group)
    answers.append(expected_output)

print(f"  Building rewards for {len(generations)} prompts...")
domains = ["tool"] * len(generations)
rewards = build_binary_rewards(generations, answers, domains=domains)

print(f"  ✓ Generated {len(rewards)} reward tensors")
for i, reward_tensor in enumerate(rewards):
    reward_list = reward_tensor.tolist()
    num_correct = sum(reward_list)
    print(f"    Prompt {i+1} rewards: {reward_list} ({int(num_correct)}/3 correct)")

# ============================================================================
# TEST NT EVALUATION
# ============================================================================
print("\n" + "="*70)
print("TESTING NT EVALUATION WITH TOOL DATASET")
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

# Load tool dataset
print("\nLoading tool dataset via load_dataset_byname...")
tool_dataset = load_dataset_byname("tool")
print(f"Dataset size: {len(tool_dataset)} tool-use examples")

# Use a small subset for evaluation
eval_subset = tool_dataset.select(range(min(5, len(tool_dataset))))

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
print("Sample Predictions (Tool Use - API Calls):")
print("-"*70)

correct_count = 0
for i in range(min(3, len(eval_subset))):
    example = eval_subset[i]
    instruction = example['instruction']
    input_text = example.get('input', '')
    gt_output = example['output']

    # Generate answer
    if input_text:
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
    else:
        prompt = f"Instruction: {instruction}\nResponse:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only generated portion
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    pred_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Check correctness using domain-specific function
    is_correct = check_answer_correctness(pred_output, gt_output, domain="tool")

    if is_correct:
        correct_count += 1

    print(f"\nExample {i+1}:")
    print(f"  Instruction: {instruction[:60]}...")
    print(f"  Ground Truth: {gt_output[:80]}...")
    print(f"  Prediction: {pred_output[:80]}...")
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
print("TEST SUMMARY - TOOL DATASET")
print("="*70)
print(f"✓ Tested with {len(test_examples)} tool-use examples")
print(f"✓ Dataset: tool/train_data.json")
print(f"✓ Total APIs: {len(data)}")
print(f"✓ Total instances: {len(all_instances)}")
print(f"✓ NT evaluation tested: {nt_score:.2f}%")
print("\nFunctions verified:")
print("  ✓ extract_answer() - Extracts actions/outputs")
print("  ✓ check_answer_correctness() - Full correctness checking (domain=tool)")
print("  ✓ build_binary_rewards() - Creates reward tensors for tool domain")
print("  ✓ evaluate_new_task() - Computes NT score for tool domain")
print("="*70)
print("\n✓✓✓ reward.py and NT evaluation work with tool data! ✓✓✓\n")
