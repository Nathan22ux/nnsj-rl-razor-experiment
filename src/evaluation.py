"""
Corrected evaluation.py with bug fixes for RL's Razor replication
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import re

EVAL_BENCHMARKS = [
    "winogrande",
    "hellaswag",
    "mmlu_high_school_mathematics",
    "mmlu_high_school_computer_science"
]


def evaluate_benchmarks(model, tokenizer, tasks=EVAL_BENCHMARKS, limit=100):
    """
    Evaluate model on standard benchmarks to measure prior task performance.

    FIXES APPLIED:
    - Increased default limit from 50 to 100 for more stable estimates
    - Better handling of different benchmark metrics

    Args:
        model: Model to evaluate (should be fine-tuned model)
        tokenizer: Tokenizer for the model
        tasks: List of benchmark tasks to evaluate on
        limit: Maximum number of examples per task

    Returns:
        dict: Dictionary with per-task and average accuracy scores
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    print(f"\n{'='*70}")
    print(f"EVALUATING MODEL ON BENCHMARK TASKS")
    print(f"{'='*70}")
    print(f"Purpose: Measure prior knowledge retention (catastrophic forgetting)")
    print(f"\nBenchmark tasks:")
    for task in tasks:
        print(f"    {task}")
    print(f"\nEvaluation settings:")
    print(f"  Few-shot examples: 0 (zero-shot evaluation)")
    print(f"  Limit per task: {limit} examples")
    print(f"{'='*70}\n")

    print(" Wrapping model for lm-eval harness...")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device="cuda",
        max_length=1024
    )
    print(" Model wrapped successfully\n")

    # Note: For multiple choice tasks, we don't need generation
    # lm-eval handles this automatically

    print(f" Running evaluation across {len(tasks)} tasks...")

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        limit=limit,
        confirm_run_unsafe_code=True,
    )

    print("\n Evaluation complete!")

    # Extract accuracy scores from results
    print("\n Extracting scores from evaluation results...")
    scores = {}

    for task in tasks:
        if task in results['results']:
            task_result = results['results'][task]

            # Try different metric names (varies by benchmark)
            if 'acc,none' in task_result:
                scores[task] = task_result['acc,none']
            elif 'acc' in task_result:
                scores[task] = task_result['acc']
            elif 'acc_norm,none' in task_result:
                scores[task] = task_result['acc_norm,none']
            elif 'acc_norm' in task_result:
                scores[task] = task_result['acc_norm']
            else:
                # Fallback: try to extract any numeric score
                for key, value in task_result.items():
                    if isinstance(value, (int, float)) and 'acc' in key.lower():
                        scores[task] = float(value)
                        break

    # Compute average accuracy across all tasks
    task_scores = [v for k, v in scores.items() if k != 'average']
    scores['average'] = np.mean(task_scores) if task_scores else 0.0

    print(f"\n{'='*70}")
    print(f"BENCHMARK EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Per-task accuracy (prior knowledge retention):")
    for task, score in scores.items():
        if task != 'average':
            print(f"    {task:40s}: {score:.4f}")
    print(f"\n   {'Average Prior Task Performance':40s}: {scores['average']:.4f}")
    print(f"{'='*70}\n")

    return scores


def compute_forward_kl(model, base_model, dataset, tokenizer, num_samples=100, response_only=True):
    """
    Compute forward KL divergence KL(base || model).

    FIXES APPLIED:
    - Added response_only parameter to compute KL only on response tokens
    - Increased default num_samples from 50 to 100
    - Better handling of prompt vs response separation

    Formula: KL(P_base || P_model) = Σ P_base * log(P_base / P_model)

    Args:
        model: Fine-tuned model
        base_model: Base model (reference for comparison)
        dataset: Dataset to compute KL on (should have 'text' field)
        tokenizer: Tokenizer for the models
        num_samples: Number of samples to use for computation
        response_only: If True, compute KL only on response tokens (recommended)

    Returns:
        float: Average KL divergence across samples
    """
    print(f"\n{'='*70}")
    print(f"COMPUTING KL DIVERGENCE (Forward KL)")
    print(f"{'='*70}")
    print(f"Formula: KL(P_base || P_model) = Σ P_base * log(P_base / P_model)")
    print(f"Response-only: {response_only}")
    print(f"Number of samples: {num_samples}")
    print(f"{'='*70}\n")

    model.eval()
    base_model.eval()

    model_device = next(model.parameters()).device
    base_device = next(base_model.parameters()).device
    print(f" Model device: {model_device}")
    print(f" Base model device: {base_device}")

    total_kl = 0.0
    count = 0
    kl_per_sample = []

    print(f"\n Selecting {min(num_samples, len(dataset))} random samples from dataset...")
    num_to_sample = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_to_sample, replace=False)

    print("\n Computing KL divergence for each sample...")

    with torch.no_grad():
        for idx in tqdm(indices, desc="Computing KL"):
            text = dataset[int(idx)]['text']

            # FIX: Identify prompt vs response boundary
            if response_only:
                # Find where the response starts (after "Answer:")
                answer_pos = text.find("Answer:")
                if answer_pos != -1:
                    prompt_text = text[:answer_pos + len("Answer:")]
                    prompt_ids = tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024,
                        padding=False
                    )['input_ids']
                    response_start_idx = prompt_ids.shape[1]
                else:
                    # Fallback: use all tokens
                    response_start_idx = 0
            else:
                response_start_idx = 0

            # Tokenize full text
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False
            )

            input_ids = inputs['input_ids'].to(model_device)
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(model_device)

            # Skip if sequence is too short
            if input_ids.shape[1] <= response_start_idx + 1:
                continue

            # Get logits from both models
            base_outputs = base_model(
                input_ids.to(base_device),
                attention_mask=attention_mask.to(base_device) if attention_mask is not None else None
            )
            model_outputs = model(input_ids, attention_mask=attention_mask)

            base_logits = base_outputs.logits
            model_logits = model_outputs.logits

            # Move to same device for computation
            base_logits = base_logits.to(model_device)

            # Use log_softmax for numerical stability
            base_log_probs = F.log_softmax(base_logits, dim=-1)
            model_log_probs = F.log_softmax(model_logits, dim=-1)

            # Convert to probs for base model (need actual probabilities for weighting)
            base_probs = torch.exp(base_log_probs)

            # FIX: Compute KL only on response tokens
            if response_only and response_start_idx > 0:
                base_log_probs = base_log_probs[:, response_start_idx:, :]
                model_log_probs = model_log_probs[:, response_start_idx:, :]
                base_probs = base_probs[:, response_start_idx:, :]

            # Handle sequence length mismatch
            min_len = min(base_log_probs.size(1), model_log_probs.size(1))
            if min_len == 0:
                continue

            base_log_probs = base_log_probs[:, :min_len, :]
            model_log_probs = model_log_probs[:, :min_len, :]
            base_probs = base_probs[:, :min_len, :]

            # Compute KL divergence per token, then average across sequence
            # KL(P || Q) = Σ P * (log(P) - log(Q))
            kl_per_token = base_probs * (base_log_probs - model_log_probs)
            kl_sum = kl_per_token.sum(dim=-1)  # Sum over vocab
            kl_mean = kl_sum.mean(dim=1)  # Average over sequence
            kl_value = kl_mean.item()

            # Sanity check: KL should be non-negative
            if kl_value < -0.01:  # Allow small numerical errors
                print(f"Warning: Negative KL value {kl_value} at sample {idx}")
                continue

            total_kl += max(0, kl_value)  # Clamp to non-negative
            kl_per_sample.append(max(0, kl_value))
            count += 1

    # Compute statistics
    avg_kl = total_kl / count if count > 0 else 0.0
    std_kl = np.std(kl_per_sample) if len(kl_per_sample) > 1 else 0.0
    min_kl = np.min(kl_per_sample) if kl_per_sample else 0.0
    max_kl = np.max(kl_per_sample) if kl_per_sample else 0.0

    print(f"\n KL Divergence Computation Complete!")
    print(f"{'='*70}")
    print(f"KL Statistics:")
    print(f"   Mean KL Divergence: {avg_kl:.6f}")
    print(f"   Std Dev: {std_kl:.6f}")
    print(f"   Min: {min_kl:.6f}")
    print(f"   Max: {max_kl:.6f}")
    print(f"   Valid Samples: {count}/{num_to_sample}")
    print(f"{'='*70}\n")

    return avg_kl


def evaluate_new_task(model, tokenizer, dataset, eval_dataset=None, max_new_tokens=64, num_samples=100):
    """
    Evaluate New Task performance (NT).

    FIXES APPLIED:
    - Added eval_dataset parameter for proper train/eval split
    - Increased default num_samples from 50 to 100
    - Improved answer extraction
    - Stricter answer matching

    Args:
        model: Fine-tuned model
        tokenizer: Model tokenizer
        dataset: Training dataset (used if eval_dataset not provided)
        eval_dataset: Separate evaluation dataset (recommended)
        max_new_tokens: Max tokens to generate
        num_samples: Number of samples to evaluate

    Returns:
        float: Accuracy on new task (0-100)
    """
    model.eval()

    # FIX: Use separate eval dataset if provided
    if eval_dataset is not None:
        eval_data = eval_dataset
        print(f"Using separate evaluation dataset with {len(eval_data)} examples")
    else:
        # Fallback: use last portion of training data (not ideal)
        eval_start = max(0, len(dataset) - num_samples)
        eval_data = dataset.select(range(eval_start, len(dataset)))
        print(f"Warning: Using end of training data for evaluation (consider providing eval_dataset)")

    correct = 0
    total = min(num_samples, len(eval_data))

    print(f"\n{'='*70}")
    print(f"EVALUATING NEW TASK PERFORMANCE")
    print(f"{'='*70}")
    print(f"Samples to evaluate: {total}")

    for i in tqdm(range(total), desc="New Task Evaluation"):
        sample = eval_data[i]

        # Extract question and answer
        question = sample["0"]["value"]
        try:
            answer = sample["1"]["ground_truth"]["value"]
        except (KeyError, TypeError):
            answer = str(sample["1"])

        prompt = f"Question: {question}\nAnswer:"
        expected_output = str(answer).strip()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the generated part
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0, input_length:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # FIX: Use improved answer checking
        if check_answer_match(prediction, expected_output):
            correct += 1

        # Debug: print first few examples
        if i < 3:
            print(f"\n  Example {i+1}:")
            print(f"    Question: {question[:80]}...")
            print(f"    Expected: {expected_output}")
            print(f"    Predicted: {prediction[:100]}")
            print(f"    Correct: {check_answer_match(prediction, expected_output)}")

    accuracy = (correct / total) * 100

    print(f"\n{'='*70}")
    print(f"NEW TASK EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*70}\n")

    return accuracy


def check_answer_match(prediction, expected):
    """
    Check if prediction matches expected answer.

    FIX: More robust answer matching with:
    - Final answer extraction
    - Numerical comparison with tolerance
    - Stricter substring matching
    """

    def extract_final_number(text):
        """Extract the final numerical answer"""
        text = str(text)

        # Check for boxed answers
        boxed = re.search(r'\\boxed{([^}]+)}', text)
        if boxed:
            nums = re.findall(r'-?\d+\.?\d*', boxed.group(1))
            if nums:
                return float(nums[-1])

        # Check for "answer is X" pattern
        answer_match = re.search(r'(?:answer|result)\s*(?:is|:)\s*(-?\d+\.?\d*)', text, re.I)
        if answer_match:
            return float(answer_match.group(1))

        # Get last number in text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return float(numbers[-1])

        return None

    # Try numerical comparison
    pred_num = extract_final_number(prediction)
    exp_num = extract_final_number(expected)

    if pred_num is not None and exp_num is not None:
        # Allow small tolerance for floating point
        if abs(exp_num) > 1e-6:
            relative_error = abs(pred_num - exp_num) / abs(exp_num)
            return relative_error < 1e-4
        else:
            return abs(pred_num - exp_num) < 1e-6

    # Fallback to string comparison
    pred_clean = str(prediction).lower().strip()
    exp_clean = str(expected).lower().strip()

    # Exact match
    if pred_clean == exp_clean:
        return True

    # Check if expected is in prediction (but be careful with numbers)
    if exp_clean in pred_clean:
        # Make sure it's not a substring of a larger number
        # e.g., "5" should not match if prediction contains "15"
        idx = pred_clean.find(exp_clean)
        # Check character before
        if idx > 0 and pred_clean[idx-1].isdigit():
            return False
        # Check character after
        end_idx = idx + len(exp_clean)
        if end_idx < len(pred_clean) and pred_clean[end_idx].isdigit():
            return False
        return True

    return False