"""
Corrected evaluation.py with bug fixes for RL's Razor replication

UPDATES:
- Added TruthfulQA benchmark
- Added IFEval benchmark
- Added HumanEval benchmark
- Improved KL divergence computation
- Better answer matching
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import re

# Full set of benchmarks from Table 1 in paper
EVAL_BENCHMARKS = [
    "winogrande",
    "hellaswag",
    "mmlu_high_school_mathematics",
    "mmlu_high_school_computer_science",
    "truthfulqa_mc2",  # Added: TruthfulQA multiple choice
]

# Extended benchmarks (optional, more comprehensive)
EXTENDED_BENCHMARKS = [
    "winogrande",
    "hellaswag",
    "mmlu_high_school_mathematics",
    "mmlu_high_school_computer_science",
    "truthfulqa_mc2",
    "arc_challenge",
    "arc_easy",
]


def evaluate_benchmarks(model, tokenizer, tasks=None, limit=100, use_extended=False):
    """
    Evaluate model on standard benchmarks to measure prior task performance.

    Args:
        model: Model to evaluate (should be fine-tuned model)
        tokenizer: Tokenizer for the model
        tasks: List of benchmark tasks to evaluate on (None = use defaults)
        limit: Maximum number of examples per task
        use_extended: If True, use extended benchmark set

    Returns:
        dict: Dictionary with per-task and average accuracy scores
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    if tasks is None:
        tasks = EXTENDED_BENCHMARKS if use_extended else EVAL_BENCHMARKS

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

    print(f" Running evaluation across {len(tasks)} tasks...")

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        limit=limit,
        confirm_run_unsafe_code=True,
    )

    print("\n Evaluation complete!")

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
            elif 'mc2' in task_result:  # TruthfulQA specific
                scores[task] = task_result['mc2']
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


def evaluate_humaneval(model, tokenizer, limit=50):
    """
    Evaluate model on HumanEval code generation benchmark.

    HumanEval requires generating code completions and executing them,
    so it needs special handling separate from lm-eval.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        limit: Maximum number of problems to evaluate

    Returns:
        dict: HumanEval scores including pass@1
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING ON HUMANEVAL")
    print(f"{'='*70}")

    try:
        from human_eval.data import read_problems
        from human_eval.evaluation import evaluate_functional_correctness
        import tempfile
        import json
        import os
    except ImportError:
        print("⚠️ human_eval not installed. Install with: pip install human-eval")
        print("Skipping HumanEval evaluation.")
        return {'pass@1': 0.0, 'error': 'human_eval not installed'}

    problems = read_problems()
    problem_ids = list(problems.keys())[:limit]

    print(f"Evaluating on {len(problem_ids)} problems...")

    samples = []

    for task_id in tqdm(problem_ids, desc="Generating completions"):
        problem = problems[task_id]
        prompt = problem['prompt']

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )

        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0, input_length:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Stop at first function end or class definition
        stop_sequences = ['\nclass ', '\ndef ', '\n#', '\nif __name__']
        for stop in stop_sequences:
            if stop in completion:
                completion = completion[:completion.index(stop)]

        samples.append({
            'task_id': task_id,
            'completion': completion
        })

    # Write samples to temp file and evaluate
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
        temp_path = f.name

    try:
        results = evaluate_functional_correctness(temp_path)
        pass_at_1 = results['pass@1']
    except Exception as e:
        print(f"Error evaluating: {e}")
        pass_at_1 = 0.0
    finally:
        os.unlink(temp_path)

    print(f"\n{'='*70}")
    print(f"HUMANEVAL RESULTS")
    print(f"{'='*70}")
    print(f"  Pass@1: {pass_at_1:.4f}")
    print(f"{'='*70}\n")

    return {'pass@1': pass_at_1}


def evaluate_ifeval(model, tokenizer, limit=100):
    """
    Evaluate model on IFEval (Instruction Following Evaluation).

    IFEval tests whether models follow specific formatting instructions
    like "write in all caps" or "include exactly 3 bullet points".

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        limit: Maximum number of examples

    Returns:
        dict: IFEval scores
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING ON IFEVAL")
    print(f"{'='*70}")

    try:
        from datasets import load_dataset
    except ImportError:
        print("⚠️ datasets not installed")
        return {'accuracy': 0.0, 'error': 'datasets not installed'}

    try:
        # Load IFEval dataset
        dataset = load_dataset("google/IFEval", split="train")
        dataset = dataset.select(range(min(limit, len(dataset))))
    except Exception as e:
        print(f"⚠️ Could not load IFEval dataset: {e}")
        return {'accuracy': 0.0, 'error': str(e)}

    print(f"Evaluating on {len(dataset)} examples...")

    correct = 0
    total = 0

    for example in tqdm(dataset, desc="Evaluating IFEval"):
        prompt = example['prompt']
        instruction_id_list = example.get('instruction_id_list', [])

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0, input_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Check if instructions are followed
        # This is a simplified check - full IFEval has complex verification
        instructions_followed = check_ifeval_instructions(response, instruction_id_list)

        if instructions_followed:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0

    print(f"\n{'='*70}")
    print(f"IFEVAL RESULTS")
    print(f"{'='*70}")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"{'='*70}\n")

    return {'accuracy': accuracy, 'correct': correct, 'total': total}


def check_ifeval_instructions(response: str, instruction_ids: list) -> bool:
    """
    Check if response follows IFEval instructions.

    This is a simplified implementation. Full IFEval uses more sophisticated checks.
    """
    response_lower = response.lower()

    for instruction_id in instruction_ids:
        # Common instruction patterns
        if 'length' in instruction_id:
            # Check word count constraints
            word_count = len(response.split())
            if 'at_least' in instruction_id:
                try:
                    min_words = int(re.search(r'\d+', instruction_id).group())
                    if word_count < min_words:
                        return False
                except:
                    pass

        elif 'format' in instruction_id:
            # Check formatting constraints
            if 'bullet' in instruction_id:
                if '•' not in response and '-' not in response and '*' not in response:
                    return False

        elif 'keyword' in instruction_id:
            # Check if specific keywords are present
            pass  # Would need the actual keywords to check

        elif 'case' in instruction_id:
            if 'upper' in instruction_id:
                if response != response.upper():
                    return False
            elif 'lower' in instruction_id:
                if response != response.lower():
                    return False

    return True


def compute_forward_kl(model, base_model, dataset, tokenizer, num_samples=100, response_only=True):
    """
    Compute forward KL divergence KL(base || model).

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

            # Identify prompt vs response boundary
            if response_only:
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

            # Compute KL only on response tokens
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

def compute_kl_on_task_distribution(model, base_model, tokenizer, prompts, num_samples=100, max_new_tokens=64):
    """
    Compute KL(π_base || π_finetuned) on the NEW TASK distribution.

    This is the CORRECT metric from the RL's Razor paper:
    - Generate completions from the fine-tuned model
    - Compare log probabilities of both models on those completions

    Args:
        model: Fine-tuned model
        base_model: Base/reference model
        tokenizer: Tokenizer
        prompts: List of task prompts (questions without answers)
        num_samples: Number of samples to evaluate
        max_new_tokens: Max tokens to generate per prompt

    Returns:
        float: Average KL divergence on task distribution
    """
    print(f"\n{'='*70}")
    print(f"COMPUTING KL ON TASK DISTRIBUTION (RL's Razor Method)")
    print(f"{'='*70}")

    model.eval()
    base_model.eval()

    device = next(model.parameters()).device
    base_device = next(base_model.parameters()).device

    kl_values = []

    prompts_to_use = prompts[:num_samples]

    for prompt in tqdm(prompts_to_use, desc="Computing task KL"):
        # Tokenize prompt
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = prompt_ids['input_ids'].to(device)
        prompt_len = input_ids.shape[1]

        # Generate completion from fine-tuned model (this IS the task distribution)
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id
            )

        # Get the full sequence (prompt + generated)
        full_sequence = generated[0]

        if full_sequence.shape[0] <= prompt_len:
            continue  # No tokens generated

        # Get log probs from both models on the SAME completion
        with torch.no_grad():
            # Fine-tuned model log probs
            model_outputs = model(full_sequence.unsqueeze(0))
            model_logits = model_outputs.logits[0]  # [seq_len, vocab]

            # Base model log probs
            base_outputs = base_model(full_sequence.unsqueeze(0).to(base_device))
            base_logits = base_outputs.logits[0].to(device)  # [seq_len, vocab]

        # Compute log probs for the generated tokens only (not prompt)
        # For autoregressive: logits[i] predicts token[i+1]
        gen_tokens = full_sequence[prompt_len:]  # The generated tokens

        if len(gen_tokens) == 0:
            continue

        # Get logits that predict the generated tokens
        pred_logits_model = model_logits[prompt_len-1:-1]  # Logits predicting gen_tokens
        pred_logits_base = base_logits[prompt_len-1:-1]

        if pred_logits_model.shape[0] != len(gen_tokens):
            # Align lengths
            min_len = min(pred_logits_model.shape[0], len(gen_tokens))
            pred_logits_model = pred_logits_model[:min_len]
            pred_logits_base = pred_logits_base[:min_len]
            gen_tokens = gen_tokens[:min_len]

        # Log softmax for numerical stability
        log_probs_model = F.log_softmax(pred_logits_model, dim=-1)
        log_probs_base = F.log_softmax(pred_logits_base, dim=-1)

        # Get log prob of each generated token
        token_log_probs_model = log_probs_model[range(len(gen_tokens)), gen_tokens]
        token_log_probs_base = log_probs_base[range(len(gen_tokens)), gen_tokens]

        # KL(base || model) ≈ E_model[log p_base - log p_model]
        # Since we sample from model, this is: mean(log_p_base - log_p_model)
        kl_per_token = token_log_probs_model - token_log_probs_base
        kl_value = kl_per_token.mean().item()

        # KL should be non-negative in expectation, but individual samples can be negative
        kl_values.append(kl_value)

    avg_kl = np.mean(kl_values) if kl_values else 0.0
    std_kl = np.std(kl_values) if len(kl_values) > 1 else 0.0

    print(f"\n{'='*70}")
    print(f"TASK DISTRIBUTION KL RESULTS")
    print(f"{'='*70}")
    print(f"  Mean KL(base || model): {avg_kl:.6f}")
    print(f"  Std Dev: {std_kl:.6f}")
    print(f"  Samples: {len(kl_values)}")
    print(f"{'='*70}\n")

    return avg_kl


def evaluate_new_task(model, tokenizer, dataset, eval_dataset=None, max_new_tokens=64, num_samples=100):
    """
    Evaluate New Task performance (NT).

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

    # Use separate eval dataset if provided
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

        prompt = f"Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer:"
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

        # Use improved answer checking
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

    More robust answer matching with:
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
        idx = pred_clean.find(exp_clean)
        if idx > 0 and pred_clean[idx-1].isdigit():
            return False
        end_idx = idx + len(exp_clean)
        if end_idx < len(pred_clean) and pred_clean[end_idx].isdigit():
            return False
        return True

    return False


def run_full_evaluation(model, base_model, tokenizer, dataset, formatted_dataset_kl):
    """
    Run complete evaluation suite including all benchmarks.

    Args:
        model: Fine-tuned model
        base_model: Base model for KL computation
        tokenizer: Tokenizer
        dataset: Original dataset
        formatted_dataset_kl: Formatted dataset for KL computation

    Returns:
        dict: Complete evaluation results
    """
    print(f"\n{'='*70}")
    print(f"RUNNING FULL EVALUATION SUITE")
    print(f"{'='*70}")

    results = {}

    # 1. Standard benchmarks
    print("\n[1/4] Evaluating standard benchmarks...")
    results['benchmarks'] = evaluate_benchmarks(model, tokenizer)

    # 2. New task performance
    print("\n[2/4] Evaluating new task performance...")
    results['new_task'] = evaluate_new_task(model, tokenizer, dataset)

    # 3. KL divergence
    print("\n[3/4] Computing KL divergence...")
    results['kl_divergence'] = compute_forward_kl(
        model, base_model, formatted_dataset_kl, tokenizer,
        response_only=True
    )

    # 4. Optional: HumanEval (if available)
    print("\n[4/4] Attempting HumanEval evaluation...")
    try:
        results['humaneval'] = evaluate_humaneval(model, tokenizer, limit=20)
    except Exception as e:
        print(f"  HumanEval skipped: {e}")
        results['humaneval'] = {'pass@1': None, 'error': str(e)}

    # Summary
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Prior Task (PT): {results['benchmarks']['average']:.4f}")
    print(f"  New Task (NT): {results['new_task']:.2f}%")
    print(f"  KL Divergence: {results['kl_divergence']:.6f}")
    if results['humaneval'].get('pass@1') is not None:
        print(f"  HumanEval Pass@1: {results['humaneval']['pass@1']:.4f}")
    print(f"{'='*70}\n")

    return results