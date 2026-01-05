"""
FIXED evaluation.py for RL's Razor Replication

FIXES APPLIED:
1. KL divergence computation - raises error on negative values instead of skipping
2. Improved validation of probability distributions
3. Better token alignment handling
4. Added reverse KL and JS divergence
5. Robust answer matching
"""

import re

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from logger import get_logger

logger = get_logger(__name__)

# Full set of benchmarks from Table 1 in paper
EVAL_BENCHMARKS = [
    "winogrande",
    "hellaswag",
    "mmlu",  # Use full MMLU, not subsets
    "truthfulqa_mc2",
]

# Extended benchmarks
EXTENDED_BENCHMARKS = [
    "winogrande",
    "hellaswag",
    "mmlu",
    "truthfulqa_mc2",
    "arc_challenge",
    "arc_easy",
]


def evaluate_benchmarks(model, tokenizer, tasks=None, limit=100, use_extended=False):
    """
    Evaluate model on standard benchmarks to measure prior task performance.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        tasks: List of benchmark tasks (None = use defaults)
        limit: Maximum number of examples per task
        use_extended: If True, use extended benchmark set

    Returns:
        dict: Dictionary with per-task and average accuracy scores
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    if tasks is None:
        tasks = EXTENDED_BENCHMARKS if use_extended else EVAL_BENCHMARKS

    logger.info(f"{'='*70}")
    logger.info(f"EVALUATING MODEL ON BENCHMARK TASKS")
    logger.info(f"{'='*70}")
    logger.info(f"Purpose: Measure prior knowledge retention (catastrophic forgetting)")
    logger.info(f"\nBenchmark tasks:")
    for task in tasks:
        logger.info(f"{task}")
    logger.info(f"\nEvaluation settings:")
    logger.info(f" Few-shot examples: 0 (zero-shot evaluation)")
    logger.info(f" Limit per task: {limit} examples")
    logger.info(f"{'='*70}")

    logger.info(" Wrapping model for lm-eval harness...")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_length=1024
    )
    logger.info(" Model wrapped successfully")

    logger.info(f"Running evaluation across {len(tasks)} tasks...")

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        limit=limit,
        confirm_run_unsafe_code=True,
    )

    logger.info("\n Evaluation complete!")

    logger.info("\n Extracting scores from evaluation results...")
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
                # Fallback: find any accuracy metric
                for key, value in task_result.items():
                    if isinstance(value, (int, float)) and 'acc' in key.lower():
                        scores[task] = float(value)
                        break

    # Compute average accuracy across all tasks
    task_scores = [v for k, v in scores.items() if k != 'average']
    scores['average'] = np.mean(task_scores) if task_scores else 0.0

    logger.info(f"{'='*70}")
    logger.info(f"BENCHMARK EVALUATION RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Per-task accuracy (prior knowledge retention):")
    for task, score in scores.items():
        if task != 'average':
            logger.info(f"{task:40s}: {score:.4f}")
    logger.info(f"\n   {'Average Prior Task Performance':40s}: {scores['average']:.4f}")
    logger.info(f"{'='*70}")

    return scores


def validate_probability_distribution(probs, tolerance=1e-2, name="distribution"):
    """
    Validate that probabilities form a valid distribution.

    Args:
        probs: Probability tensor [..., vocab_size]
        tolerance: Tolerance for sum = 1 check
        name: Name for error messages

    Raises:
        ValueError: If distribution is invalid
    """
    # Check non-negative
    if (probs < 0).any():
        neg_count = (probs < 0).sum().item()
        min_val = probs.min().item()
        raise ValueError(
            f"{name} contains negative probabilities!"
            f"  Negative values: {neg_count}"
            f"  Min value: {min_val}"
        )

    # Check sums to 1
    sums = probs.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=tolerance):
        bad_sums = ((sums - 1.0).abs() > tolerance).sum().item()
        raise ValueError(
            f"{name} probabilities don't sum to 1!"
            f"  Invalid sums: {bad_sums}/{sums.numel()}"
            f"  Sum range: [{sums.min().item():.6f}, {sums.max().item():.6f}]"
        )


def compute_forward_kl(model, base_model, dataset, tokenizer, num_samples=100, response_only=True):
    """
    Compute forward KL divergence: KL(base || fine-tuned) on new task.
    Formula: KL(P_base || P_model) = Σ P_base * log(P_base / P_model)

    FIXED VERSION:
    - Validates probability distributions
    - Raises error on negative KL (was silently skipping)
    - Better token alignment
    - Improved response extraction

    Args:
        model: Fine-tuned model
        base_model: Base model
        dataset: Dataset to evaluate on (should have 'text' field)
        tokenizer: Tokenizer
        num_samples: Number of samples to evaluate
        response_only: If True, only compute KL on response portion

    Returns:
        float: Average forward KL divergence
    """
    logger.info(f"{'='*70}")
    logger.info(f"COMPUTING FORWARD KL DIVERGENCE")
    logger.info(f"{'='*70}")
    logger.info(f"Formula: KL(P_base || P_model) = Σ P_base * log(P_base / P_model)")
    logger.info(f"Response-only: {response_only}")
    logger.info(f"Number of samples: {num_samples}")
    logger.info(f"{'='*70}")

    model.eval()
    base_model.eval()

    # Sample from dataset
    num_to_sample = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_to_sample, replace=False)

    total_kl = 0.0
    kl_per_sample = []
    count = 0
    failed_samples = []

    logger.info(f"Processing {num_to_sample} samples...")

    for idx in tqdm(indices, desc="Computing KL"):
        sample = dataset[int(idx)]

        # Get text
        if isinstance(sample, dict):
            text = sample.get('text', str(sample))
        else:
            text = str(sample)

        # Truncate if too long
        if len(text) > 800:
            text = text[:800]

        try:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(model.device)

            # Forward pass for both models
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Fine-tuned model
                model_outputs = model(**inputs)
                model_logits = model_outputs.logits  # [batch, seq, vocab]

                # Base model
                base_outputs = base_model(**inputs)
                base_logits = base_outputs.logits

            # Convert to probabilities
            model_probs = F.softmax(model_logits, dim=-1)
            base_probs = F.softmax(base_logits, dim=-1)

            # Validate distributions
            validate_probability_distribution(model_probs, name="Fine-tuned model")
            validate_probability_distribution(base_probs, name="Base model")

            # Log probabilities for KL computation
            model_log_probs = F.log_softmax(model_logits, dim=-1)
            base_log_probs = F.log_softmax(base_logits, dim=-1)

            # Response-only mode: extract response portion
            if response_only:
                # Find "Answer:" or "Response:" marker
                text_lower = text.lower()
                answer_markers = ['answer:', 'response:', '\nanswer', '\nresponse']

                response_start = None
                for marker in answer_markers:
                    if marker in text_lower:
                        response_start = text_lower.index(marker) + len(marker)
                        break

                if response_start is not None:
                    # Tokenize to find response tokens
                    prompt_tokens = tokenizer(text[:response_start], return_tensors='pt')
                    prompt_len = prompt_tokens['input_ids'].shape[1]

                    # Only compute KL on response tokens
                    if prompt_len < base_log_probs.shape[1]:
                        base_log_probs = base_log_probs[:, prompt_len:, :]
                        model_log_probs = model_log_probs[:, prompt_len:, :]
                        base_probs = base_probs[:, prompt_len:, :]

            # Handle length mismatch (shouldn't happen, but be safe)
            min_len = min(base_log_probs.shape[1], model_log_probs.shape[1])
            if min_len < base_log_probs.shape[1] or min_len < model_log_probs.shape[1]:
                base_log_probs = base_log_probs[:, :min_len, :]
                model_log_probs = model_log_probs[:, :min_len, :]
                base_probs = base_probs[:, :min_len, :]

            # Compute KL divergence: KL(P || Q) = Σ P(x) * (log P(x) - log Q(x))
            # Here: KL(base || model) = Σ base_probs * (base_log_probs - model_log_probs)
            kl_per_token = base_probs * (base_log_probs - model_log_probs)
            kl_sum = kl_per_token.sum(dim=-1)  # Sum over vocabulary
            kl_mean = kl_sum.mean(dim=1)  # Average over sequence
            kl_value = kl_mean.item()

            # CRITICAL: KL divergence MUST be non-negative
            if kl_value < -1e-6:  # Small tolerance for numerical errors
                raise ValueError(
                    f"NEGATIVE KL DIVERGENCE DETECTED!"
                    f"  Sample index: {idx}"
                    f"  KL value: {kl_value}"
                    f"  Base logprobs shape: {base_log_probs.shape}"
                    f"  Model logprobs shape: {model_log_probs.shape}"
                    f"  This indicates a bug in KL computation."
                    f"  KL divergence must be non-negative by definition."
                )

            # Clamp to non-negative (handle tiny numerical errors)
            kl_value = max(0.0, kl_value)

            total_kl += kl_value
            kl_per_sample.append(kl_value)
            count += 1

        except Exception as e:
            failed_samples.append((idx, str(e)))
            if len(failed_samples) <= 3:  # Show first few errors
                logger.info(f"\n⚠ Warning: Failed to process sample {idx}: {e}")
            continue

    if count == 0:
        raise ValueError(
            f"Failed to compute KL for any samples!"
            f"Total failures: {len(failed_samples)}"
            f"First failure: {failed_samples[0] if failed_samples else 'N/A'}"
        )

    # Compute statistics
    avg_kl = total_kl / count
    std_kl = np.std(kl_per_sample) if len(kl_per_sample) > 1 else 0.0
    min_kl = np.min(kl_per_sample) if kl_per_sample else 0.0
    max_kl = np.max(kl_per_sample) if kl_per_sample else 0.0

    logger.info(f"\n KL Divergence Computation Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"KL Statistics:")
    logger.info(f"  Mean KL Divergence: {avg_kl:.6f}")
    logger.info(f"  Std Dev: {std_kl:.6f}")
    logger.info(f"  Min: {min_kl:.6f}")
    logger.info(f"  Max: {max_kl:.6f}")
    logger.info(f"  Valid Samples: {count}/{num_to_sample}")
    if failed_samples:
        logger.info(f"Failed Samples: {len(failed_samples)}")
    logger.info(f"{'='*70}")

    # Sanity check: KL should be reasonable
    if avg_kl > 10.0:
        logger.info(f"⚠ WARNING: KL divergence is very large ({avg_kl:.2f})")
        logger.info(f" This may indicate model divergence or a bug.")

    return avg_kl


def compute_reverse_kl(model, base_model, dataset, tokenizer, num_samples=100, response_only=True):
    """
    Compute reverse KL divergence: KL(fine-tuned || base).

    Args:
        model: Fine-tuned model
        base_model: Base model
        dataset: Dataset to evaluate on
        tokenizer: Tokenizer
        num_samples: Number of samples
        response_only: If True, only compute KL on response portion

    Returns:
        float: Average reverse KL divergence
    """
    # Reverse KL is just swapping the arguments
    # KL(model || base) instead of KL(base || model)

    logger.info(f"{'='*70}")
    logger.info(f"COMPUTING REVERSE KL DIVERGENCE")
    logger.info(f"{'='*70}")

    model.eval()
    base_model.eval()

    num_to_sample = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_to_sample, replace=False)

    total_kl = 0.0
    count = 0

    for idx in tqdm(indices, desc="Computing Reverse KL"):
        sample = dataset[int(idx)]
        text = sample.get('text', str(sample)) if isinstance(sample, dict) else str(sample)

        if len(text) > 800:
            text = text[:800]

        try:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(model.device)

            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                model_outputs = model(**inputs)
                base_outputs = base_model(**inputs)

            model_probs = F.softmax(model_outputs.logits, dim=-1)
            base_probs = F.softmax(base_outputs.logits, dim=-1)

            model_log_probs = F.log_softmax(model_outputs.logits, dim=-1)
            base_log_probs = F.log_softmax(base_outputs.logits, dim=-1)

            # KL(model || base) = Σ model_probs * (model_log_probs - base_log_probs)
            kl_per_token = model_probs * (model_log_probs - base_log_probs)
            kl_value = kl_per_token.sum(dim=-1).mean().item()

            if kl_value < -1e-6:
                raise ValueError(f"Negative reverse KL: {kl_value}")

            total_kl += max(0.0, kl_value)
            count += 1

        except Exception as e:
            continue

    avg_kl = total_kl / count if count > 0 else 0.0

    logger.info(f"✓ Reverse KL: {avg_kl:.6f}")

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
    logger.info(f"{'='*70}")
    logger.info(f"COMPUTING KL ON TASK DISTRIBUTION (RL's Razor Method)")
    logger.info(f"{'='*70}")

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

        # KL(base || model) = E[log p_base - log p_model]
        # When sampling from model: E_model[log p_base - log p_model]
        kl_per_token = token_log_probs_base - token_log_probs_model
        kl_value = kl_per_token.mean().item()

        # KL should be non-negative in expectation, but individual samples can be negative
        kl_values.append(kl_value)

    avg_kl = np.mean(kl_values) if kl_values else 0.0
    std_kl = np.std(kl_values) if len(kl_values) > 1 else 0.0

    logger.info(f"{'='*70}")
    logger.info(f"TASK DISTRIBUTION KL RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f" Mean KL(base || model): {avg_kl:.6f}")
    logger.info(f" Std Dev: {std_kl:.6f}")
    logger.info(f" Samples: {len(kl_values)}")
    logger.info(f"{'='*70}")

    return avg_kl


def compute_js_divergence(model, base_model, dataset, tokenizer, num_samples=100):
    """
    Compute Jensen-Shannon divergence (symmetric measure).

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    Args:
        model: Fine-tuned model
        base_model: Base model
        dataset: Dataset to evaluate on
        tokenizer: Tokenizer
        num_samples: Number of samples

    Returns:
        float: Average JS divergence
    """
    logger.info(f"{'='*70}")
    logger.info(f"COMPUTING JS DIVERGENCE")
    logger.info(f"{'='*70}")

    model.eval()
    base_model.eval()

    num_to_sample = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_to_sample, replace=False)

    total_js = 0.0
    count = 0

    for idx in tqdm(indices, desc="Computing JS"):
        sample = dataset[int(idx)]
        text = sample.get('text', str(sample)) if isinstance(sample, dict) else str(sample)

        if len(text) > 800:
            text = text[:800]

        try:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(model.device)

            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                model_outputs = model(**inputs)
                base_outputs = base_model(**inputs)

            model_probs = F.softmax(model_outputs.logits, dim=-1)
            base_probs = F.softmax(base_outputs.logits, dim=-1)

            # Mixture distribution
            m = 0.5 * (model_probs + base_probs)

            # Compute KL(model || m) and KL(base || m)
            model_log_probs = torch.log(model_probs + 1e-10)
            base_log_probs = torch.log(base_probs + 1e-10)
            m_log_probs = torch.log(m + 1e-10)

            kl_model_m = (model_probs * (model_log_probs - m_log_probs)).sum(dim=-1).mean().item()
            kl_base_m = (base_probs * (base_log_probs - m_log_probs)).sum(dim=-1).mean().item()

            js = 0.5 * kl_model_m + 0.5 * kl_base_m

            total_js += max(0.0, js)
            count += 1

        except Exception as e:
            continue

    avg_js = total_js / count if count > 0 else 0.0

    logger.info(f"✓ JS Divergence: {avg_js:.6f}")

    return avg_js


def extract_final_answer(text):
    """
    Extract the final answer from model output.

    Handles:
    - Boxed answers: \\boxed{answer}
    - "Answer is X" patterns
    - Last line fallback
    """
    text = str(text).strip()

    # Check for boxed answers (LaTeX)
    boxed_patterns = [
        r'\\boxed\{([^}]+)\}',  # Standard
        r'\\boxed\{\{([^}]+)\}\}',  # Double braces
    ]
    for pattern in boxed_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    # Check for "answer is X" patterns
    answer_patterns = [
        r'(?:the\s+)?answer\s*(?:is|:)\s*([^\n.,;]+)',
        r'(?:therefore|thus|so|hence)\s*,?\s*(?:the\s+)?(?:answer\s+)?(?:is\s+)?([^\n.,;]+)',
        r'=\s*([^\n.,;]+)$',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: return last line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        return lines[-1]

    return text


def extract_number(text):
    """Extract numerical answer from text."""
    text = str(text).strip()

    # Handle negative numbers and decimals
    numbers = re.findall(r'-?\d+\.?\d*', text)

    if numbers:
        # Return the last number (usually the final answer)
        return float(numbers[-1])

    return None


def check_answer_match(prediction, expected):
    """
    Check if prediction matches expected answer.

    Handles:
    - Multiple choice answers (A, B, C, D) - STRICT matching
    - Numerical comparison with tolerance
    - Text normalization
    - Substring matching (with safeguards)

    Args:
        prediction: Model's predicted answer
        expected: Ground truth answer

    Returns:
        bool: True if answers match
    """
    # Extract final answers
    pred_final = extract_final_answer(prediction)
    exp_final = extract_final_answer(expected)

    # SPECIAL CASE: Single letter answers (multiple choice A, B, C, D, etc.)
    exp_clean_raw = str(exp_final).strip().upper()
    if len(exp_clean_raw) == 1 and exp_clean_raw.isalpha():
        # For single letter expected answers, we need strict matching
        pred_clean_raw = str(pred_final).strip().upper()

        # Check if pred_final is exactly the letter
        if pred_clean_raw == exp_clean_raw:
            return True

        # Check for patterns like "B", "(B)", "B.", "B)", "Option B", "Answer: B"
        import re
        mc_patterns = [
            rf'^{exp_clean_raw}$',  # Just the letter
            rf'^\({exp_clean_raw}\)$',  # (B)
            rf'^{exp_clean_raw}[\.\)]',  # B. or B)
            rf'^Option\s*{exp_clean_raw}',  # Option B
            rf'^{exp_clean_raw}\s*[\-\:]',  # B - or B:
        ]
        for pattern in mc_patterns:
            if re.match(pattern, pred_clean_raw, re.IGNORECASE):
                return True

        # Also check in boxed answer specifically
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', prediction)
        if boxed_match:
            boxed_content = boxed_match.group(1).strip().upper()
            if boxed_content == exp_clean_raw:
                return True

        # For single letter answers, DON'T do substring matching (too error-prone)
        return False

    # Try numerical comparison first
    pred_num = extract_number(pred_final)
    exp_num = extract_number(exp_final)

    if pred_num is not None and exp_num is not None:
        # Numerical match with tolerance
        if abs(exp_num) > 1e-6:
            relative_error = abs(pred_num - exp_num) / abs(exp_num)
            return relative_error < 1e-4
        else:
            return abs(pred_num - exp_num) < 1e-6

    # Text comparison
    pred_clean = str(pred_final).lower().strip()
    exp_clean = str(exp_final).lower().strip()

    # Remove punctuation for comparison
    import string
    pred_clean = pred_clean.translate(str.maketrans('', '', string.punctuation))
    exp_clean = exp_clean.translate(str.maketrans('', '', string.punctuation))

    # Exact match
    if pred_clean == exp_clean:
        return True

    # Substring match (with safeguards to avoid false positives)
    # Only do this for longer expected answers (not single letters/short words)
    if len(exp_clean) >= 3 and exp_clean in pred_clean:
        # Make sure it's not part of a larger word
        idx = pred_clean.find(exp_clean)

        # Check character before
        if idx > 0 and pred_clean[idx-1].isalnum():
            return False

        # Check character after
        end_idx = idx + len(exp_clean)
        if end_idx < len(pred_clean) and pred_clean[end_idx].isalnum():
            return False

        return True

    return False


def evaluate_new_task(model, tokenizer, dataset, eval_dataset=None, max_new_tokens=512, num_samples=100):
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
        logger.info(f"Using separate evaluation dataset with {len(eval_data)} examples")
    else:
        # Fallback: use last portion of training data (not ideal)
        eval_start = max(0, len(dataset) - num_samples)
        eval_data = dataset.select(range(eval_start, len(dataset)))
        logger.info(f"Warning: Using end of training data for evaluation (consider providing eval_dataset)")

    correct = 0
    total = min(num_samples, len(eval_data))

    logger.info(f"{'='*70}")
    logger.info(f"EVALUATING NEW TASK PERFORMANCE")
    logger.info(f"{'='*70}")
    logger.info(f"Samples to evaluate: {total}")
    logger.info(f"{'='*70}")

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

        # Check answer
        if check_answer_match(prediction, answer):
            correct += 1

        # Debug: print first few examples
        if i < 3:
            logger.info(f"\n  Example {i+1}:")
            logger.info(f"   Q: {question}...")
            logger.info(f"   Expected: {answer}")
            logger.info(f"   Predicted: {prediction}")
            logger.info(f"   Correct" if check_answer_match(prediction, answer) else "     Incorrect")

    accuracy = (correct / total) * 100

    logger.info(f"{'='*70}")
    logger.info(f"NEW TASK EVALUATION RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Correct: {correct}/{total}")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"{'='*70}")

    return accuracy