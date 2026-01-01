"""
Training utilities for SFT and GRPO training.

This module provides functions for:
- Supervised Fine-Tuning (SFT) training
- GRPO (Group Relative Policy Optimization) training
- Support for multiple dataset formats
"""

import logging
import re

import torch
from transformers import TrainingArguments
from trl import GRPOConfig, GRPOTrainer, SFTTrainer

logger = logging.getLogger(__name__)


def train_sft(model, dataset, tokenizer, learning_rate=3e-5, batch_size=32, epochs=1, max_samples=500, eval_dataset=None):
    """
    Train model using Supervised Fine-Tuning (SFT).

    Args:
        model: Model to train
        dataset: Training dataset
        tokenizer: Tokenizer
        learning_rate: Learning rate
        batch_size: Batch size per device
        epochs: Number of epochs
        max_samples: Maximum training samples
        eval_dataset: Optional evaluation dataset

    Returns:
        tuple: (trained_model, trainer, NT_score)
    """
    logger.info("=" * 70)
    logger.info("STARTING SFT TRAINING")
    logger.info("=" * 70)
    logger.info(f"Hyperparameters: LR={learning_rate}, Batch Size={batch_size}, "
                f"Epochs={epochs}, Max Samples={max_samples}")

    # Enable gradient checkpointing to save memory
    logger.info("Enabling gradient checkpointing to save memory...")
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled")

    # Formating the dataset, creating a 'text' field
    def format_dataset(examples):
        # Converting nested structure to text format
        texts = []

        # Determine format from first example
        if '0' in examples and '1' in examples:
            # Open-Reasoner format
            for i in range(len(examples['0'])):
                question = examples['0'][i]['value']
                try:
                    answer = examples['1'][i]['ground_truth']['value']
                except (KeyError, TypeError):
                    answer = str(examples['1'][i])

                text = f"Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer: {answer}"
                texts.append(text)

        elif 'question' in examples and 'answer' in examples:
            # GSM8K format
            for i in range(len(examples['question'])):
                question = examples['question'][i]
                answer = examples['answer'][i]

                text = f"Question: {question}\nAnswer: {answer}"
                texts.append(text)

        elif 'instruction' in examples and 'output' in examples:
            # Alpaca format
            for i in range(len(examples['instruction'])):
                instruction = examples['instruction'][i]
                input_text = examples.get('input', [''] * len(examples['instruction']))[i]
                output = examples['output'][i]

                if input_text:
                    text = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
                else:
                    text = f"Instruction: {instruction}\nResponse: {output}"
                texts.append(text)

        else:
            raise ValueError(f"Unknown dataset format. Keys: {examples.keys()}")

        return {'text': texts}

    logger.info("Formatting dataset...")
    try:
        formatted_dataset = dataset.map(format_dataset, batched=True, remove_columns=dataset.column_names)
        logger.info(f"Dataset formatted: {len(formatted_dataset)} examples")
    except Exception as e:
        logger.error(f"Error formatting dataset: {e}")
        raise

    # Select subset
    formatted_dataset = formatted_dataset.select(range(min(max_samples, len(formatted_dataset))))
    logger.info(f"Using {len(formatted_dataset)} examples for training")

    # Training arguments
    gradient_accumulation_steps = 4
    effective_batch_size = batch_size * gradient_accumulation_steps
    logger.info(f"Effective batch size: {effective_batch_size}")

    training_args = TrainingArguments(
        output_dir=f"./results/sft_lr{learning_rate}_bs{batch_size}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=50,
        bf16=True,
        max_grad_norm=1.0,
        weight_decay=0,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",
        report_to="none",
        gradient_checkpointing=True,
    )

    # Formatting function for SFTTrainer
    def formatting_func(examples):
        return examples['text']

    logger.info("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    logger.info("SFT Trainer initialized")

    logger.info("=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)

    trainer.train()

    logger.info("=" * 70)
    logger.info("SFT TRAINING COMPLETE")
    logger.info("=" * 70)

    # Evaluate on new task
    from evaluation import evaluate_new_task
    NT = evaluate_new_task(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        eval_dataset=eval_dataset,
        num_samples=100
    )

    return model, trainer, NT


def extract_boxed_answer(text):
    """
    Extract answer from \\boxed{} notation.

    Handles:
    - Standard: \\boxed{answer}
    - Double braces: \\boxed{{answer}}
    - Nested braces (counts braces)
    """
    text = str(text)

    # Try simple pattern first
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()

    # Try double braces
    match = re.search(r'\\boxed\{\{([^}]+)\}\}', text)
    if match:
        return match.group(1).strip()

    # Handle nested braces by finding matching braces
    start = text.find('\\boxed{')
    if start != -1:
        start += 7  # len('\\boxed{')
        brace_count = 1
        i = start

        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1

        if brace_count == 0:
            return text[start:i-1].strip()

    return None


def extract_final_answer(text):
    """
    Extract final answer from text.

    Priority order:
    1. Boxed answer
    2. "Answer is X" pattern
    3. Last line
    """
    text = str(text).strip()

    # Check for boxed answer
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    # Check for "answer is X" patterns
    answer_patterns = [
        r'(?:the\s+)?answer\s*(?:is|:)\s*([^\n.,;]+)',
        r'(?:therefore|thus|so|hence)\s*,?\s*([^\n.,;]+)',
        r'=\s*([^\n.,;]+)$',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: last line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines[-1] if lines else text


def extract_number(text):
    """Extract numerical value from text."""
    text = str(text).strip()

    # First try to extract from boxed answer
    boxed = extract_boxed_answer(text)
    if boxed:
        text = boxed

    # Find all numbers (including negative and decimals)
    numbers = re.findall(r'-?\d+\.?\d*', text)

    if numbers:
        # Return last number (usually the final answer)
        try:
            return float(numbers[-1])
        except ValueError:
            return None

    return None


def check_answer_correctness(predicted_answer, ground_truth_answer):
    """
    Check if predicted answer matches ground truth.

    FIXES APPLIED:
    - Better final answer extraction
    - Robust boxed answer handling
    - Numerical comparison with tolerance
    - Text normalization

    Args:
        predicted_answer: Model's predicted answer
        ground_truth_answer: Correct answer

    Returns:
        bool: True if answers match
    """
    # Extract final answers
    pred_final = extract_final_answer(predicted_answer)
    true_final = extract_final_answer(ground_truth_answer)

    # Try numerical comparison
    pred_num = extract_number(pred_final)
    true_num = extract_number(true_final)

    if pred_num is not None and true_num is not None:
        # Numerical match with tolerance
        if abs(true_num) > 1e-6:
            relative_error = abs(pred_num - true_num) / abs(true_num)
            return relative_error < 1e-4
        else:
            return abs(pred_num - true_num) < 1e-6

    # Text comparison
    pred_clean = str(pred_final).lower().strip()
    true_clean = str(true_final).lower().strip()

    # Remove punctuation
    import string
    pred_clean = pred_clean.translate(str.maketrans('', '', string.punctuation))
    true_clean = true_clean.translate(str.maketrans('', '', string.punctuation))

    # Exact match
    if pred_clean == true_clean:
        return True

    # Substring match (with safeguards) -
    if true_clean and true_clean in pred_clean:
        # Make sure not part of larger number
        idx = pred_clean.find(true_clean)

        # Check before
        if idx > 0 and pred_clean[idx-1].isdigit():
            return False

        # Check after
        end_idx = idx + len(true_clean)
        if end_idx < len(pred_clean) and pred_clean[end_idx].isdigit():
            return False

        return True

    return False


def compute_partial_reward(completion, ground_truth):
    """
    Compute a partial reward for GRPO training.
    This provides gradient signal even when exact match fails.

    Returns:
        float: Reward between 0 and 1
    """
    reward = 0.0

    # Base reward for generating any numeric content
    if any(c.isdigit() for c in completion):
        reward = 0.1

    # Reward for showing work / reasoning
    reasoning_indicators = ['because', 'therefore', 'since', 'so', 'thus', 'step', '=']
    if any(ind in completion.lower() for ind in reasoning_indicators):
        reward += 0.1

    # Reward for format (having "answer" or similar)
    if 'answer' in completion.lower() or '=' in completion:
        reward += 0.1

    # Exact match gets full reward
    if check_answer_correctness(completion, ground_truth):
        reward = 1.0
    else:
        # Partial credit if numbers are close
        pred_num = extract_number_from_text(completion)
        true_num = extract_number_from_text(ground_truth)
        if pred_num is not None and true_num is not None:
            # Partial credit based on how close the answer is
            if true_num != 0:
                relative_error = abs(pred_num - true_num) / abs(true_num)
                if relative_error < 0.1:  # Within 10%
                    reward = max(reward, 0.7)
                elif relative_error < 0.5:  # Within 50%
                    reward = max(reward, 0.4)

    return reward


def extract_number_from_text(text):
    """Extract the most likely final answer number from text"""
    text = str(text)

    # Check for boxed answers first
    boxed = re.search(r'\\boxed{([^}]+)}', text)
    if boxed:
        nums = re.findall(r'-?\d+\.?\d*', boxed.group(1))
        if nums:
            return float(nums[-1])

    # Find all numbers
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return float(numbers[-1])
    return None


def normalize_question(text):
    """Extract and normalize question for consistent hashing."""
    # Remove common prefixes
    text = re.sub(r'^(Question:|Q:)\s*', '', text.strip())
    # Remove whitespace variations
    text = ' '.join(text.split())
    return text.lower()


def question_to_key(question):
    """Create stable key from question text using hash."""
    import hashlib
    normalized = normalize_question(question)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def train_grpo(model, dataset, tokenizer, learning_rate=2e-5, batch_size=32, max_samples=500, eval_dataset=None):
    """
    Train model using Group Relative Policy Optimization (GRPO).

    Args:
        model: Model to train
        dataset: Training dataset
        tokenizer: Tokenizer
        learning_rate: Learning rate
        batch_size: Batch size per device
        max_samples: Maximum training samples
        eval_dataset: Optional evaluation dataset
    """
    logger.info("=" * 70)
    logger.info("STARTING GRPO (RL) TRAINING")
    logger.info("=" * 70)
    logger.info(f"Configuration: LR={learning_rate}, Batch Size={batch_size}, "
                f"Max Samples={max_samples}, KL Coefficient=0.0 (implicit KL minimization)")

    model.gradient_checkpointing_enable()

    # FIXED: Store answers using robust question hashing
    question_to_answer = {}

    def format_for_grpo(examples):
        prompts = []
        for i in range(len(examples['0'])):
            question = examples['0'][i]['value']
            try:
                answer = examples['1'][i]['ground_truth']['value']
            except (KeyError, TypeError):
                answer = str(examples['1'][i])

            # Clean prompt WITHOUT ground truth embedded
            prompt = f"Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer:"
            prompts.append(prompt)

            # Store answer with robust hash key
            key = question_to_key(question)
            question_to_answer[key] = answer

        return {'prompt': prompts}

    logger.info("Formatting dataset for GRPO training...")
    formatted_dataset = dataset.map(format_for_grpo, batched=True, remove_columns=dataset.column_names)
    formatted_dataset = formatted_dataset.select(range(min(max_samples, len(formatted_dataset))))
    logger.info(f"Using {len(formatted_dataset)} examples for GRPO training")
    logger.info(f"Answer lookup table has {len(question_to_answer)} entries")

    gradient_accumulation_steps = 4
    logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")

    grpo_config = GRPOConfig(
        output_dir=f"./results/grpo_lr{learning_rate}_bs{batch_size}",
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=50,
        max_grad_norm=1.0,
        logging_steps=10,
        report_to="none",
        # Note: GRPO has implicit KL minimization (no explicit kl_coef parameter)
        gradient_checkpointing=True,
    )

    # Track reward statistics for debugging
    reward_stats = {'found': 0, 'not_found': 0, 'correct': 0, 'incorrect': 0}

    def reward_fn(completions, prompts, **kwargs):
        """
        FIXED: Reward function using robust question hashing.
        """
        # Validate inputs
        if len(completions) != len(prompts):
            raise ValueError(
                f"Length mismatch: {len(completions)} completions vs {len(prompts)} prompts\n"
                f"This should never happen - check GRPO trainer configuration."
            )

        rewards = []

        for completion, prompt in zip(completions, prompts):
            # Extract question from prompt using regex
            q_match = re.search(r'Question:\s*(.+?)\nPlease reason', prompt, re.DOTALL)
            if q_match:
                question_text = q_match.group(1).strip()
                key = question_to_key(question_text)
                ground_truth = question_to_answer.get(key, "")
            else:
                # Fallback: try to extract any question-like content
                q_match_fallback = re.search(r'Question:\s*(.+?)\n', prompt)
                if q_match_fallback:
                    key = question_to_key(q_match_fallback.group(1))
                    ground_truth = question_to_answer.get(key, "")
                else:
                    ground_truth = ""

            if not ground_truth:
                reward_stats['not_found'] += 1
                if reward_stats['not_found'] <= 3:  # Only log first few
                    logger.warning(f"No ground truth found for prompt: {prompt[:80]}...")
                rewards.append(0.0)
                continue

            reward_stats['found'] += 1

            # Binary outcome supervision (paper's approach)
            if check_answer_correctness(completion, ground_truth):
                rewards.append(1.0)
                reward_stats['correct'] += 1
            else:
                rewards.append(0.0)
                reward_stats['incorrect'] += 1

        return rewards

    logger.info("Initializing GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    logger.info("=" * 70)
    logger.info("BEGINNING GRPO TRAINING LOOP")
    logger.info("=" * 70)

    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    logger.info("=" * 70)
    logger.info("GRPO TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Reward Statistics: Answers found={reward_stats['found']}, "
                f"Not found={reward_stats['not_found']}, "
                f"Correct={reward_stats['correct']}, Incorrect={reward_stats['incorrect']}")
    if reward_stats['found'] > 0:
        accuracy = reward_stats['correct'] / reward_stats['found'] * 100
        logger.info(f"Training accuracy: {accuracy:.1f}%")
    logger.info("=" * 70)

    # Evaluate on new task
    from evaluation import evaluate_new_task
    NT = evaluate_new_task(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        eval_dataset=eval_dataset,
        num_samples=100
    )

    return model, trainer, NT

# Import numpy for reward logging
import numpy as np