"""
Corrected training.py with bug fixes for RL's Razor replication
"""

from transformers import TrainingArguments
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
from evaluation import evaluate_new_task
import logging
import re

logger = logging.getLogger(__name__)


def train_sft(model, dataset, tokenizer, learning_rate=3e-5, batch_size=32, epochs=1, max_samples=500):
    """
    Train model using Supervised Fine-Tuning (SFT).

    FIXES APPLIED:
    - max_samples parameter instead of hardcoded 50
    - Gradient accumulation documented

    Args:
        model: The model to train
        dataset: Training dataset
        tokenizer: Tokenizer for the model
        learning_rate: Learning rate for training
        batch_size: Batch size per device
        epochs: Number of training epochs
        max_samples: Maximum number of training samples (default 500)

    Returns:
        tuple: (trained_model, trainer, NT_score)
    """
    print(f"\n{'='*70}")
    print(f"STARTING SFT TRAINING")
    print(f"{'='*70}")
    print(f"Hyperparameters:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Max Samples: {max_samples}")
    print(f"{'='*70}\n")

    # Enable gradient checkpointing to save memory
    print("Enabling gradient checkpointing to save memory...")
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")

    # Formating the dataset, creating a 'text' field
    def format_dataset(examples):
        # Converting nested structure to text format
        texts = []
        for i in range(len(examples['0'])):
            question = examples['0'][i]['value']

            # Get answer from ground_truth if available
            try:
                answer = examples['1'][i]['ground_truth']['value']
            except (KeyError, TypeError):
                answer = str(examples['1'][i])

            # Format as conversation
            text = f"Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer: {answer}"

            texts.append(text)

        return {'text': texts}

    # Apply formatting
    print("\n Formatting dataset for SFT training...")
    formatted_dataset = dataset.map(format_dataset, batched=True, remove_columns=dataset.column_names)
    print(f"Dataset formatted: {len(formatted_dataset)} total examples")

    # FIX: Use configurable max_samples instead of hardcoded 50
    formatted_dataset = formatted_dataset.select(range(min(max_samples, len(formatted_dataset))))
    print(f"Using {len(formatted_dataset)} examples for training")

    # Calculate effective batch size for logging
    gradient_accumulation_steps = 4
    effective_batch_size = batch_size * gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch_size} (batch_size={batch_size} x grad_accum={gradient_accumulation_steps})")

    print("\n Setting up training arguments...")
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
    print("Training arguments configured")

    # Define formatting function for SFTTrainer
    def formatting_func(examples):
        return examples['text']

    print("\nInitializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    print("SFT Trainer initialized")

    print(f"\n{'='*70}")
    print(f"BEGINNING TRAINING LOOP (lr={learning_rate}, bs={batch_size}, epochs={epochs})")
    print(f"{'='*70}")
    print("This will take a while... Progress will be shown below:\n")

    trainer.train()

    print(f"\n{'='*70}")
    print(f"SFT TRAINING COMPLETE")
    print(f"{'='*70}\n")

    NT = evaluate_new_task(model=model, tokenizer=tokenizer, dataset=dataset)

    return model, trainer, NT


def check_answer_correctness(predicted_answer, ground_truth_answer):
    """
    Check if predicted answer matches ground truth.
    Handles both numerical and text-based answers.

    FIXES APPLIED:
    - Better final answer extraction
    - Support for boxed answers
    - Cleaner number extraction

    Args:
        predicted_answer: Model's predicted answer
        ground_truth_answer: Correct answer

    Returns:
        bool: True if answers match, False otherwise
    """

    def extract_final_answer(text):
        """Extract the final answer from model output"""
        text = str(text)

        # Check for boxed answers (common in math)
        boxed = re.search(r'\\boxed{([^}]+)}', text)
        if boxed:
            return boxed.group(1).strip()

        # Check for "the answer is X" patterns
        answer_patterns = [
            r'(?:the\s+)?answer\s*(?:is|:)\s*([^\n.,]+)',
            r'(?:therefore|thus|so)\s*,?\s*([^\n.,]+)',
            r'=\s*([^\n.,]+)$',
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(1).strip()

        # Return the last line as fallback
        lines = text.strip().split('\n')
        return lines[-1].strip() if lines else text

    def extract_number(text):
        """Extract numerical answer from text"""
        text = str(text).lower()

        # Remove common prefixes
        text = re.sub(r'(the answer is|therefore|thus|so|final answer:)', '', text)

        # Handle negative numbers and decimals
        numbers = re.findall(r'-?\d+\.?\d*', text)

        if numbers:
            # Return the last number found (usually the final answer)
            return float(numbers[-1])
        return None

    def normalize_text(text):
        """Normalize text for string comparison"""
        text = str(text).lower().strip()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    # Extract final answers
    pred_final = extract_final_answer(predicted_answer)
    true_final = extract_final_answer(ground_truth_answer)

    # Try numerical comparison first (for math problems)
    pred_num = extract_number(pred_final)
    true_num = extract_number(true_final)

    if pred_num is not None and true_num is not None:
        # Allow small numerical tolerance
        return abs(pred_num - true_num) < 1e-4

    # Fall back to string matching
    pred_normalized = normalize_text(pred_final)
    true_normalized = normalize_text(true_final)

    # Exact match
    if pred_normalized == true_normalized:
        return True

    # Check if one contains the other (handles different formatting)
    # But be stricter - require it to be a substantial portion
    if len(true_normalized) > 0:
        if true_normalized in pred_normalized:
            # Make sure it's not just a substring of a larger number
            # e.g., "5" should not match "15"
            pred_num_check = extract_number(pred_normalized)
            true_num_check = extract_number(true_normalized)
            if pred_num_check is not None and true_num_check is not None:
                return abs(pred_num_check - true_num_check) < 1e-4
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


def train_grpo(model, dataset, tokenizer, learning_rate=2e-5, batch_size=32, max_samples=500):
    """
    Train model using Group Relative Policy Optimization (GRPO).

    FIXES APPLIED:
    - Proper reward function that accesses ground truth from prompts
    - Configurable batch_size
    - max_samples parameter
    """
    print(f"\n{'='*70}")
    print(f"STARTING GRPO (RL) TRAINING")
    print(f"{'='*70}")
    print(f"Hyperparameters:")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Max Samples: {max_samples}")
    print(f"{'='*70}\n")

    model.gradient_checkpointing_enable()

    # Format dataset for GRPO - include answer in a way the reward function can access
    def format_for_grpo(examples):
        prompts = []
        # Store answers in a format we can parse from the prompt
        for i in range(len(examples['0'])):
            question = examples['0'][i]['value']
            try:
                answer = examples['1'][i]['ground_truth']['value']
            except (KeyError, TypeError):
                answer = str(examples['1'][i])

            # Include answer as metadata in prompt (will be parsed by reward function)
            # Use a delimiter that won't appear in math problems
            prompt = f"Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n[GROUND_TRUTH]{answer}[/GROUND_TRUTH]\nAnswer:"
            prompts.append(prompt)

        return {'prompt': prompts}

    print("Formatting dataset for GRPO training...")
    formatted_dataset = dataset.map(format_for_grpo, batched=True, remove_columns=dataset.column_names)
    formatted_dataset = formatted_dataset.select(range(min(max_samples, len(formatted_dataset))))
    print(f"Using {len(formatted_dataset)} examples for GRPO training")

    gradient_accumulation_steps = 4
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")

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
        kl_coef=0.0,
        gradient_checkpointing=True,
    )

    def reward_fn(completions, prompts, **kwargs):
        """
        Reward function that extracts ground truth from the prompt.

        Args:
            completions: List of model completions
            prompts: List of prompts (contains ground truth)
        """
        rewards = []

        for i, completion in enumerate(completions):
            # Extract ground truth from prompt
            prompt = prompts[i] if i < len(prompts) else ""

            # Parse ground truth from prompt
            gt_match = re.search(r'\[GROUND_TRUTH\](.*?)\[/GROUND_TRUTH\]', prompt)
            if gt_match:
                ground_truth = gt_match.group(1)
            else:
                ground_truth = ""
                logger.warning(f"Could not extract ground truth from prompt {i}")

            # Binary outcome supervision (paper's approach)
            if check_answer_correctness(completion, ground_truth):
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        return rewards

    print("Initializing GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,  # Note: single function, not list
    )

    # Remove ground truth markers from prompts before generation
    # This requires a custom data collator or preprocessing
    original_collator = trainer.data_collator

    def clean_prompt_collator(features):
        # Remove ground truth markers from prompts before they're used for generation
        for feature in features:
            if 'prompt' in feature:
                feature['prompt'] = re.sub(
                    r'\[GROUND_TRUTH\].*?\[/GROUND_TRUTH\]\n',
                    '',
                    feature['prompt']
                )
        return original_collator(features)

    trainer.data_collator = clean_prompt_collator

    print(f"\n{'='*70}")
    print(f"BEGINNING GRPO TRAINING LOOP")
    print(f"{'='*70}\n")

    trainer.train()

    print(f"\n{'='*70}")
    print(f"GRPO TRAINING COMPLETE")
    print(f"{'='*70}\n")

    NT = evaluate_new_task(model=model, tokenizer=tokenizer, dataset=dataset)

    return model, trainer, NT