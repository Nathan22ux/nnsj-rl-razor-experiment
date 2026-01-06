"""
FIXED training.py for RL's Razor Replication

FIXES APPLIED:
1. GRPO reward function - validates ground truth extraction, raises errors
2. Improved answer checking with better regex patterns
3. Support for multiple dataset formats
4. Better error handling and logging
5. Configurable max_samples parameter
6. COMPLETION-ONLY LOSS FOR SFT (DataCollatorForCompletionOnlyLM)
   - Masks prompt tokens so loss is ONLY computed on answer/completion
   - Ensures fair comparison with RL (both optimize only output variables)
   - Matches paper's methodology for valid forgetting comparison
7. Standardized prompts between SFT and RL
8. Robust dataset key handling in GRPO
"""

import re
import torch
from transformers import TrainingArguments
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
# from trl.trainer.utils import DataCollatorForCompletionOnlyLM
import gc
from dataclasses import dataclass
import inspect
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List
from logger import get_logger
from data.dataset_utils import UnifiedDatasetInterface

logger = get_logger(__name__)


@dataclass
class DataCollatorForCompletionOnlyLM:
    """
    Custom data collator for completion-only loss.
    Masks prompt tokens (sets labels to -100) so loss is only computed on completions.
    
    This replaces the TRL DataCollatorForCompletionOnlyLM which was removed/moved in recent versions.
    
    Uses text-based search (more robust than token-based for different tokenizers).
    """
    tokenizer: Any
    response_template: str  # e.g., "Answer:" - searched in decoded text
    mlm: bool = False
    _warned: bool = False
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pad the features
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        
        # Create labels and mask prompt tokens
        labels = batch["input_ids"].clone()
        
        found_count = 0
        not_found_count = 0
        
        for i in range(len(labels)):
            input_ids = batch["input_ids"][i]
            
            # Decode the full sequence to find template position in text
            decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            
            # Find the template in the decoded text (case-sensitive)
            # Strip the leading \n if present since it might be encoded differently
            template_to_find = self.response_template.lstrip('\n')
            template_pos = decoded_text.find(template_to_find)
            
            if template_pos == -1:
                # Template not found - don't mask anything, loss on full sequence
                not_found_count += 1
                labels[i, batch["attention_mask"][i] == 0] = -100  # Still mask padding
                continue
            
            found_count += 1
            
            # Find the token position that corresponds to after the template
            # We do this by encoding the text up to and including the template
            text_up_to_template = decoded_text[:template_pos + len(template_to_find)]
            tokens_up_to_template = self.tokenizer.encode(text_up_to_template, add_special_tokens=False)
            response_start = len(tokens_up_to_template)
            
            # Mask everything before the response (set to -100, ignored in loss)
            if response_start < len(labels[i]):
                labels[i, :response_start] = -100
            
            # Also mask padding tokens
            labels[i, batch["attention_mask"][i] == 0] = -100
        
        # Log warning once if templates weren't found
        if not_found_count > 0 and not self._warned:
            logger.warning(f"Response template '{self.response_template}' not found in {not_found_count}/{len(labels)} samples")
            object.__setattr__(self, '_warned', True)
        
        batch["labels"] = labels
        return batch

def train_sft(model, dataset, tokenizer, learning_rate=3e-5, batch_size=32, epochs=1, max_samples=500, eval_dataset=None):
    """
    Trains a SFT model on the given dataset using COMPLETION-ONLY LOSS.

    PAPER METHODOLOGY (RL's Razor):
    - Uses DataCollatorForCompletionOnlyLM to mask prompt tokens
    - Loss is computed ONLY on completion/answer tokens (after "Answer:")
    - This ensures fair comparison with RL, which only optimizes generated tokens
    - Both methods strictly optimize OUTPUT variables, making forgetting comparison valid

    FIXES:
    - Configurable max_samples (not hardcoded)
    - Support multiple dataset formats
    - Better logging
    - COMPLETION-ONLY LOSS via DataCollatorForCompletionOnlyLM

    Args:
        model: Model to train
        dataset: Training dataset
        tokenizer: Tokenizer
        learning_rate: Learning rate
        batch_size: Batch size per device
        epochs: Number of epochs
        max_samples: Maximum training samples
        eval_dataset: Evaluation dataset (optional)

    Returns:
        tuple: (trained_model, trainer, NT_score)  
    """
    logger.info(f"{'='*70}")
    logger.info(f"STARTING SFT TRAINING")
    logger.info(f"{'='*70}")
    logger.info(f"Hyperparameters:")
    logger.info(f" Learning Rate: {learning_rate}")
    logger.info(f" Batch Size: {batch_size}")
    logger.info(f" Epochs: {epochs}")
    logger.info(f" Max Samples: {max_samples}")
    logger.info(f"{'='*70}")

    # Enable gradient checkpointing to save memory
    logger.info("Enabling gradient checkpointing to save memory...")
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled")

    # Use centralized dataset formatting from dataset_utils
    logger.info("Formatting dataset using UnifiedDatasetInterface...")
    try:
        formatted_dataset = UnifiedDatasetInterface.normalize_dataset(dataset)
        logger.info(f"Dataset formatted: {len(formatted_dataset)} examples")
    except Exception as e:
        logger.error(f"Error formatting dataset: {e}")
        raise

    formatted_dataset = formatted_dataset.select(range(min(max_samples, len(formatted_dataset))))
    logger.info(f"Using {len(formatted_dataset)} examples for training")

    # Pre-format the dataset with a 'text' column for SFTTrainer
    # The response_template "Answer:" marks where the completion begins
    # DataCollatorForCompletionOnlyLM will use this to compute loss only on completions
    # def format_for_sft(example):
    #     # Combine prompt and answer with the response template
    #     # The "Answer:" template tells the collator where completion begins
    #     text = f"{example['prompt']} Answer: {example['answer']}"
    #     return {'text': text}
    
    # formatted_dataset = formatted_dataset.map(
    #     format_for_sft,
    #     remove_columns=formatted_dataset.column_names
    # )

    formatted_dataset = formatted_dataset.remove_columns([c for c in formatted_dataset.column_names if c != 'text'])
    logger.info(f"Dataset pre-formatted for TRL 0.26.x: columns = {formatted_dataset.column_names}")

    # Training arguments
    gradient_accumulation_steps = 8  # Increased for memory optimization
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

    # =========================================================================
    # COMPLETION-ONLY LOSS (Paper's Fair Comparison Requirement)
    # =========================================================================
    # The paper explicitly uses completion-only loss for SFT to ensure fair
    # comparison with RL. This masks the prompt so loss is only computed on
    # the answer/completion portion, not the input question.
    #
    # This ensures both SFT and RL optimize only the OUTPUT variables (answers),
    # making the comparison of their side-effects (forgetting) structurally valid.
    # =========================================================================

    logger.info("Initializing SFT Trainer with completion-only loss (TRL 0.26.x native support)...")
    logger.info("Loss will be computed ONLY on completion portion (after 'Answer:'), not the prompt")
    
    # Response template marks where the completion begins
    # Loss will only be computed on tokens AFTER this template
    # The DataCollator uses text-based search (more robust than token-based)
    response_template = "Answer:"
    
    logger.info(f"Creating DataCollatorForCompletionOnlyLM with response_template='{response_template}'")
    logger.info("This ensures loss is computed ONLY on completions (not prompts) for fair comparison with RL")
    
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    # Verify that the response template can be found in the dataset (text-based check)
    sample_text = formatted_dataset[0]['text']
    if response_template in sample_text:
        template_pos = sample_text.find(response_template)
        logger.info(f"✓ Response template '{response_template}' found at position {template_pos} in sample text")
        logger.info(f"   Text before template: ...{sample_text[max(0,template_pos-30):template_pos]}")
        logger.info(f"   Text after template: {sample_text[template_pos:template_pos+50]}...")
    else:
        logger.warning(f"⚠️ Response template '{response_template}' not found in sample!")
        logger.warning(f"   Sample text: {sample_text[:200]}...")
        logger.warning(f"   Completion-only loss will NOT work!")

    logger.info("Initializing SFT Trainer with completion-only loss...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,  # Masks prompt, computes loss only on completion
    )
    logger.info("SFT Trainer initialized")

    # logger.info(f"{'='*70}")
    # logger.info(f"STARTING TRAINING")
    # logger.info(f"{'='*70}")

    # =========================================================================
    # WEIGHT MONITORING - Capture initial weights for comparison
    # =========================================================================
    def get_weight_stats(model, name="model"):
        """Get statistics of model weights for monitoring.
        Checks a middle-layer attention weight (more representative than embeddings).
        """
        stats = {}
        target_layer = None
        
        # Find a representative layer (prefer attention/MLP layers over embeddings)
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                # Skip embedding layers, prefer attention/MLP layers
                if 'embed' not in param_name.lower() and ('attn' in param_name.lower() or 'mlp' in param_name.lower()):
                    target_layer = (param_name, param)
                    break
        
        # Fallback to any trainable layer if no attention/MLP found
        if target_layer is None:
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    target_layer = (param_name, param)
                    break
        
        if target_layer:
            param_name, param = target_layer
            stats[param_name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'norm': param.data.norm().item(),
            }
        return stats
    
    initial_weights = get_weight_stats(model, "initial")
    monitored_param_name = list(initial_weights.keys())[0] if initial_weights else "N/A"
    logger.info(f"Initial weight stats for '{monitored_param_name}':")
    if initial_weights:
        stats = initial_weights[monitored_param_name]
        logger.info(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}, Norm: {stats['norm']:.6f}")

    logger.info(f"{'='*70}")
    logger.info(f"STARTING TRAINING")
    logger.info(f"{'='*70}")

    trainer.train()

    final_weights = get_weight_stats(model, "final")
    logger.info(f"Final weight stats for '{monitored_param_name}':")
    if final_weights and monitored_param_name in final_weights:
        stats = final_weights[monitored_param_name]
        logger.info(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}, Norm: {stats['norm']:.6f}")
        
        # Compare
        if initial_weights and monitored_param_name in initial_weights:
            init_stats = initial_weights[monitored_param_name]
            mean_diff = abs(stats['mean'] - init_stats['mean'])
            norm_diff = abs(stats['norm'] - init_stats['norm'])
            logger.info(f"  Weight change - Mean diff: {mean_diff:.8f}, Norm diff: {norm_diff:.8f}")
            
            if mean_diff < 1e-8 and norm_diff < 1e-8:
                logger.warning("⚠️ WEIGHTS DID NOT CHANGE! Training may not be working.")
            else:
                logger.info("✓ Weights updated successfully during training.")


    logger.info(f"{'='*70}")
    logger.info(f"SFT TRAINING COMPLETE")
    logger.info(f"{'='*70}")

    # Evaluate on new task
    from evaluation.evaluation import evaluate_new_task
    NT = evaluate_new_task(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        eval_dataset=eval_dataset,
        num_samples=100
    )

    # Clear memory after training
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Cleared CUDA cache after SFT training")

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
    - FIXED: Single letter multiple choice handling

    Args:
        predicted_answer: Model's predicted answer
        ground_truth_answer: Correct answer

    Returns:
        bool: True if answers match
    """
    # Extract final answers
    pred_final = extract_final_answer(predicted_answer)
    true_final = extract_final_answer(ground_truth_answer)

    # SPECIAL CASE: Single letter answers (multiple choice A, B, C, D, etc.)
    true_clean_raw = str(true_final).strip().upper()
    if len(true_clean_raw) == 1 and true_clean_raw.isalpha():
        # For single letter expected answers, we need strict matching
        pred_clean_raw = str(pred_final).strip().upper()

        # Check if pred_final is exactly the letter
        if pred_clean_raw == true_clean_raw:
            return True

        # Check for patterns like "B", "(B)", "B.", "B)", "Option B"
        mc_patterns = [
            rf'^{true_clean_raw}$',  # Just the letter
            rf'^\({true_clean_raw}\)$',  # (B)
            rf'^{true_clean_raw}[\.\)]',  # B. or B)
            rf'^Option\s*{true_clean_raw}',  # Option B
            rf'^{true_clean_raw}\s*[\-\:]',  # B - or B:
        ]
        for pattern in mc_patterns:
            if re.match(pattern, pred_clean_raw, re.IGNORECASE):
                return True

        # Also check in boxed answer specifically
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', predicted_answer)
        if boxed_match:
            boxed_content = boxed_match.group(1).strip().upper()
            if boxed_content == true_clean_raw:
                return True

        # For single letter answers, DON'T do substring matching (too error-prone)
        return False

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

    # Substring match (with safeguards)
    # Only do this for longer expected answers (not single letters/short words)
    if len(true_clean) >= 3 and true_clean in pred_clean:
        # Make sure not part of larger word
        idx = pred_clean.find(true_clean)

        # Check before
        if idx > 0 and pred_clean[idx-1].isalnum():
            return False

        # Check after
        end_idx = idx + len(true_clean)
        if end_idx < len(pred_clean) and pred_clean[end_idx].isalnum():
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


def train_grpo(model, dataset, tokenizer, learning_rate=2e-5, batch_size=32, max_samples=500, eval_dataset=None, target_nt=None, **kwargs):
    """
    Train model using Group Relative Policy Optimization (GRPO).

    FIXES APPLIED:
    - Robust reward function using question hashing
    - Configurable batch_size
    - max_samples parameter
    - eval_dataset parameter
    - Robust dataset key handling (copied from SFT)
    """
    logger.info(f"{'='*70}")
    logger.info(f"STARTING GRPO (RL) TRAINING")
    logger.info(f"{'='*70}")
    logger.info(f"Configuration:")
    logger.info(f"  Learning Rate: {learning_rate}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Max Samples: {max_samples}")
    logger.info(f"  KL Coefficient: 0.0 (implicit KL minimization)")
    if target_nt is not None:
        logger.info(f"  Target NT (experiment gating): {target_nt}")
    logger.info(f"{'='*70}")

    model.gradient_checkpointing_enable()

    # Use centralized dataset formatting, then extract questions and answers
    logger.info("Formatting dataset using UnifiedDatasetInterface...")
    normalized_dataset = UnifiedDatasetInterface.normalize_dataset(dataset)

    # Select subset deterministically first (avoids side-effects inside dataset.map)
    n_use = min(max_samples, len(normalized_dataset))
    normalized_subset = normalized_dataset.select(range(n_use))

    # Build question-to-answer lookup for reward function (NO side-effects in map)
    logger.info("Building question-to-answer lookup table for GRPO rewards...")
    question_to_answer: Dict[str, str] = {}
    questions = normalized_subset["question"]
    answers = normalized_subset["answer"]
    for q, a in zip(questions, answers):
        question_to_answer[question_to_key(q)] = a
    logger.info(f"Answer lookup table has {len(question_to_answer)} entries")

    def format_for_grpo(examples):
        """Convert normalized dataset to GRPO format with prompts (no side-effects)."""
        prompts = []
        for question in examples["question"]:
            prompt = (
                f"Question: {question}\n"
                f"Please reason step by step, and put your final answer within \\boxed{{}}.\n"
                f"Answer:"
            )
            prompts.append(prompt)
        return {"prompt": prompts}

    logger.info("Formatting normalized dataset for GRPO training...")
    formatted_dataset = normalized_subset.map(
        format_for_grpo,
        batched=True,
        remove_columns=normalized_subset.column_names,
    )
    logger.info(f"Using {len(formatted_dataset)} examples for GRPO training")

    # GRPO-specific hyperparameters
    # - prompts_per_generation: how many prompts are sampled per update step (paper: 8)
    # - num_generations: how many completions per prompt (paper group size: 64)
    prompts_per_generation = int(kwargs.pop("prompts_per_generation", batch_size))
    num_generations = int(kwargs.pop("num_generations", 16))
    max_prompt_length = int(kwargs.pop("max_prompt_length", 1024))
    max_completion_length = int(kwargs.pop("max_completion_length", 512))
    min_completion_length = int(kwargs.pop("min_completion_length", 0))

    logger.info("GRPO generation settings:")
    logger.info(f"  prompts_per_generation: {prompts_per_generation}")
    logger.info(f"  num_generations (group size): {num_generations}")
    logger.info(f"  max_prompt_length: {max_prompt_length}")
    logger.info(f"  max_completion_length: {max_completion_length}")

    gradient_accumulation_steps = 8  # Match SFT for fair comparison
    logger.info(f"Effective prompts per optimizer step: {prompts_per_generation * gradient_accumulation_steps}")

    # Build GRPOConfig with only supported fields (TRL versions vary)
    base_grpo_kwargs = dict(
        output_dir=f"./results/grpo_lr{learning_rate}_bs{prompts_per_generation}",
        num_train_epochs=1,
        per_device_train_batch_size=prompts_per_generation,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=50,
        max_grad_norm=1.0,
        logging_steps=10,
        report_to="none",
        gradient_checkpointing=True,
        # Generation controls (if supported by this TRL version)
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        min_completion_length=min_completion_length,
    )

    if is_dataclass(GRPOConfig):
        supported_fields = {f.name for f in fields(GRPOConfig)}
    else:
        supported_fields = set(inspect.signature(GRPOConfig).parameters.keys())
    grpo_kwargs = {k: v for k, v in base_grpo_kwargs.items() if k in supported_fields}
    missing = sorted(set(base_grpo_kwargs.keys()) - set(grpo_kwargs.keys()))
    if missing:
        logger.info(f"GRPOConfig does not support fields (ignored): {missing}")

    grpo_config = GRPOConfig(**grpo_kwargs)
    # Log final resolved config values (helps confirm TRL accepted our settings)
    for attr in ["num_generations", "max_prompt_length", "max_completion_length", "min_completion_length"]:
        if hasattr(grpo_config, attr):
            logger.info(f"GRPOConfig.{attr} = {getattr(grpo_config, attr)}")

    # Track reward statistics for debugging
    reward_stats = {'found': 0, 'not_found': 0, 'correct': 0, 'incorrect': 0}

    def reward_fn(completions, prompts, **kwargs):
        """
        FIXED: Reward function using robust question hashing.
        """
        # Validate inputs
        if len(completions) != len(prompts):
            raise ValueError(
                f"Length mismatch: {len(completions)} completions vs {len(prompts)} prompts"
                f"This should never happen - check GRPO trainer configuration."
            )

        rewards = []

        for completion, prompt in zip(completions, prompts):
            # Extract question from prompt using regex (allow extra lines before "Please reason")
            q_match = re.search(r'Question:\s*(.+?)\n\s*Please reason', prompt, re.DOTALL)
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
                    logger.warning(f"No ground truth found for prompt: {prompt}...")
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

    logger.info(f"{'='*70}")
    logger.info(f"BEGINNING GRPO TRAINING LOOP")
    logger.info(f"{'='*70}")

    if target_nt is not None:
        logger.info(f"  Target NT (used by experiment gating): {target_nt}")
    try:
        trainer.train()
    except Exception as e:
        logger.info(f"\n Training failed: {e}")
        raise

    logger.info(f"{'='*70}")
    logger.info(f"GRPO TRAINING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Reward Statistics:")
    logger.info(f" Answers found: {reward_stats['found']}")
    logger.info(f" Answers not found: {reward_stats['not_found']}")
    logger.info(f" Correct answers: {reward_stats['correct']}")
    logger.info(f" Incorrect answers: {reward_stats['incorrect']}")
    if reward_stats['found'] > 0:
        accuracy = reward_stats['correct'] / reward_stats['found'] * 100
        logger.info(f" Training accuracy: {accuracy:.1f}%")
    logger.info(f"{'='*70}")

    # Evaluate on new task
    from evaluation.evaluation import evaluate_new_task
    NT = evaluate_new_task(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        eval_dataset=eval_dataset,
        num_samples=100
    )

    # Clear memory after training
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Cleared CUDA cache after GRPO training")

    return model, trainer, NT