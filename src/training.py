"""
FIXED training.py for RL's Razor Replication

FIXES APPLIED:
1. GRPO reward function - validates ground truth extraction, raises errors
2. Improved answer checking with better regex patterns
3. Support for multiple dataset formats
4. Better error handling and logging
5. Configurable max_samples parameter
"""

from transformers import TrainingArguments
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
import logging
import re
import torch

logger = logging.getLogger(__name__)


def train_sft(model, dataset, tokenizer, learning_rate=3e-5, batch_size=32, epochs=1, max_samples=500, eval_samples=100):
    """
    Train model using Supervised Fine-Tuning (SFT).
    
    FIXES:
    - Configurable max_samples (not hardcoded)
    - Support multiple dataset formats
    - Better logging
    
    Args:
        model: Model to train
        dataset: Training dataset
        tokenizer: Tokenizer
        learning_rate: Learning rate
        batch_size: Batch size per device
        epochs: Number of epochs
        max_samples: Maximum training samples
    
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
    
    print("Formatting dataset...")
    try:
        formatted_dataset = dataset.map(format_dataset, batched=True, remove_columns=dataset.column_names)
        print(f"Dataset formatted: {len(formatted_dataset)} examples\n")
    except Exception as e:
        print(f"Error formatting dataset: {e}")
        raise
    
    # Select subset
    formatted_dataset = formatted_dataset.select(range(min(max_samples, len(formatted_dataset))))
    print(f"Using {len(formatted_dataset)} examples for training\n")
    
    # Training arguments
    gradient_accumulation_steps = 4
    effective_batch_size = batch_size * gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch_size}\n")
    
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
    
    print("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    print("SFT Trainer initialized\n")
    
    print(f"{'='*70}")
    print(f"STARTING TRAINING")
    print(f"{'='*70}\n")
    
    trainer.train()
    
    print(f"\n{'='*70}")
    print(f"SFT TRAINING COMPLETE")
    print(f"{'='*70}\n")
    
    # Evaluate on new task
    from evaluation import evaluate_new_task
    NT = evaluate_new_task(model=model, tokenizer=tokenizer, dataset=dataset, num_samples=eval_samples)
    
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
    
    # Substring match (with safeguards)
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


def train_grpo(model, dataset, tokenizer, learning_rate=2e-5, batch_size=32, max_samples=500, eval_samples=100):
    """
    Train model using Group Relative Policy Optimization (GRPO).
    
    FIXES:
    - Validates ground truth extraction
    - Raises errors on failures (not silent)
    - Better reward function
    - Configurable batch_size and max_samples
    
    Args:
        model: Model to train
        dataset: Training dataset
        tokenizer: Tokenizer
        learning_rate: Learning rate
        batch_size: Batch size
        max_samples: Maximum training samples
    
    Returns:
        tuple: (trained_model, trainer, NT_score)
    """
    print(f"\n{'='*70}")
    print(f"STARTING GRPO (RL) TRAINING")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Max Samples: {max_samples}")
    print(f"   KL Coefficient: 0.0 (implicit KL minimization)")
    print(f"{'='*70}\n")
    
    model.gradient_checkpointing_enable()
    
    # Format dataset for GRPO
    def format_for_grpo(examples):
        """Format dataset with embedded ground truth for reward function"""
        prompts = []
        
        # Determine format
        if '0' in examples and '1' in examples:
            # Open-Reasoner format
            for i in range(len(examples['0'])):
                question = examples['0'][i]['value']
                try:
                    answer = examples['1'][i]['ground_truth']['value']
                except (KeyError, TypeError):
                    answer = str(examples['1'][i])
                
                # Embed ground truth in special markers
                prompt = (
                    f"Question: {question}\n"
                    f"Please reason step by step, and put your final answer within \\boxed{{}}.\n"
                    f"[GROUND_TRUTH]{answer}[/GROUND_TRUTH]\n"
                    f"Answer:"
                )
                prompts.append(prompt)
        
        elif 'question' in examples and 'answer' in examples:
            # GSM8K format
            for i in range(len(examples['question'])):
                question = examples['question'][i]
                answer = examples['answer'][i]
                
                prompt = (
                    f"Question: {question}\n"
                    f"[GROUND_TRUTH]{answer}[/GROUND_TRUTH]\n"
                    f"Answer:"
                )
                prompts.append(prompt)
        
        elif 'instruction' in examples and 'output' in examples:
            # Alpaca format
            for i in range(len(examples['instruction'])):
                instruction = examples['instruction'][i]
                input_text = examples.get('input', [''] * len(examples['instruction']))[i]
                output = examples['output'][i]
                
                if input_text:
                    prompt = (
                        f"Instruction: {instruction}\n"
                        f"Input: {input_text}\n"
                        f"[GROUND_TRUTH]{output}[/GROUND_TRUTH]\n"
                        f"Response:"
                    )
                else:
                    prompt = (
                        f"Instruction: {instruction}\n"
                        f"[GROUND_TRUTH]{output}[/GROUND_TRUTH]\n"
                        f"Response:"
                    )
                prompts.append(prompt)
        
        else:
            raise ValueError(f"Unknown dataset format: {examples.keys()}")
        
        return {'prompt': prompts}
    
    print("Formatting dataset for GRPO...")
    try:
        formatted_dataset = dataset.map(format_for_grpo, batched=True, remove_columns=dataset.column_names)
        print(f" Dataset formatted: {len(formatted_dataset)} examples\n")
    except Exception as e:
        print(f" Error formatting dataset: {e}")
        raise
    
    formatted_dataset = formatted_dataset.select(range(min(max_samples, len(formatted_dataset))))
    print(f"→ Using {len(formatted_dataset)} examples for training\n")
    
    gradient_accumulation_steps = 4
    print(f" Effective batch size: {batch_size * gradient_accumulation_steps}\n")
    
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
        gradient_checkpointing=True,
    )
    
    def reward_fn(completions, prompts, **kwargs):
        """
        FIXED reward function with proper validation.
        
        Binary reward based on answer correctness.
        Validates ground truth extraction and raises errors on failures.
        
        Args:
            completions: List of model completions
            prompts: List of prompts (contain embedded ground truth)
        
        Returns:
            List of rewards (0.0 or 1.0)
        """
        # Validate inputs
        if len(completions) != len(prompts):
            raise ValueError(
                f"Length mismatch: {len(completions)} completions vs {len(prompts)} prompts\n"
                f"This should never happen - check GRPO trainer configuration."
            )
        
        rewards = []
        failed_extractions = []
        reward_stats = {'correct': 0, 'incorrect': 0}
        
        for i, (completion, prompt) in enumerate(zip(completions, prompts)):
            # Extract ground truth from prompt
            gt_match = re.search(r'\[GROUND_TRUTH\](.*?)\[/GROUND_TRUTH\]', prompt)
            
            if not gt_match:
                failed_extractions.append(i)
                # Conservative: assign 0 reward on extraction failure
                rewards.append(0.0)
                reward_stats['incorrect'] += 1
                continue
            
            ground_truth = gt_match.group(1).strip()
            
            # Check answer correctness
            if check_answer_correctness(completion, ground_truth):
                rewards.append(1.0)
                reward_stats['correct'] += 1
            else:
                rewards.append(0.0)
                reward_stats['incorrect'] += 1
        
        # Alert if too many extraction failures
        failure_rate = len(failed_extractions) / len(prompts)
        if failure_rate > 0.1:  # >10% failures
            raise ValueError(
                f"Ground truth extraction failed for {len(failed_extractions)}/{len(prompts)} prompts ({failure_rate*100:.1f}%)!\n"
                f"Failed indices: {failed_extractions[:10]}...\n"
                f"Check prompt formatting. Example prompt:\n{prompts[0][:200]}..."
            )
        
        # Log reward statistics occasionally
        if np.random.random() < 0.1:  # 10% of batches
            success_rate = reward_stats['correct'] / len(rewards) if rewards else 0
            logger.info(f"Batch reward stats: {reward_stats['correct']}/{len(rewards)} correct ({success_rate*100:.1f}%)")
        
        return rewards
    
    print("Initializing GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )
    print(" Trainer initialized\n")
    
    # Clean prompts before generation (remove ground truth markers)
    original_collator = trainer.data_collator
    
    def clean_prompt_collator(features):
        """Remove ground truth markers before generation"""
        for feature in features:
            if 'prompt' in feature:
                feature['prompt'] = re.sub(
                    r'\[GROUND_TRUTH\].*?\[/GROUND_TRUTH\]\n',
                    '',
                    feature['prompt']
                )
        return original_collator(features)
    
    trainer.data_collator = clean_prompt_collator
    
    print(f"{'='*70}")
    print(f"STARTING GRPO TRAINING")
    print(f"{'='*70}\n")
    
    try:
        trainer.train()
    except Exception as e:
        print(f"\n Training failed: {e}")
        raise
    
    print(f"\n{'='*70}")
    print(f"GRPO TRAINING COMPLETE")
    print(f"{'='*70}\n")
    
    # Evaluate on new task
    from evaluation import evaluate_new_task
    NT = evaluate_new_task(model=model, tokenizer=tokenizer, dataset=dataset, num_samples=eval_samples)
    
    return model, trainer, NT


# Import numpy for reward logging
import numpy as np