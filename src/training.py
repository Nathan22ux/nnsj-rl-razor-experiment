from transformers import TrainingArguments
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
from evaluation import evaluate_new_task

def train_sft(model, dataset, tokenizer, learning_rate=3e-5, batch_size=32, epochs=1):
    """
    Train model using Supervised Fine-Tuning (SFT).
    
    Args:
        model: The model to train
        dataset: Training dataset
        tokenizer: Tokenizer for the model
        learning_rate: Learning rate for training
        batch_size: Batch size per device
        epochs: Number of training epochs
        
    Returns:
        tuple: (trained_model, trainer)
    """
    print(f"\n{'='*70}")
    print(f"STARTING SFT TRAINING")
    print(f"{'='*70}")
    print(f"Hyperparameters:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
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
            text = f"Question: {question}\nAnswer: {answer}"
            texts.append(text)
        
        return {'text': texts}
    
    # Apply formatting
    print("\n Formatting dataset for SFT training...")
    formatted_dataset = dataset.map(format_dataset, batched=True, remove_columns=dataset.column_names)
    print(f"Dataset formatted: {len(formatted_dataset)} total examples")
    
    # Bc og GPU limitations selected 50 examples for small run
    formatted_dataset = formatted_dataset.select(range(min(30, len(formatted_dataset))))
    print(f"Using {len(formatted_dataset)} examples for training (limited for GPU constraints)")
    
    print("\n Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=f"./results/sft_lr{learning_rate}_bs{batch_size}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
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
    
    Args:
        predicted_answer: Model's predicted answer
        ground_truth_answer: Correct answer
        
    Returns:
        bool: True if answers match, False otherwise
    """
    import re
    
    def extract_number(text):
        # Extract the final numerical answer from text and remove common answer prefixes
        text = text.lower()
        text = re.sub(r'(the answer is|therefore|thus|so|final answer:)', '', text)
        
        # Try to find numbers (including decimals and fractions)
        numbers = re.findall(r'-?\d+\.?\d*', text)
        
        if numbers:
            return float(numbers[-1])  # Return last number found
        return None
    
    def normalize_text(text):
        # Normalize text for string comparison
        text = str(text).lower().strip()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    # Try numerical comparison first (for math problems)
    pred_num = extract_number(str(predicted_answer))
    true_num = extract_number(str(ground_truth_answer))
    
    if pred_num is not None and true_num is not None:
        # Allow small numerical tolerance
        return abs(pred_num - true_num) < 1e-4
    
    # Fall back to string matching
    pred_normalized = normalize_text(predicted_answer)
    true_normalized = normalize_text(ground_truth_answer)
    
    # Check if one contains the other (handles different formatting)
    return (pred_normalized in true_normalized or
            true_normalized in pred_normalized or
            pred_normalized == true_normalized)


def train_grpo(model, dataset, tokenizer, learning_rate=2e-5):
    """
    Train model using Group Relative Policy Optimization (GRPO).
    
    Args:
        model: The model to train
        dataset: Training dataset
        tokenizer: Tokenizer for the model
        learning_rate: Learning rate for training
        
    Returns:
        tuple: (trained_model, trainer)
    """
    print(f"\n{'='*70}")
    print(f"STARTING GRPO (RL) TRAINING")
    print(f"{'='*70}")
    print(f"Hyperparameters:")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Batch Size: 64")
    print(f"   Epochs: 1")
    print(f"{'='*70}\n")
    
    # Format dataset for GRPO - this needs a 'prompt' field
    def format_for_grpo(examples):
        prompts = []
        answers = []
        for i in range(len(examples['0'])):
            question = examples['0'][i]['value']
            try:
                answer = examples['1'][i]['ground_truth']['value']
            except (KeyError, TypeError):
                answer = str(examples['1'][i])
            
            prompt = f"Question: {question}\nAnswer:"
            prompts.append(prompt)
            answers.append(answer)
        
        return {'prompt': prompts, 'answer': answers}
    
    # Format dataset
    print("Formatting dataset for GRPO training...")
    formatted_dataset = dataset.map(format_for_grpo, batched=True, remove_columns=dataset.column_names)
    print(f"Dataset formatted: {len(formatted_dataset)} total examples")
    
    # Reduce to 50 examples bc of Colab limits (delete on jupyter)
    formatted_dataset = formatted_dataset.select(range(min(30, len(formatted_dataset))))
    print(f"Using {len(formatted_dataset)} examples for GRPO training (limited for GPU constraints)")
    
    print("\nSetting up GRPO configuration...")
    grpo_config = GRPOConfig(
        output_dir=f"./results/grpo_lr{learning_rate}",
        num_train_epochs=1,
        per_device_train_batch_size=64,
        learning_rate=learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=50,
        max_grad_norm=1.0,
        logging_steps=10,
        report_to="none",
    )
    print("GRPO configuration set")
    
    print("\nDefining reward function...")
    def reward_fn(prompts, completions, completion_ids, **kwargs):
        """
        Reward function with correct signature for GRPOTrainer
        Args:
            prompts: List of prompt strings
            completions: List of completion strings
            completion_ids: List of completion token IDs
        Returns:
            List of reward scores (float)
        """
        rewards = []
        # currently checking whether context is producted or not, but not learning
        for prompt, completion in zip(prompts, completions):
            try:
                # This is a basic reward fn, it checks if completion has content
                if completion and len(completion.strip()) > 0:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except:
                rewards.append(0.0)
        
        return rewards
    print("Reward function defined")
    
    print("\nInitializing GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
    )
    print("GRPO Trainer initialized")
    
    print(f"\n{'='*70}")
    print(f"BEGINNING GRPO TRAINING LOOP (lr={learning_rate})")
    print(f"{'='*70}")
    print("This will take a while... Progress will be shown below:\n")
    
    trainer.train()
    
    print(f"\n{'='*70}")
    print(f"GRPO TRAINING COMPLETE")
    print(f"{'='*70}\n")

    NT = evaluate_new_task(model=model, tokenizer=tokenizer, dataset=dataset)
    
    return model, trainer, NT
