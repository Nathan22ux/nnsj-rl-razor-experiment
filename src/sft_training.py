import torch
from transformers import TrainingArguments
from trl import SFTTrainer

SFT_CONFIG = {
    'learning_rates': [1e-5, 3e-5, 5e-5, 7e-5, 9e-5],
    'batch_sizes': [16, 32, 64, 128],
    'epochs': [1, 2],
    'lr_scheduler': ['constant_with_warmup', 'cosine_with_warmup'],
    'warmup_steps': 50,
    'optimizer': 'adamw',
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'bf16': True,
}

def train_sft(model, dataset, tokenizer, learning_rate=5e-5, batch_size=32, 
              num_epochs=1, output_dir="./sft_output", scheduler_type="cosine"):
    """
    Train model using Supervised Fine-Tuning (SFT).
    """
    
    # Format function for the dataset
    def format_prompt(example):
        if 'text' in example:
            return example
        elif 'question' in example and 'answer' in example:
            return {"text": f"Question: {example['question']}\n\nAnswer: {example['answer']}"}
        elif 'problem' in example and 'solution' in example:
            return {"text": f"Problem: {example['problem']}\n\nSolution: {example['solution']}"}
        else:
            return {"text": str(example)}
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        warmup_steps=50,
        learning_rate=learning_rate,
        lr_scheduler_type=scheduler_type,
        max_grad_norm=1.0,
        weight_decay=0,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        optim="adamw_torch",
        seed=42,
    )
    
    # Create SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512,
        formatting_func=format_prompt,
    )
    
    print(f"Starting SFT training with lr={learning_rate}, batch_size={batch_size}")
    
    # Train
    trainer.train()
    
    print("SFT training completed")
    
    return model, trainer
