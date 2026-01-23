import os
import gc
import json
import torch
import logging
from transformers import TrainingArguments, Trainer
from data.dataset_utils import UnifiedDatasetInterface
# if running test.py
# from src.data.dataset_utils import UnifiedDatasetInterface
from trl import SFTTrainer
from trl import DataCollatorForCompletionOnlyLM


logger = logging.getLogger(__name__)

def train_sft_baseline(model, 
                       tokenizer, 
                       dataset, 
                       learning_rate, 
                       batch_size, 
                       epochs, 
                       max_samples=3000, 
                       eval_dataset = None):
    """
    Baseline (π₀) SFT training for RL's Razor replication.

    Key properties:
    ------------------------------------------------------
        Loss ONLY on completion (via DataCollatorForCompletionOnlyLM)
        Dataset formatting normalized via UnifiedDatasetInterface
        No weight decay (paper config)
        AdamW + warmup = 50 steps (paper config)
        bfloat16 precision (paper config)
        Gradient checkpointing enabled for memory efficiency
        Evaluation on NT after SFT (paper behavior)
        Consistent prompting format for forgetting eval
    """
    logger.info("=" * 70)
    logger.info("INITIALIZING PAPER-GRADE SFT TRAINING (π₀)")
    logger.info("=" * 70)

    # Importing Config values are kept

    logger.info(f"Current Learning Rate : {learning_rate}, Batch Size : {batch_size}, Epochs : {epochs}, Max Samples : {max_samples}")

    model.gradient_checkpointing_enable()
    tokenizer.model_max_length = 4096
    tokenizer.truncation_side = 'left'

    # Dataset Normalization
    logger.info("Formatting dataset via UnifiedDatasetInterface...")
    dataset = UnifiedDatasetInterface.normalize_dataset(dataset)
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    logger.info(f"Training samples: {len(dataset)}")

    # Group level batching
    gradient_accumulation_steps = 4
    effective_bs = batch_size * gradient_accumulation_steps
    logger.info(f"Effective batch size: {effective_bs}")

    training_args = TrainingArguments(
        output_dir=f"./results_sft/lr{learning_rate}_bs{effective_bs}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=50,
        bf16=True,
        optim="adamw_torch",   
        weight_decay=0,        
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        report_to="none",
    )

    logger.info("Detecting prompt format for completion-only masking...")
    fmt = UnifiedDatasetInterface.detect_format(dataset[0])
    response_template = "Answer:" if fmt != 'alpaca' else "Response:"
    logger.info(f"Using response_template='{response_template}'")

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    logger.info("Creating SFTTrainer (TRL) ...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        formatting_func=lambda e: e["text"], 
    )


    logger.info("=" * 70)
    logger.info("START TRAINING π₀ (SFT)")
    logger.info("=" * 70)

    trainer.train()

    logger.info("=" * 70)
    logger.info("FINISHED π₀ (SFT)")
    logger.info("=" * 70)


    NT = None
    if eval_dataset is not None:
        logger.info("Evaluating π₀ on NEW TASK dataset (NT)")
        from evaluation.evaluation import evaluate_new_task
        NT = evaluate_new_task(
            model=trainer.model,
            tokenizer=tokenizer,
            dataset=eval_dataset,
            num_samples=100
        )
        logger.info(f"π₀ NT score: {NT:.3f}")

    # cleanup
    gc.collect()
    torch.cuda.empty_cache()

    return trainer.model, NT