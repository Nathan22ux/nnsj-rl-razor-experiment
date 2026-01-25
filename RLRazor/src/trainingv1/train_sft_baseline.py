import os
import gc
import json
import torch
import logging
from data.dataset_utils import UnifiedDatasetInterface
# if running test.py
# from src.data.dataset_utils import UnifiedDatasetInterface
from trl import SFTTrainer, SFTConfig


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
        Loss ONLY on completion (via SFTConfig completion_only_loss)
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

    # Detect format BEFORE normalization (important for response_template)
    original_format = UnifiedDatasetInterface.detect_format(dataset[0])
    logger.info(f"Detected original format: {original_format}")

    # Dataset Normalization
    logger.info("Formatting dataset via UnifiedDatasetInterface...")
    dataset = UnifiedDatasetInterface.normalize_dataset(dataset)
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    logger.info(f"Training samples: {len(dataset)}")

    # Group level batching
    gradient_accumulation_steps = 4
    effective_bs = batch_size * gradient_accumulation_steps
    logger.info(f"Effective batch size: {effective_bs}")

    # Use format detected BEFORE normalization for response_template
    response_template = "Response:" if original_format == 'alpaca' else "Answer:"
    logger.info(f"Original format: {original_format}, using response_template='{response_template}'")

    # TRL 0.26.0: Use SFTConfig with completion_only_loss instead of DataCollatorForCompletionOnlyLM
    sft_config = SFTConfig(
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
        # TRL 0.26.0: Completion-only loss settings
        dataset_text_field="text",  # Use pre-formatted text directly
        completion_only_loss=True,  # Only compute loss on completion (after response_template)
        response_template=response_template,  # Where completion begins
    )

    logger.info("Creating SFTTrainer (TRL 0.26.0) with completion_only_loss=True...")
    logger.info(f"Response template for completion-only loss: '{response_template}'")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )


    logger.info("=" * 70)
    logger.info("START TRAINING (SFT)")
    logger.info("=" * 70)

    trainer.train()

    logger.info("=" * 70)
    logger.info("FINISHED (SFT)")
    logger.info("=" * 70)


    NT = None
    if eval_dataset is not None:
        logger.info("Evaluating NEW TASK dataset (NT)")
        from evaluation.evaluation import evaluate_new_task
        NT = evaluate_new_task(
            model=trainer.model,
            tokenizer=tokenizer,
            dataset=eval_dataset,
            num_samples=100
        )
        logger.info(f"NT score: {NT:.3f}")

    # cleanup
    gc.collect()
    torch.cuda.empty_cache()

    return trainer.model, NT
