"""
Experiment orchestration module.

This module runs the full RL's Razor replication experiment with:
- Hyperparameter sweeps for SFT and GRPO
- Model evaluation and metric computation
- Result aggregation and visualization
"""

import gc
import json
import logging
import os

import torch
from transformers import AutoModelForCausalLM

from config import (
    MODEL_NAME,
    data_config,
    rl_config,
    sft_config,
)
from evaluation import compute_forward_kl, compute_kl_on_task_distribution, evaluate_benchmarks
from training import train_grpo, train_sft

logger = logging.getLogger(__name__)




def run_full_experiment(dataset, tokenizer, dataset_name="math", config_mode="minimal"):
    """
    Run full experiment with hyperparameter sweeps.

    Args:
        dataset: Training dataset
        tokenizer: Tokenizer
        dataset_name: Dataset name for saving results
        config_mode: Configuration mode ('quick', 'minimal', 'full')

    Returns:
        dict: Experiment results with 'sft' and 'rl' keys
    """
    logger.info("=" * 70)
    logger.info("STARTING EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Max training samples: {data_config['max_samples']}")
    logger.info(f"KL samples: {data_config['kl_samples']}")
    logger.info("=" * 70)

    logger.info("Creating train/eval split...")
    dataset_size = len(dataset)
    eval_size = min(200, int(dataset_size * 0.1))  # 10% or 200, whichever is smaller
    train_size = dataset_size - eval_size

    # Shuffle indices for random split
    import random
    all_indices = list(range(dataset_size))
    random.seed(42)  # Reproducibility
    random.shuffle(all_indices)

    train_indices = all_indices[:train_size]
    eval_indices = all_indices[train_size:]

    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(eval_indices)

    logger.info(f"Train set: {len(train_dataset)} examples")
    logger.info(f"Eval set: {len(eval_dataset)} examples")

    # Check for existing results to resume from
    os.makedirs("results", exist_ok=True)
    results_file = f"results/results_{dataset_name}_{config_mode}.json"
    
    if os.path.exists(results_file):
        logger.info(f"Found existing results: {results_file}")
        with open(results_file, 'r') as f:
            results = json.load(f)
        logger.info(f"Completed: {len(results.get('sft', []))} SFT, {len(results.get('rl', []))} RL")
    else:
        results = {'sft': [], 'rl': []}
    
    # Load base model ONCE
    logger.info(f"Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    logger.info("Base model loaded")
    
    # SFT sweep
    logger.info(f"{'='*70}")
    logger.info(f"SFT HYPERPARAMETER SWEEP")
    logger.info(f"{'='*70}")
    
    for lr in sft_cfg['learning_rates']:
        for bs in sft_cfg['batch_sizes']:
            for epochs in sft_cfg['epochs']:
                
                # Check if already done
                if any(r['lr']==lr and r['batch_size']==bs and r['epochs']==epochs 
                       for r in results.get('sft', [])):
                    logger.info(f"Skipping SFT lr={lr}, bs={bs}, epochs={epochs} (done)")
                    continue
                
                logger.info(f"Training SFT: lr={lr}, bs={bs}, epochs={epochs}")
                
                # Clone model
                sft_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
                )
                logger.info("Model loaded")

                # FIX: Pass max_samples from data_config
                # FIX: Pass max_samples from data_config and use train_dataset
                sft_model, trainer, NT = train_sft(
                    sft_model, train_dataset, tokenizer,
                    learning_rate=lr,
                    batch_size=bs,
                    epochs=epochs,
                    max_samples=data_config['max_samples'],
                    eval_dataset=eval_dataset
                )
                
                # Evaluate
                logger.info("Evaluating trained SFT model...")
                prior_scores = evaluate_benchmarks(sft_model, tokenizer)

                # FIX: Use response_only=True and num_samples from config
                # FIX: Use task distribution KL (paper's method)
                logger.info("Computing KL divergence on task distribution...")
                task_prompts = []
                for i in range(min(data_config['kl_samples'], len(dataset))):
                    question = dataset[i]['0']['value']
                    task_prompts.append(f"Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer:")

                kl_div = compute_kl_on_task_distribution(
                    sft_model, base_model, tokenizer, task_prompts,
                    num_samples=data_config['kl_samples']
                )
                
                # Save results
                results['sft'].append({
                    'lr': lr,
                    'batch_size': bs,
                    'epochs': epochs,
                    'NT': NT,
                    'PT': prior_scores['average'],
                    'kl_divergence': kl_div,
                    'detailed_scores': prior_scores,
                })
                
                logger.info(f"NT: {NT:.2f}%, PT: {prior_scores['average']:.4f}, KL: {kl_div:.4f}")
                
                # Save checkpoint
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Cleanup
                del sft_model, trainer
                torch.cuda.empty_cache()
                gc.collect()
    
    # RL sweep (similar structure)
    logger.info(f"{'='*70}")
    logger.info(f"RL HYPERPARAMETER SWEEP")
    logger.info(f"{'='*70}")
    
    for lr in rl_cfg['learning_rates']:
        for bs in rl_cfg['batch_sizes']:
            
            if any(r['lr']==lr and r['batch_size']==bs for r in results.get('rl', [])):
                logger.info(f"Skipping RL lr={lr}, bs={bs} (done)")
                continue
            
            logger.info(f"Training RL: lr={lr}, bs={bs}")
            
            rl_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
            )
            logger.info("Model loaded")

            # FIX: Pass batch_size, max_samples, and use train_dataset
            rl_model, trainer, NT = train_grpo(
                rl_model, train_dataset, tokenizer,
                learning_rate=lr,
                batch_size=bs,
                max_samples=data_config['max_samples'],
                eval_dataset=eval_dataset
            )

            # Evaluate
            logger.info("Evaluating trained RL model...")
            prior_scores = evaluate_benchmarks(rl_model, tokenizer)

            # FIX: Use response_only=True and num_samples from config
            logger.info("Computing KL divergence on task distribution...")
            task_prompts = []
            for i in range(min(data_config['kl_samples'], len(dataset))):
                question = dataset[i]['0']['value']
                task_prompts.append(f"Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer:")

            kl_div = compute_kl_on_task_distribution(
                rl_model, base_model, tokenizer, task_prompts,
                num_samples=data_config['kl_samples']
            )
            
            results['rl'].append({
                'lr': lr,
                'batch_size': bs,
                'NT': NT,
                'PT': prior_scores['average'],
                'kl_divergence': kl_div,
                'detailed_scores': prior_scores,
            })
            
logger.info(f"NT: {NT:.2f}%, PT: {prior_scores['average']:.4f}, KL: {kl_div:.4f}")
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            del rl_model, trainer
            torch.cuda.empty_cache()
            gc.collect()
    
    # Cleanup base model
    del base_model
    torch.cuda.empty_cache()
    
    logger.info(f"{'='*70}")
    logger.info(f"EXPERIMENT COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"SFT runs: {len(results['sft'])}")
    logger.info(f"RL runs: {len(results['rl'])}")
    logger.info(f"{'='*70}")
    
    return results