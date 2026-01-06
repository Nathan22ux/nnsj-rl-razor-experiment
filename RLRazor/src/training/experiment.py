"""
Corrected experiment.py for RL's Razor Replication

FIXES APPLIED:
- Imports data_config for max_samples
- Passes max_samples to train_sft() and train_grpo()
- Passes batch_size to train_grpo()
- Uses response_only=True for KL computation
- Sweeps batch_size for RL (fair comparison with SFT)
"""

import gc
import json
import os
import random
import torch
from transformers import AutoModelForCausalLM

from config.CONFIG import MODEL_NAME, get_config
from evaluation.evaluation import evaluate_benchmarks, compute_forward_kl, compute_kl_on_task_distribution
from logger import get_logger
from training.training import train_sft, train_grpo
from data.dataset_utils import UnifiedDatasetInterface

logger = get_logger(__name__)


def extract_questions_from_dataset(dataset, num_samples):
    """
    Extract questions from any dataset format using UnifiedDatasetInterface.

    Args:
        dataset: Dataset in any supported format
        num_samples: Number of questions to extract

    Returns:
        List of formatted prompts for KL divergence computation
    """
    task_prompts = []

    # Normalize first sample to detect format
    first_example = dataset[0]
    format_hint = UnifiedDatasetInterface.detect_format(first_example)

    for i in range(min(num_samples, len(dataset))):
        normalized = UnifiedDatasetInterface.normalize_example(dataset[i], format_hint)
        question = normalized['question']
        prompt = f"Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer:"
        task_prompts.append(prompt)

    return task_prompts


def run_full_experiment(dataset, tokenizer, dataset_name="math", config_mode="minimal"):
    """
    Run full experiment with FIXED implementations.

    Args:
        dataset: Training dataset (will be normalized)
        tokenizer: Tokenizer
        dataset_name: Dataset name for saving results
        config_mode: Configuration mode ('quick', 'minimal', 'full')
    """

    sft_cfg, rl_cfg, data_config = get_config(config_mode)
    target_nt = float(data_config.get("target_nt", 70.0))
    kl_device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"{'='*70}")
    logger.info(f"STARTING EXPERIMENT")
    logger.info(f"{'='*70}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Config mode: {config_mode}")
    logger.info(f"Max training samples: {data_config['max_samples']}")
    logger.info(f"Target NT: {target_nt}")
    logger.info(f"KL samples: {data_config['kl_samples']}")
    logger.info(f"KL device: {kl_device}")
    logger.info("=" * 70)

    # train/eval split
    logger.info("Creating train/eval split...")
    dataset_size = len(dataset)
    eval_size = min(200, int(dataset_size * 0.1))  # 10% or 200, whichever is smaller
    train_size = dataset_size - eval_size

    # Train / eval split
    dataset_size = len(dataset)
    eval_size = min(200, int(dataset_size * 0.1))
    indices = list(range(dataset_size))
    random.seed(42)
    random.shuffle(indices)

    train_dataset = dataset.select(indices[:-eval_size])
    eval_dataset = dataset.select(indices[-eval_size:])

    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Eval size: {len(eval_dataset)}")

    # Results file
    os.makedirs("results", exist_ok=True)
    results_file = f"results/results_{dataset_name}_{config_mode}.json"

    results = {"sft": [], "rl": []}
    if os.path.exists(results_file):
        logger.info(f"Found existing results: {results_file}")
        with open(results_file, "r") as f:
            results = json.load(f)
            logger.info(f"Completed: {len(results.get('sft', []))} SFT, {len(results.get('rl', []))} RL")

    # Load base model ONCE (used for KL computation)
    logger.info(f"Loading base model: {MODEL_NAME}")
    base_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=base_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    logger.info(" Base model loaded, loaded on CPU")

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
                    MODEL_NAME,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map=None, 
                    trust_remote_code=True,
                )
                if torch.cuda.is_available():
                    sft_model = sft_model.to("cuda")
                logger.info(" Model loaded")

                # FIX: Pass max_samples from data_config
                # FIX: Pass max_samples from data_config and use train_dataset
                sft_model, trainer, NT = train_sft(
                    sft_model, 
                    train_dataset, 
                    tokenizer,
                    learning_rate=lr,
                    batch_size=bs,
                    epochs=epochs,
                    max_samples=data_config['max_samples'],
                    eval_dataset=eval_dataset
                )

                # Evaluate
                # logger.info(f" Evaluating trained SFT model...")
                # # prior_scores = evaluate_benchmarks(sft_model, tokenizer)

                # FIX: Use response_only=True and num_samples from config
                # FIX: Use task distribution KL (paper's method)
                logger.info(f" Computing KL divergence on task distribution...")
                task_prompts = extract_questions_from_dataset(dataset, data_config['kl_samples'])

                if kl_device == "cuda":
                    base_model.to("cuda")
                # kl_div = compute_kl_on_task_distribution(
                #     sft_model, 
                #     base_model, 
                #     tokenizer, 
                #     task_prompts,
                #     num_samples=data_config['kl_samples']
                # )
                kl_div = compute_forward_kl(
                    sft_model,
                    base_model,
                    dataset, # Pass the full dataset here
                    tokenizer,
                    num_samples=data_config['kl_samples'],
                    response_only=True # Important: Focus on the answer part
                )
                if kl_device == "cuda":
                    base_model.to("cpu")
                    torch.cuda.empty_cache()

                # Prior task evaluation (PT) - used for Pareto plots
                logger.info(" Evaluating prior task performance (PT)...")
                prior_scores = evaluate_benchmarks(
                    sft_model,
                    tokenizer,
                    limit=int(data_config.get("eval_samples", 100)),
                    use_extended=False,
                )
                pt_avg = float(prior_scores.get("average", 0.0)) * 100.0

                # Save results
                results['sft'].append({
                    'lr': lr,
                    'batch_size': bs,
                    'epochs': epochs,
                    'NT': NT,
                    'PT': pt_avg,
                    'kl_divergence': kl_div,
                })

                logger.info(f" NT: {NT:.2f}%, PT: {pt_avg:.2f}%, KL_divergence: {kl_div:.4f}")

                # Save checkpoint
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Cleanup
                logger.info("Checking if model and trainer are deleted")
                if sft_model is not None:
                    logger.info("SFT Model is not deleted")
                    del sft_model
                    logger.info("SFT Model deleted")
                if trainer is not None:
                    logger.info("Trainer is not deleted")
                    trainer.model = None
                    trainer.optimizer = None
                    trainer.lr_scheduler = None
                    del trainer
                    logger.info("Trainer deleted")
                logger.info("Checking if CUDA cache is empty")
                if torch.cuda.is_available():
                    logger.info("CUDA is available")
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    logger.info("CUDA is not available")

    # RL sweep 
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
                MODEL_NAME,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=None,  
                trust_remote_code=True,
            )
            if torch.cuda.is_available():
                rl_model = rl_model.to("cuda")
            logger.info(" Model loaded")

            # FIX: Pass batch_size, max_samples, and use train_dataset
            rl_model, trainer, NT = train_grpo(
                rl_model, 
                train_dataset, 
                tokenizer,
                learning_rate=lr,
                batch_size=bs,
                max_samples=data_config['max_samples'],
                eval_dataset=eval_dataset,
                target_nt=target_nt,
                # Wire GRPO paper hyperparameters from config
                num_generations=int(rl_cfg.get("num_generations", 16)),
                prompts_per_generation=int(rl_cfg.get("prompts_per_generation", bs)),
            )

            if NT < target_nt:
                logger.info("RL did not reach target NT, skipping KL")
                trainer.model = None
                del trainer, rl_model
                gc.collect()
                torch.cuda.empty_cache()
                continue

            # ---- KL computation
            logger.info(f" Computing KL divergence on task distribution...")
            task_prompts = extract_questions_from_dataset(
                dataset, data_config["kl_samples"]
            )
            if kl_device == "cuda":
                base_model.to("cuda")
            # kl_div = compute_kl_on_task_distribution(
            #     rl_model,
            #     base_model,
            #     tokenizer,
            #     task_prompts,
            #     num_samples=data_config["kl_samples"],
            # )
            kl_div = compute_forward_kl(
                    sft_model,
                    base_model,
                    dataset, # Pass the full dataset here
                    tokenizer,
                    num_samples=data_config['kl_samples'],
                    response_only=True # Important: Focus on the answer part
                )
            if kl_device == "cuda":
                base_model.to("cpu")
                torch.cuda.empty_cache()

            # Prior task evaluation (PT)
            logger.info(" Evaluating prior task performance (PT)...")
            prior_scores = evaluate_benchmarks(
                rl_model,
                tokenizer,
                limit=int(data_config.get("eval_samples", 100)),
                use_extended=False,
            )
            pt_avg = float(prior_scores.get("average", 0.0)) * 100.0

            # Evaluate
            # logger.info(f" Evaluating trained RL model...")
            # prior_scores = evaluate_benchmarks(rl_model, tokenizer)

            # FIX: Use response_only=True and num_samples from config

            results['rl'].append({
                'lr': lr,
                'batch_size': bs,
                'NT': NT,
                'PT': pt_avg,
                'kl_divergence': kl_div,
            })

            logger.info(f" NT: {NT:.2f}%, PT: {pt_avg:.2f}%, KL_divergence: {kl_div:.4f}")

            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            #clean up
            logger.info("Checking if model and trainer are deleted")
            if rl_model is not None:
                logger.info("RL Model is not deleted")
                del rl_model
                logger.info("RL Model deleted")
            if trainer is not None:
                logger.info("Trainer is not deleted")
                trainer.model = None
                trainer.optimizer = None
                trainer.lr_scheduler = None
                del trainer
                logger.info("Trainer deleted")
            logger.info("Checking if CUDA cache is empty")
            if torch.cuda.is_available():   
                logger.info("CUDA is available")
                gc.collect()
                torch.cuda.empty_cache()
            else:
                logger.info("CUDA is not available")

    # Cleanup base model
    logger.info("Checking if base model is deleted")
    if base_model is not None:
        logger.info("Base model is not deleted")
        del base_model
    logger.info("Checking if CUDA cache is empty")
    if torch.cuda.is_available():
        logger.info("CUDA is available")
        gc.collect()
        torch.cuda.empty_cache()
    else:
        logger.info("CUDA is not available")

    logger.info(f"{'='*70}")
    logger.info(f"EXPERIMENT COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"SFT runs: {len(results['sft'])}")
    logger.info(f"RL runs: {len(results['rl'])}")
    logger.info(f"{'='*70}")

    return results