"""
experiment_v2.py - Experiment pipeline using trainingv1 implementations

This version uses the corrected implementations from trainingv1/:
- train_sft_baseline.py - Fixed SFT training
- train_dr_grpo.py - Fixed Dr.GRPO with correct log probs and rewards
- reward.py - Domain-specific correctness checking (math/science/tool)

Keeps the same experiment structure as experiment.py:
- Hyperparameter sweeps for SFT and RL
- KL divergence computation
- Prior task evaluation
- Results tracking and JSON export
"""

import gc
import json
import os
import random
import torch
from transformers import AutoModelForCausalLM

from config.CONFIG import MODEL_NAME, get_config
from evaluation.evaluation import evaluate_benchmarks, compute_forward_kl
from logger import get_logger
from data.dataset_utils import UnifiedDatasetInterface

# Import from trainingv1 instead of training
from trainingv1.train_sft_baseline import train_sft_baseline
from trainingv1.train_dr_grpo import train_dr_grpo

logger = get_logger(__name__)


def run_full_experiment(dataset, tokenizer, dataset_name="math", config_mode="minimal"):
    """
    Run full experiment using trainingv1 implementations.

    Args:
        dataset: Training dataset (will be normalized)
        tokenizer: Tokenizer
        dataset_name: Dataset name for saving results and domain detection
        config_mode: Configuration mode ('quick', 'minimal', 'full')
    """

    sft_cfg, rl_cfg, data_config = get_config(config_mode)
    target_nt = float(data_config.get("target_nt", 70.0))
    kl_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Map dataset name to domain for reward functions
    domain_map = {
        "math": "math",
        "science": "science",
        "tool": "tool"
    }
    domain = domain_map.get(dataset_name, "math")

    logger.info(f"{'='*70}")
    logger.info(f"STARTING EXPERIMENT (trainingv1)")
    logger.info(f"{'='*70}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Domain: {domain}")
    logger.info(f"Config mode: {config_mode}")
    logger.info(f"Max training samples: {data_config['max_samples']}")
    logger.info(f"Target NT: {target_nt}")
    logger.info(f"KL samples: {data_config['kl_samples']}")
    logger.info(f"KL device: {kl_device}")
    logger.info("=" * 70)

    # Train / eval split
    dataset_size = len(dataset)
    eval_size = min(200, int(dataset_size * 0.1))
    indices = list(range(dataset_size))
    random.seed(42)
    random.shuffle(indices)

    train_dataset = dataset.select(indices[:-eval_size])
    eval_dataset_raw = dataset.select(indices[-eval_size:])

    # Normalize eval_dataset so evaluation gets proper answers (especially for MCQ)
    eval_dataset = UnifiedDatasetInterface.normalize_dataset(eval_dataset_raw)

    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Eval size: {len(eval_dataset)}")

    # Results file
    os.makedirs("results", exist_ok=True)
    results_file = f"results/results_{dataset_name}_{config_mode}_v2.json"

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
    logger.info("✓ Base model loaded on CPU")

    # SFT sweep
    logger.info(f"{'='*70}")
    logger.info(f"SFT HYPERPARAMETER SWEEP (trainingv1)")
    logger.info(f"{'='*70}")

    # Paper sweeps both constant_with_warmup and cosine schedulers
    sft_schedulers = sft_cfg.get('schedulers', [sft_cfg.get('lr_scheduler', 'constant_with_warmup')])
    if isinstance(sft_schedulers, str):
        sft_schedulers = [sft_schedulers]

    for lr in sft_cfg['learning_rates']:
        for bs in sft_cfg['batch_sizes']:
            for epochs in sft_cfg['epochs']:
                for sched_type in sft_schedulers:

                    # Check if already done
                    effective_bs = bs * 4  # gradient_accumulation_steps = 4 (from train_sft_baseline)
                    if any(r['lr']==lr and r['batch_size']==effective_bs and r['epochs']==epochs
                           and r.get('scheduler', 'constant_with_warmup')==sched_type
                           for r in results.get('sft', [])):
                        logger.info(f"Skipping SFT lr={lr}, bs={effective_bs}, epochs={epochs}, sched={sched_type} (done)")
                        continue

                    logger.info(f"Training SFT: lr={lr}, bs={effective_bs}, epochs={epochs}, scheduler={sched_type}")

                    # Clone model
                    sft_model = AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME,
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    logger.info("✓ Model loaded")

                    # Use trainingv1 SFT implementation
                    sft_model, NT = train_sft_baseline(
                        model=sft_model,
                        tokenizer=tokenizer,
                        dataset=train_dataset,
                        learning_rate=lr,
                        batch_size=bs,
                        epochs=epochs,
                        max_samples=data_config['max_samples'],
                        eval_dataset=eval_dataset,
                        lr_scheduler_type=sched_type,
                    )

                    # Compute KL divergence on task distribution
                    logger.info(f"✓ Computing KL divergence on task distribution...")
                    if kl_device == "cuda":
                        base_model.to("cuda")

                    kl_div = compute_forward_kl(
                        sft_model,
                        base_model,
                        dataset,
                        tokenizer,
                        num_samples=data_config['kl_samples'],
                        response_only=True
                    )

                    if kl_device == "cuda":
                        base_model.to("cpu")
                        torch.cuda.empty_cache()

                    # Prior task evaluation (PT)
                    logger.info("✓ Evaluating prior task performance (PT)...")
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
                        'batch_size': effective_bs,  # bs * 4
                        'epochs': epochs,
                        'scheduler': sched_type,
                        'NT': NT,
                        'PT': pt_avg,
                        'kl_divergence': kl_div,
                    })

                    logger.info(f"✓ NT: {NT:.2f}%, PT: {pt_avg:.2f}%, KL: {kl_div:.4f}")

                    # Save checkpoint
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2)

                    # Cleanup
                    del sft_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # RL sweep
    logger.info(f"{'='*70}")
    logger.info(f"RL HYPERPARAMETER SWEEP (Dr.GRPO - trainingv1)")
    logger.info(f"{'='*70}")

    # Paper sweeps mu_iterations {1, 2}
    rl_mu_iterations = rl_cfg.get('num_iterations', [2])
    rl_group_size = rl_cfg.get('num_generations', 64)
    rl_prompts_per_gen = rl_cfg.get('prompts_per_generation', 8)
    rl_grad_accum = rl_cfg.get('gradient_accumulation_steps', 1)

    for lr in rl_cfg['learning_rates']:
        for bs in rl_cfg['batch_sizes']:
            for mu_iter in rl_mu_iterations:

                # Check if already done
                if any(r['lr']==lr and r['batch_size']==rl_prompts_per_gen and r.get('mu_iterations', 2)==mu_iter
                       for r in results.get('rl', [])):
                    logger.info(f"Skipping RL lr={lr}, prompts_per_gen={rl_prompts_per_gen}, μ={mu_iter} (done)")
                    continue

                logger.info(f"Training RL (Dr.GRPO): lr={lr}, prompts_per_gen={rl_prompts_per_gen}, grad_accum={rl_grad_accum}, μ={mu_iter}")

                rl_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                )
                logger.info("✓ Model loaded")

                # Use trainingv1 Dr.GRPO implementation
                rl_model, NT = train_dr_grpo(
                    model=rl_model,
                    tokenizer=tokenizer,
                    dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    domain=domain,
                    μ_iterations=mu_iter,
                    lr=lr,
                    group_size=rl_group_size,
                    prompts_per_gen=rl_prompts_per_gen,
                    gradient_accumulation_steps=rl_grad_accum,
                    target_nt=target_nt,
                    max_samples=data_config['max_samples']
                )

                if NT < target_nt:
                    logger.info(f"✗ RL did not reach target NT ({NT:.2f}% < {target_nt}%), skipping KL")
                    del rl_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                # Compute KL divergence
                logger.info(f"✓ Computing KL divergence on task distribution...")
                if kl_device == "cuda":
                    base_model.to("cuda")

                kl_div = compute_forward_kl(
                    rl_model,
                    base_model,
                    dataset,
                    tokenizer,
                    num_samples=data_config["kl_samples"],
                    response_only=True
                )

                if kl_device == "cuda":
                    base_model.to("cpu")
                    torch.cuda.empty_cache()

                # Prior task evaluation (PT)
                logger.info("✓ Evaluating prior task performance (PT)...")
                prior_scores = evaluate_benchmarks(
                    rl_model,
                    tokenizer,
                    limit=int(data_config.get("eval_samples", 100)),
                    use_extended=False,
                )
                pt_avg = float(prior_scores.get("average", 0.0)) * 100.0

                # Save results using configured prompts_per_gen for Dr.GRPO
                results['rl'].append({
                    'lr': lr,
                    'batch_size': rl_prompts_per_gen,
                    'mu_iterations': mu_iter,
                    'NT': NT,
                    'PT': pt_avg,
                    'kl_divergence': kl_div,
                })

                logger.info(f"✓ NT: {NT:.2f}%, PT: {pt_avg:.2f}%, KL: {kl_div:.4f}")

                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)

                # Cleanup
                del rl_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Cleanup base model
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"{'='*70}")
    logger.info(f"EXPERIMENT COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"SFT runs: {len(results['sft'])}")
    logger.info(f"RL runs: {len(results['rl'])}")
    logger.info(f"{'='*70}")

    return results

