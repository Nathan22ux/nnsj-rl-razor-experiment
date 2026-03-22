"""
experiment_v2.py - Experiment pipeline using trainingv1 implementations

This version uses the corrected implementations from trainingv1/:
- train_sft_baseline.py - Fixed SFT training
- train_dr_grpo.py - Fixed Dr.GRPO with correct log probs and rewards
- reward.py - Domain-specific correctness checking (math/science/tool)
"""

import gc
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM

from config.CONFIG import LIMIT_PER_BENCHMARK, MODEL_NAME, get_config
from data.dataset_utils import UnifiedDatasetInterface
# from evaluation.evaluation import compute_forward_kl, evaluate_benchmarks
from evaluation.evaluation import evaluate_benchmarks
from trainingv1.eval_kl_forward import compute_forward_kl
from logger import get_logger
from trainingv1.train_dr_grpo import train_dr_grpo
from trainingv1.train_sft_baseline import train_sft_baseline

logger = get_logger(__name__)


def run_full_experiment(dataset, tokenizer, dataset_name="math", config_mode="minimal"):
    """
    Run full experiment using trainingv1 implementations.
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
        logger.info("Found existing results: %s", results_file)
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        logger.info(
            "Completed: %s SFT, %s RL",
            len(results.get("sft", [])),
            len(results.get("rl", [])),
        )

    # Use Flash Attention 2 when available for 2-4x attention speedup
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info("Flash Attention 2 available")
    except ImportError:
        attn_impl = "eager"
        logger.info("Flash Attention 2 not available, using eager attention")

    logger.info("Loading base model: %s (attn: %s)", MODEL_NAME, attn_impl)
    base_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=base_dtype,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    logger.info("Base model loaded on CPU")

    logger.info("%s", "=" * 70)
    logger.info("SFT HYPERPARAMETER SWEEP (trainingv1)")
    logger.info("%s", "=" * 70)

    sft_schedulers = sft_cfg.get("schedulers")
    if not sft_schedulers:
        scheduler_cfg = sft_cfg.get("lr_scheduler", "constant_with_warmup")
        if isinstance(scheduler_cfg, (list, tuple)):
            sft_schedulers = list(scheduler_cfg)
        else:
            sft_schedulers = [scheduler_cfg]

    for lr in sft_cfg["learning_rates"]:
        for bs in sft_cfg["batch_sizes"]:
            for epochs in sft_cfg["epochs"]:
                for scheduler in sft_schedulers:
                    effective_bs = bs * sft_cfg.get("gradient_accumulation_steps", 4)
                    if any(
                        r.get("lr") == lr
                        and r.get("batch_size") == effective_bs
                        and r.get("epochs") == epochs
                        and r.get("lr_scheduler", "constant_with_warmup") == scheduler
                        for r in results.get("sft", [])
                    ):
                        logger.info(
                            "Skipping SFT lr=%s, bs=%s, epochs=%s, scheduler=%s (done)",
                            lr,
                            effective_bs,
                            epochs,
                            scheduler,
                        )
                        continue

                    logger.info(
                        "Training SFT: lr=%s, bs=%s, epochs=%s, scheduler=%s",
                        lr,
                        effective_bs,
                        epochs,
                        scheduler,
                    )

                    sft_model = AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME,
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto",
                        trust_remote_code=True,
                        attn_implementation=attn_impl,
                    )
                    logger.info("Model loaded")

                    sft_model, nt = train_sft_baseline(
                        model=sft_model,
                        tokenizer=tokenizer,
                        dataset=train_dataset,
                        learning_rate=lr,
                        batch_size=bs,
                        epochs=epochs,
                        max_samples=data_config["max_samples"],
                        eval_dataset=eval_dataset,
                        lr_scheduler_type=scheduler,
                    )

                    model_save_path = f"./results_sft/lr{lr}_bs{bs}_ep{epochs}_{scheduler}/model"
                    logger.info("Saving SFT model to %s", model_save_path)
                    sft_model.save_pretrained(model_save_path)
                    tokenizer.save_pretrained(model_save_path)
                    logger.info("SFT model saved")

                    logger.info("Computing KL divergence on task distribution...")
                    if kl_device == "cuda":
                        base_model.to("cuda")

                    kl_div = compute_forward_kl(
                        base_model=base_model,
                        target_model=sft_model,
                        tokenizer=tokenizer,
                        dataset=train_dataset,
                        num_samples=data_config["kl_samples"],
                    )

                    if kl_device == "cuda":
                        base_model.to("cpu")
                        torch.cuda.empty_cache()

                    logger.info("Evaluating prior task performance (PT)...")
                    prior_scores = evaluate_benchmarks(
                        sft_model,
                        tokenizer,
                        limit=LIMIT_PER_BENCHMARK,
                        use_extended=False,
                    )
                    pt_avg = float(prior_scores.get("average", 0.0)) * 100.0

                    results["sft"].append(
                        {
                            "lr": lr,
                            "batch_size": effective_bs,
                            "epochs": epochs,
                            "lr_scheduler": scheduler,
                            "NT": nt,
                            "PT": pt_avg,
                            "kl_divergence": kl_div,
                        }
                    )

                    logger.info("NT: %.2f%%, PT: %.2f%%, KL: %.4f", nt, pt_avg, kl_div)

                    with open(results_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2)

                    del sft_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    logger.info("%s", "=" * 70)
    logger.info("RL HYPERPARAMETER SWEEP (Dr.GRPO - trainingv1)")
    logger.info("%s", "=" * 70)

    rl_group_size = int(rl_cfg.get("num_generations", 64))
    rl_iterations = rl_cfg.get("num_iterations", [2])
    if not isinstance(rl_iterations, (list, tuple)):
        rl_iterations = [int(rl_iterations)]

    for lr in rl_cfg["learning_rates"]:
        for bs in rl_cfg["batch_sizes"]:
            for mu in rl_iterations:
                grad_acc_steps = int(rl_cfg.get("gradient_accumulation_steps", 1))
                effective_bs = bs * grad_acc_steps

                if any(
                    r.get("lr") == lr
                    and r.get("batch_size") == effective_bs
                    and r.get("num_iterations") is not None
                    and int(r.get("num_iterations")) == int(mu)
                    for r in results.get("rl", [])
                ):
                    logger.info(
                        "Skipping RL lr=%s, bs=%s, mu=%s (done)",
                        lr,
                        effective_bs,
                        mu,
                    )
                    continue

                logger.info(
                    "Training RL (Dr.GRPO): lr=%s, prompts_per_gen=%s, group_size=%s, "
                    "mu=%s, grad_accum=%s, effective_bs=%s",
                    lr,
                    bs,
                    rl_group_size,
                    mu,
                    grad_acc_steps,
                    effective_bs,
                )

                rl_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation=attn_impl,
                )
                logger.info("Model loaded")

                rl_model, nt = train_dr_grpo(
                    model=rl_model,
                    tokenizer=tokenizer,
                    dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    domain=domain,
                    μ_iterations=int(mu),
                    lr=lr,
                    group_size=rl_group_size,
                    prompts_per_gen=bs,
                    gradient_accumulation_steps=grad_acc_steps,
                    target_nt=target_nt,
                    max_samples=data_config["max_samples"],
                    max_completion_length=int(rl_cfg.get("max_completion_length", 512)),
                    warmup_steps=int(rl_cfg.get("warmup_steps", 50)),
                )

                model_save_path = f"./results_rl/lr{lr}_mu{mu}/model"
                logger.info("Saving RL model to %s", model_save_path)
                rl_model.save_pretrained(model_save_path)
                tokenizer.save_pretrained(model_save_path)
                logger.info("RL model saved")

                if nt < target_nt:
                    logger.info(
                        "RL did not reach target NT (%.2f%% < %.2f%%), skipping KL",
                        nt,
                        target_nt,
                    )
                    del rl_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                logger.info("Computing KL divergence on task distribution...")
                if kl_device == "cuda":
                    base_model.to("cuda")

                kl_div = compute_forward_kl(
                    base_model=base_model,
                    target_model=rl_model,
                    tokenizer=tokenizer,
                    dataset=train_dataset,
                    num_samples=data_config["kl_samples"],
                )

                if kl_device == "cuda":
                    base_model.to("cpu")
                    torch.cuda.empty_cache()

                logger.info("Evaluating prior task performance (PT)...")
                prior_scores = evaluate_benchmarks(
                    rl_model,
                    tokenizer,
                    limit=LIMIT_PER_BENCHMARK,
                    use_extended=False,
                )
                pt_avg = float(prior_scores.get("average", 0.0)) * 100.0

                results["rl"].append(
                    {
                        "lr": lr,
                        "batch_size": effective_bs,
                        "num_iterations": int(mu),
                        "NT": nt,
                        "PT": pt_avg,
                        "kl_divergence": kl_div,
                    }
                )

                logger.info("NT: %.2f%%, PT: %.2f%%, KL: %.4f", nt, pt_avg, kl_div)

                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

                del rl_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("%s", "=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("%s", "=" * 70)
    logger.info("Results saved to: %s", results_file)
    logger.info("SFT runs: %s", len(results["sft"]))
    logger.info("RL runs: %s", len(results["rl"]))
    logger.info("%s", "=" * 70)

    return results
