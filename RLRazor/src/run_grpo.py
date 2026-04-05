"""
run_grpo.py — Standalone Dr.GRPO (RL) training for RL's Razor replication.

Paper hyperparameters (Table from paper):
- Model: Qwen2.5 3B-Instruct
- LR: Custom subset [1.041e-05, 3.617e-05]
- Scheduler: Sweep [constant_with_warmup, cosine_with_warmup]
- Optimizer: AdamW, weight_decay=0
- Warmup: 50 steps
- bf16: True
- Max Grad Norm: 1.0
- KL reg: 0 (implicit)
- Group Size: 64 (Reverted to paper standard)
- Prompts per generation: 8 (Reverted to paper standard)
- μ iterations: [1] (Strictly 1 epoch for RL's Razor)
- Loss type: Dr.GRPO

Usage:
    python run_grpo.py --dataset math
    python run_grpo.py --dataset science
    python run_grpo.py --dataset tool
"""

import argparse
import gc
import json
import os
import random
import sys
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.CONFIG import (
    MODEL_NAME, LIMIT_PER_BENCHMARK, FULL_LR_SWEEP, RL_ITERATIONS,
    NUM_GENERATIONS, PROMPTS_PER_GENERATION, WARMUP_STEPS, GRADIENT_ACCUMULATION_STEPS
)
from data.load_data import load_dataset_byname
from data.dataset_utils import UnifiedDatasetInterface
from evaluation.evaluation import evaluate_benchmarks, compute_forward_kl
from trainingv1.train_dr_grpo import train_dr_grpo
from logger import get_logger

logger = get_logger(__name__)

# Paper-exact RL (Dr.GRPO) hyperparameters

LEARNING_RATES = [1.041689121305663e-05,0.000015774069623151423,1e-05,3e-05]
SCHEDULERS = ["constant_with_warmup"]
GROUP_SIZE = NUM_GENERATIONS
PROMPTS_PER_GEN = PROMPTS_PER_GENERATION
MU_ITERATIONS = RL_ITERATIONS
MAX_SAMPLES = 2200                
MAX_COMPLETION_LENGTH = 512     
WARMUP_STEPS = WARMUP_STEPS
KL_SAMPLES = 200
GRADIENT_ACCUMULATION_STEPS = 4
# TARGET_NT = 70.0                 


# Plotting helpers

def plot_grpo_results(results, dataset_name, output_dir="results"):
    """Generate plots from completed GRPO runs."""
    if not results:
        logger.info("No results to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    labels = []
    nt_scores = []
    pt_scores = []
    kl_vals = []

    for r in results:
        sched_short = r.get("scheduler", "const").split("_")[0]
        lbl = f"lr={r['lr']:.2e}\nμ={r['num_iterations']}\n{sched_short}"
        labels.append(lbl)
        nt_scores.append(r.get("NT", 0) or 0)
        pt_scores.append(r.get("PT", 0) or 0)
        kl_vals.append(r.get("kl_divergence", 0) or 0)

    x = np.arange(len(labels))
    w = 0.35

    # ── 1. NT vs PT grouped bar chart ─────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 5))
    ax.bar(x - w / 2, nt_scores, w, label="NT (New Task %)", color="#4C72B0")
    ax.bar(x + w / 2, pt_scores, w, label="PT (Prior Task %)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Dr.GRPO — NT vs PT Accuracy ({dataset_name})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"grpo_{dataset_name}_nt_pt.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved NT/PT chart → {path}")

    # ── 2. KL divergence bar chart ────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 5))
    bars = ax.bar(x, kl_vals, color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Forward KL(π₀ ∥ π)")
    ax.set_title(f"Dr.GRPO — Forward KL Divergence ({dataset_name})")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, kl_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, f"grpo_{dataset_name}_kl.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved KL chart → {path}")

    # ── 3. Forgetting Law: KL vs PT scatter ───────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(kl_vals, pt_scores, s=80, c="#C44E52", edgecolors="black", zorder=3)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl.replace("\n", " "), (kl_vals[i], pt_scores[i]),
                    fontsize=7, textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Forward KL(π₀ ∥ π)")
    ax.set_ylabel("PT Accuracy (%)")
    ax.set_title(f"Dr.GRPO — Forgetting Law ({dataset_name})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"grpo_{dataset_name}_forgetting_law.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Forgetting Law scatter → {path}")


def plot_grpo_training_curves(checkpoint_dir, run_id, dataset_name, output_dir="results"):
    """Plot loss and reward curves from train_dr_grpo's training_metrics.json."""
    metrics_path = os.path.join(checkpoint_dir, "training_metrics.json")
    if not os.path.exists(metrics_path):
        logger.info(f"No training metrics found at {metrics_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    if not metrics:
        return

    steps = [m["step"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    rewards = [m["reward_mean"] for m in metrics]
    mus = [m.get("mu", 1) for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(steps, losses, linewidth=1.2, color="#4C72B0")
    # Mark μ boundaries
    prev_mu = mus[0]
    for i, mu in enumerate(mus):
        if mu != prev_mu:
            ax1.axvline(x=steps[i], color="gray", linestyle="--", alpha=0.5)
            ax1.text(steps[i], max(losses) * 0.95, f"μ={mu}", fontsize=8, color="gray")
            prev_mu = mu
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Dr.GRPO Training Loss — {run_id}")
    ax1.grid(alpha=0.3)

    # Reward curve
    ax2.plot(steps, rewards, linewidth=1.2, color="#55A868")
    prev_mu = mus[0]
    for i, mu in enumerate(mus):
        if mu != prev_mu:
            ax2.axvline(x=steps[i], color="gray", linestyle="--", alpha=0.5)
            prev_mu = mu
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Mean Reward")
    ax2.set_title(f"Dr.GRPO Mean Reward — {run_id}")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, f"grpo_{dataset_name}_curves_{run_id}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved training curves → {path}")


def main():
    parser = argparse.ArgumentParser(description="RL's Razor — Dr.GRPO training")
    parser.add_argument("--dataset", type=str, default="math",
                        choices=["math", "science", "tool"],
                        help="Dataset/domain to train on")
    args = parser.parse_args()

    dataset_name = args.dataset
    domain = dataset_name  # math/science/tool maps directly

    logger.info("=" * 70)
    logger.info("RL's Razor — Standalone Dr.GRPO Training")
    logger.info("=" * 70)
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        logger.info(f"CUDA available — device: {torch.cuda.get_device_name(device_id)}")
    else:
        logger.info("CUDA not available — running on CPU")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Domain: {domain}")
    logger.info(f"LRs: {LEARNING_RATES}")
    logger.info(f"Schedulers: {SCHEDULERS}")
    logger.info(f"Group size: {GROUP_SIZE}")
    logger.info(f"Prompts/gen: {PROMPTS_PER_GEN}")
    logger.info(f"μ iterations: {MU_ITERATIONS}")
    logger.info(f"Max samples: {MAX_SAMPLES}")
    logger.info(f"Max completion length: {MAX_COMPLETION_LENGTH}")
    logger.info(f"Warmup steps: {WARMUP_STEPS}")
    logger.info(f"KL samples: {KL_SAMPLES}")
    logger.info("=" * 70)

    # ── Load dataset ──────────────────────────────────────
    logger.info(f"[Step 1/3] Loading dataset: '{dataset_name}'...")
    dataset = load_dataset_byname(dataset_name)

    # Train / eval split (deterministic, same seed as SFT for fair comparison)
    dataset_size = len(dataset)
    eval_size = min(200, int(dataset_size * 0.1))
    indices = list(range(dataset_size))
    random.seed(42)
    random.shuffle(indices)

    train_dataset_raw = dataset.select(indices[:-eval_size])
    eval_dataset_raw = dataset.select(indices[-eval_size:])

    logger.info("Normalizing train and eval datasets...")
    train_dataset = UnifiedDatasetInterface.normalize_dataset(train_dataset_raw)
    eval_dataset = UnifiedDatasetInterface.normalize_dataset(eval_dataset_raw)

    logger.info(f"Dataset loaded — Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    # ── Native PyTorch SDPA Attention (Alternative to flash-attn) ─────────
    # This takes advantage of PyTorch >= 2.0 native Flash Attention
    # implementation without needing the external flash-attn package.
    attn_impl = "sdpa"

    # ── Base model for KL (kept on CPU) ───────────────────
    logger.info(f"[Step 2/3] Loading base model '{MODEL_NAME}' onto CPU (used for KL reference)...")
    base_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=base_dtype, device_map="cpu",
        trust_remote_code=True, attn_implementation=attn_impl,
    )
    logger.info("Base model loaded on CPU.")
    logger.info(f"Loading tokenizer for '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer ready.")

    # ── Results ───────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    results_file = f"results/grpo_{dataset_name}.json"
    results = []
    if os.path.exists(results_file):
        logger.info(f"Found existing results file: {results_file} — resuming from checkpoint.")
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} previously completed run(s).")
    else:
        logger.info(f"No existing results found. Starting fresh sweep.")

    # ── Sweep ─────────────────────────────────────────────
    total_runs = len(LEARNING_RATES) * len(MU_ITERATIONS) * len(SCHEDULERS)
    completed_runs = 0
    logger.info(f"[Step 3/3] Starting hyperparameter sweep — {total_runs} total run(s) planned.")
    
    pbar = tqdm(total=total_runs, desc="Hyperparameter Sweep runs")
    for lr in LEARNING_RATES:
        for mu in MU_ITERATIONS:
            for scheduler_type in SCHEDULERS:
                run_id = f"lr{lr:.2e}_mu{mu}_{scheduler_type}"

                # Skip if done
                if any(
                    r.get("lr") == lr and r.get("num_iterations") == mu and r.get("scheduler") == scheduler_type
                    for r in results
                ):
                    logger.info(f"Skipping {run_id} (already done). [{completed_runs + 1}/{total_runs}]")
                    completed_runs += 1
                    pbar.update(1)
                    continue

                completed_runs += 1
                logger.info("=" * 70)
                logger.info(f"GRPO RUN [{completed_runs}/{total_runs}]: {run_id}  (lr={lr}, μ={mu}, scheduler={scheduler_type})")
                logger.info("=" * 70)

                logger.info(f"Loading RL model '{MODEL_NAME}' onto GPU...")
                rl_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto", trust_remote_code=True,
                    attn_implementation=attn_impl,
                )
                
                logger.info("RL model loaded.")

                checkpoint_dir = f"./checkpoints/rl/{run_id}"
                logger.info(f"  [Run {completed_runs}/{total_runs} | Phase 1/3] Starting Dr.GRPO training — checkpoint dir: {checkpoint_dir}")
                rl_model, nt = train_dr_grpo(
                    model=rl_model,
                    tokenizer=tokenizer,
                    dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    domain=domain,
                    μ_iterations=mu,
                    lr=lr,
                    group_size=GROUP_SIZE,
                    prompts_per_gen=PROMPTS_PER_GEN,
                    max_samples=MAX_SAMPLES,
                    max_completion_length=MAX_COMPLETION_LENGTH,
                    warmup_steps=WARMUP_STEPS,
                    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                    checkpoint_dir=checkpoint_dir,
                    scheduler_type=scheduler_type,
                    # target_nt=TARGET_NT,
                )

                logger.info(f"Training complete for {run_id}. New Task (NT) accuracy: {nt:.2f}%")

                # Plot per-run loss/reward curves
                logger.info("Plotting training curves...")
                plot_grpo_training_curves(checkpoint_dir, run_id, dataset_name)

                # Save final model
                save_path = f"./results_rl/{run_id}/model"
                os.makedirs(save_path, exist_ok=True)
                logger.info(f"Saving fine-tuned model to: {save_path}")
                rl_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                logger.info("Model and tokenizer saved.")

                # ── KL divergence (paper method) ──────────────
                logger.info(f"  [Run {completed_runs}/{total_runs} | Phase 2/3] Computing forward KL divergence (samples={KL_SAMPLES})...")
                kl_device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"KL computation device: {kl_device}")
                if kl_device == "cuda":
                    torch.cuda.empty_cache()
                    logger.info("Moving base model to CUDA for KL computation...")
                    base_model = base_model.to("cuda")

                kl_div = compute_forward_kl(
                    rl_model,           # fine-tuned model
                    base_model,         # base/reference model
                    eval_dataset,       # held-out eval set (same split used for NT)
                    tokenizer,
                    num_samples=KL_SAMPLES,
                    response_only=True, # KL only on response tokens (same as main_v2 pipeline)
                )

                logger.info(f"Forward KL computed: {kl_div:.4f}")
                if kl_device == "cuda":
                    logger.info("Moving base model back to CPU...")
                    base_model = base_model.to("cpu")
                    torch.cuda.empty_cache()

                # ── Prior task benchmarks ─────────────────────
                logger.info(f"  [Run {completed_runs}/{total_runs} | Phase 3/3] Evaluating Prior Task (PT) benchmarks (limit={LIMIT_PER_BENCHMARK})...")
                prior_scores = evaluate_benchmarks(
                    rl_model, tokenizer,
                    limit=LIMIT_PER_BENCHMARK,
                    use_extended=False,
                )
                pt_avg = float(prior_scores.get("average", 0.0)) * 100.0

                # ── Save results ──────────────────────────────
                results.append({
                    "lr": lr,
                    "num_iterations": mu,
                    "scheduler": scheduler_type,
                    "NT": nt,
                    "PT": pt_avg,
                    "kl_divergence": kl_div,
                    "model_path": save_path,
                })

                logger.info(f"Run {run_id} complete — NT={nt:.2f}%, PT={pt_avg:.2f}%, KL={kl_div:.4f}")
                logger.info(f"Results saved to: {results_file}")

                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

                logger.info("Freeing GPU memory...")
                del rl_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("GPU memory freed.")
                pbar.update(1)

    pbar.close()
    # Cleanup
    logger.info("Cleaning up base model from memory...")
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleanup complete.")

    # ── Generate summary plots ────────────────────────────
    logger.info("Generating summary plots for all runs...")
    plot_grpo_results(results, dataset_name)

    logger.info("=" * 70)
    logger.info(f"ALL GRPO RUNS COMPLETE — {len(results)} results saved to {results_file}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()