"""
run_grpo.py — Standalone Dr.GRPO (RL) training for RL's Razor replication.

Paper hyperparameters (Table from paper):
- Model: Qwen2.5 3B-Instruct
- LR: {3e-5, 5e-5}
- Scheduler: constant_with_warmup
- Optimizer: AdamW, weight_decay=0
- Warmup: 50 steps
- bf16: True
- Max Grad Norm: 1.0
- KL reg: 0 (implicit)
- Group Size: 64
- Prompts per generation: 8
- μ iterations: {1, 2}
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

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.CONFIG import MODEL_NAME, LIMIT_PER_BENCHMARK
from data.load_data import load_dataset_byname
from data.dataset_utils import UnifiedDatasetInterface
from evaluation.evaluation import evaluate_benchmarks
from trainingv1.eval_kl_forward import compute_forward_kl
from trainingv1.train_dr_grpo import train_dr_grpo
from logger import get_logger

logger = get_logger(__name__)


# Paper-exact RL (Dr.GRPO) hyperparameters

LEARNING_RATES = [3e-5, 5e-5]
GROUP_SIZE = 64
PROMPTS_PER_GEN = 8
MU_ITERATIONS = [1, 2]
MAX_SAMPLES = 2200                 # Paper appendix
MAX_COMPLETION_LENGTH = 256
WARMUP_STEPS = 50
KL_SAMPLES = 200



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
        lbl = f"lr={r['lr']:.0e}\nμ={r['num_iterations']}"
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
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Domain: {domain}")
    logger.info(f"LRs: {LEARNING_RATES}")
    logger.info(f"Group size: {GROUP_SIZE}")
    logger.info(f"Prompts/gen: {PROMPTS_PER_GEN}")
    logger.info(f"μ iterations: {MU_ITERATIONS}")
    logger.info("=" * 70)

    # ── Load dataset ──────────────────────────────────────
    dataset = load_dataset_byname(dataset_name)

    # Train / eval split (deterministic, same seed as SFT for fair comparison)
    dataset_size = len(dataset)
    eval_size = min(200, int(dataset_size * 0.1))
    indices = list(range(dataset_size))
    random.seed(42)
    random.shuffle(indices)

    train_dataset = dataset.select(indices[:-eval_size])
    eval_dataset_raw = dataset.select(indices[-eval_size:])
    eval_dataset = UnifiedDatasetInterface.normalize_dataset(eval_dataset_raw)

    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # ── Flash Attention ───────────────────────────────────
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"

    # ── Base model for KL (kept on CPU) ───────────────────
    base_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=base_dtype, device_map="cpu",
        trust_remote_code=True, attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Results ───────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    results_file = f"results/grpo_{dataset_name}.json"
    results = []
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

    # ── Sweep ─────────────────────────────────────────────
    for lr in LEARNING_RATES:
        for mu in MU_ITERATIONS:
            run_id = f"lr{lr}_mu{mu}"

            # Skip if done
            if any(
                r.get("lr") == lr and r.get("num_iterations") == mu
                for r in results
            ):
                logger.info(f"Skipping {run_id} (already done)")
                continue

            logger.info("=" * 70)
            logger.info(f"GRPO RUN: {run_id}")
            logger.info("=" * 70)

            rl_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto", trust_remote_code=True,
                attn_implementation=attn_impl,
            )

            checkpoint_dir = f"./checkpoints/rl/{run_id}"
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
                checkpoint_dir=checkpoint_dir,
            )

            # Plot per-run loss/reward curves
            plot_grpo_training_curves(checkpoint_dir, run_id, dataset_name)

            # Save final model
            save_path = f"./results_rl/{run_id}/model"
            os.makedirs(save_path, exist_ok=True)
            rl_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            # ── KL divergence (paper method) ──────────────
            logger.info("Computing forward KL...")
            kl_device = "cuda" if torch.cuda.is_available() else "cpu"
            if kl_device == "cuda":
                base_model.to("cuda")

            kl_div = compute_forward_kl(
                base_model=base_model,
                target_model=rl_model,
                tokenizer=tokenizer,
                dataset=train_dataset,
                num_samples=KL_SAMPLES,
            )

            if kl_device == "cuda":
                base_model.to("cpu")
                torch.cuda.empty_cache()

            # ── Prior task benchmarks ─────────────────────
            logger.info("Evaluating PT benchmarks...")
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
                "NT": nt,
                "PT": pt_avg,
                "kl_divergence": kl_div,
                "model_path": save_path,
            })

            logger.info(f"NT={nt:.2f}%, PT={pt_avg:.2f}%, KL={kl_div:.4f}")

            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            del rl_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Cleanup
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Generate summary plots ────────────────────────────
    plot_grpo_results(results, dataset_name)

    logger.info("=" * 70)
    logger.info(f"ALL GRPO RUNS COMPLETE — {len(results)} results saved to {results_file}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
