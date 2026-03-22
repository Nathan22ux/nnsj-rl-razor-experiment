"""
run_sft.py — Standalone SFT training for RL's Razor replication.

Paper hyperparameters (Table from paper):
- Model: Qwen2.5 3B-Instruct
- LR: {3e-5, 5e-5}
- Batch Size: 32 (effective = 8 × 4 grad_accum)
- Epochs: {1, 2}
- Scheduler: {constant_with_warmup, cosine}
- Optimizer: AdamW, weight_decay=0
- Warmup: 50 steps
- bf16: True
- Max Grad Norm: 1.0

Usage:
    python run_sft.py --dataset math
    python run_sft.py --dataset science
    python run_sft.py --dataset tool
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
from trainingv1.train_sft_baseline import train_sft_baseline
from logger import get_logger

logger = get_logger(__name__)


# Paper-exact SFT hyperparameters

LEARNING_RATES = [3e-5, 5e-5]
BATCH_SIZE = 8                     # per-device (× 4 grad_accum = 32 effective)
EPOCHS = [1, 2]
SCHEDULERS = ["constant_with_warmup", "cosine"]
MAX_SAMPLES = 2200                 # Paper appendix
KL_SAMPLES = 200
EVAL_SAMPLES = 500



# Plotting helpers


def plot_sft_results(results, dataset_name, output_dir="results"):
    """Generate plots from completed SFT runs."""
    if not results:
        logger.info("No results to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    labels = []
    nt_scores = []
    pt_scores = []
    kl_vals = []

    for r in results:
        lbl = f"lr={r['lr']:.0e}\nep={r['epochs']}\n{r['lr_scheduler'][:4]}"
        labels.append(lbl)
        nt_scores.append(r.get("NT", 0) or 0)
        pt_scores.append(r.get("PT", 0) or 0)
        kl_vals.append(r.get("kl_divergence", 0) or 0)

    x = np.arange(len(labels))
    w = 0.35

    # ── 1. NT vs PT grouped bar chart ─────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    ax.bar(x - w / 2, nt_scores, w, label="NT (New Task %)", color="#4C72B0")
    ax.bar(x + w / 2, pt_scores, w, label="PT (Prior Task %)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"SFT — NT vs PT Accuracy ({dataset_name})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"sft_{dataset_name}_nt_pt.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved NT/PT chart → {path}")

    # ── 2. KL divergence bar chart ────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    bars = ax.bar(x, kl_vals, color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Forward KL(π₀ ∥ π)")
    ax.set_title(f"SFT — Forward KL Divergence ({dataset_name})")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, kl_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    path = os.path.join(output_dir, f"sft_{dataset_name}_kl.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved KL chart → {path}")

    # ── 3. Forgetting Law: NT vs KL scatter ───────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(kl_vals, pt_scores, s=80, c="#C44E52", edgecolors="black", zorder=3)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl.replace("\n", " "), (kl_vals[i], pt_scores[i]),
                    fontsize=6, textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Forward KL(π₀ ∥ π)")
    ax.set_ylabel("PT Accuracy (%)")
    ax.set_title(f"SFT — Forgetting Law ({dataset_name})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"sft_{dataset_name}_forgetting_law.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Forgetting Law scatter → {path}")


def plot_training_loss(log_history, run_id, dataset_name, output_dir="results"):
    """Plot training loss curve from TRL trainer log_history."""
    os.makedirs(output_dir, exist_ok=True)

    steps = [entry["step"] for entry in log_history if "loss" in entry]
    losses = [entry["loss"] for entry in log_history if "loss" in entry]

    if not steps:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, losses, linewidth=1.2, color="#4C72B0")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"SFT Training Loss — {run_id}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"sft_{dataset_name}_loss_{run_id}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved loss curve → {path}")


def main():
    parser = argparse.ArgumentParser(description="RL's Razor — SFT training")
    parser.add_argument("--dataset", type=str, default="math",
                        choices=["math", "science", "tool"],
                        help="Dataset/domain to train on")
    args = parser.parse_args()

    dataset_name = args.dataset
    logger.info("=" * 70)
    logger.info("RL's Razor — Standalone SFT Training")
    logger.info("=" * 70)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"LRs: {LEARNING_RATES}")
    logger.info(f"Batch size (effective): {BATCH_SIZE * 4}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Schedulers: {SCHEDULERS}")
    logger.info("=" * 70)

    # ── Load dataset ──────────────────────────────────────
    dataset = load_dataset_byname(dataset_name)

    # Train / eval split (deterministic)
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
    results_file = f"results/sft_{dataset_name}.json"
    results = []
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

    # ── Sweep ─────────────────────────────────────────────
    for lr in LEARNING_RATES:
        for epochs in EPOCHS:
            for scheduler in SCHEDULERS:
                effective_bs = BATCH_SIZE * 4
                run_id = f"lr{lr}_bs{effective_bs}_ep{epochs}_{scheduler}"

                # Skip if done
                if any(
                    r.get("lr") == lr
                    and r.get("epochs") == epochs
                    and r.get("lr_scheduler") == scheduler
                    for r in results
                ):
                    logger.info(f"Skipping {run_id} (already done)")
                    continue

                logger.info("=" * 70)
                logger.info(f"SFT RUN: {run_id}")
                logger.info("=" * 70)

                sft_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto", trust_remote_code=True,
                    attn_implementation=attn_impl,
                )

                sft_model, nt = train_sft_baseline(
                    model=sft_model,
                    tokenizer=tokenizer,
                    dataset=train_dataset,
                    learning_rate=lr,
                    batch_size=BATCH_SIZE,
                    epochs=epochs,
                    max_samples=MAX_SAMPLES,
                    eval_dataset=eval_dataset,
                    lr_scheduler_type=scheduler,
                )

                # Plot per-run training loss if trainer state is available
                try:
                    from transformers.trainer_utils import get_last_checkpoint
                    log_dir = f"./results_sft/lr{lr}_bs{BATCH_SIZE * 4}_ep{epochs}_{scheduler}"
                    trainer_state_path = os.path.join(log_dir, "trainer_state.json")
                    if os.path.exists(trainer_state_path):
                        with open(trainer_state_path, "r") as _f:
                            trainer_state = json.load(_f)
                        plot_training_loss(
                            trainer_state.get("log_history", []),
                            run_id, dataset_name,
                        )
                except Exception as e:
                    logger.info(f"Could not plot training loss: {e}")

                # Save checkpoint
                save_path = f"./results_sft/{run_id}/model"
                os.makedirs(save_path, exist_ok=True)
                sft_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

                # ── KL divergence (paper method) ──────────
                logger.info("Computing forward KL...")
                kl_device = "cuda" if torch.cuda.is_available() else "cpu"
                if kl_device == "cuda":
                    base_model.to("cuda")

                kl_div = compute_forward_kl(
                    base_model=base_model,
                    target_model=sft_model,
                    tokenizer=tokenizer,
                    dataset=train_dataset,
                    num_samples=KL_SAMPLES,
                )

                if kl_device == "cuda":
                    base_model.to("cpu")
                    torch.cuda.empty_cache()

                # ── Prior task benchmarks ─────────────────
                logger.info("Evaluating PT benchmarks...")
                prior_scores = evaluate_benchmarks(
                    sft_model, tokenizer,
                    limit=LIMIT_PER_BENCHMARK,
                    use_extended=False,
                )
                pt_avg = float(prior_scores.get("average", 0.0)) * 100.0

                # ── Save results ──────────────────────────
                results.append({
                    "lr": lr,
                    "batch_size": effective_bs,
                    "epochs": epochs,
                    "lr_scheduler": scheduler,
                    "NT": nt,
                    "PT": pt_avg,
                    "kl_divergence": kl_div,
                    "model_path": save_path,
                })

                logger.info(f"NT={nt:.2f}%, PT={pt_avg:.2f}%, KL={kl_div:.4f}")

                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

                del sft_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Cleanup
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Generate summary plots ────────────────────────────
    plot_sft_results(results, dataset_name)

    logger.info("=" * 70)
    logger.info(f"ALL SFT RUNS COMPLETE — {len(results)} results saved to {results_file}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
