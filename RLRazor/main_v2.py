"""
main_v2.py - RL's Razor experiment using trainingv1 implementations

Usage:
    python main_v2.py --mode quick --dataset math
    python main_v2.py --mode minimal --dataset science
    python main_v2.py --mode full --dataset tool

This version uses the corrected implementations from trainingv1/:
- Custom Dr.GRPO with fixed log probability calculation
- Domain-specific reward functions (math/science/tool)
- Fixed answer correctness checking
- All critical bugs from code review addressed

Datasets:
- Math Reasoning: Qwen 2.5 3B + Open-Reasoner-Zero
- Science Q&A: Qwen 2.5 3B + SciKnowEval Chemistry
- Tool Use: Qwen 2.5 3B + ToolAlpaca
"""

import os
import sys
import numpy as np
import argparse
from transformers import AutoTokenizer

# Force unbuffered output so prints appear immediately
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("="*70, flush=True)
print("STARTING RL'S RAZOR REPLICATION (trainingv1)", flush=True)
print("="*70, flush=True)
print("Importing modules...", flush=True)

from config.CONFIG import MODEL_NAME
from models.load_model import check_device
from data.load_data import load_dataset_byname
from training.experiment_v2 import run_full_experiment  # Use experiment_v2
from visualization.visualization import plot_pareto_frontier, plot_results, plot_NT_PT

print("All modules imported successfully", flush=True)


def mean_ignore_none(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None

def main():
    parser = argparse.ArgumentParser(description="RL's Razor Replication (trainingv1)")
    parser.add_argument(
        '--mode',
        type=str,
        default='minimal',
        choices=['quick', 'minimal', 'full'],
        help='Configuration mode (default: minimal)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='math',
        choices=['math', 'science', 'tool'],
        help='Dataset to use (default: math)'
    )

    args = parser.parse_args()
    print("\n" + "="*70, flush=True)
    print("RL'S RAZOR REPLICATION (trainingv1)", flush=True)
    print("="*70, flush=True)
    print("\nConfiguration:", flush=True)
    print(f"  Model: {MODEL_NAME}", flush=True)
    print(f"  Mode: {args.mode}", flush=True)
    print(f"  Dataset: {args.dataset}", flush=True)
    print(f"  Training: trainingv1 (fixed implementations)", flush=True)
    print(f"  Hyperparameters: Exactly from Table 2", flush=True)
    print("="*70 + "\n", flush=True)

    # Check device
    print("Checking device...", flush=True)
    device = check_device()

    # Load tokenizer
    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load selected dataset
    dataset_name = args.dataset
    print(f"\nLoading {dataset_name} dataset...", flush=True)
    dataset = load_dataset_byname(dataset_name)

    print("\n" + "="*70, flush=True)
    print(f"SELECTED DATASET: {dataset_name.upper()}", flush=True)
    print("="*70, flush=True)
    print(f"Dataset size: {len(dataset)} examples", flush=True)
    print("="*70, flush=True)

    # Run experiment using trainingv1 implementations
    print(f"\nStarting experiment on {dataset_name} dataset (trainingv1)...", flush=True)

    results = run_full_experiment(dataset, tokenizer, dataset_name=dataset_name, config_mode=args.mode)

    # Create pareto frontier
    plot_pareto_frontier(results, dataset_name)

    # Create visualizations
    plot_results(results)

    # Create plot for NT vs PT
    plot_NT_PT(results)

    print("\n" + "="*70, flush=True)
    print(" EXPERIMENT COMPLETE ", flush=True)
    print("="*70, flush=True)
    print("\nFinal Summary:", flush=True)
    print("-" * 70, flush=True)
    print(f"{'Metric':<30} {'RL (Dr.GRPO)':<20} {'SFT':<20}", flush=True)
    print("-" * 70, flush=True)

    rl_avg_prior = mean_ignore_none([r.get('PT') for r in results.get('rl', [])])
    sft_avg_prior = mean_ignore_none([r.get('PT') for r in results.get('sft', [])])
    rl_avg_kl = mean_ignore_none([r.get('kl_divergence') for r in results.get('rl', [])])
    sft_avg_kl = mean_ignore_none([r.get('kl_divergence') for r in results.get('sft', [])])

    if rl_avg_prior is not None and sft_avg_prior is not None:
        print(f"{'Average Prior Task Score (%)':<30} {rl_avg_prior:<20.4f} {sft_avg_prior:<20.4f}", flush=True)
    else:
        print(f"{'Average Prior Task Score (%)':<30} {'N/A':<20} {'N/A':<20}", flush=True)

    if rl_avg_kl is not None and sft_avg_kl is not None:
        print(f"{'Average KL Divergence':<30} {rl_avg_kl:<20.4f} {sft_avg_kl:<20.4f}", flush=True)
    else:
        print(f"{'Average KL Divergence':<30} {'N/A':<20} {'N/A':<20}", flush=True)
    print("-" * 70, flush=True)

    # Determine winner
    if rl_avg_prior is not None and sft_avg_prior is not None:
        if rl_avg_prior > sft_avg_prior:
            print("\n✓ RL (Dr.GRPO) achieved higher prior task performance!", flush=True)
        elif sft_avg_prior > rl_avg_prior:
            print("\n✓ SFT achieved higher prior task performance!", flush=True)
        else:
            print("\n✓ RL and SFT achieved equal prior task performance!", flush=True)
    else:
        print("\n✗ Prior task performance (PT) not available in results.", flush=True)

    if rl_avg_kl is not None and sft_avg_kl is not None:
        if rl_avg_kl < sft_avg_kl:
            print("✓ RL (Dr.GRPO) has lower KL divergence (less forgetting)!", flush=True)
        elif sft_avg_kl < rl_avg_kl:
            print("✓ SFT has lower KL divergence (less forgetting)!", flush=True)
        else:
            print("✓ RL and SFT have equal KL divergence!", flush=True)
    else:
        print("✗ KL divergence not available in results.", flush=True)

    print("\n" + "="*70, flush=True)
    print(" Output Files:", flush=True)
    print(f"  pareto_frontier_{dataset_name}.png - Main Pareto frontier plot")
    print(f"  results_{dataset_name}_{args.mode}_v2.json - Full experiment results (trainingv1)", flush=True)
    print(f"  kl_vs_forgetting.png - KL divergence vs performance plot", flush=True)
    print(f"  sft_vs_rl_comparison.png - Method comparison chart", flush=True)
    print("="*70 + "\n", flush=True)


if __name__ == "__main__":
    main()
