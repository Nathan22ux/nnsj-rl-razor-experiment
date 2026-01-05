"""
Usage:
    python main.py --mode quick --dataset math
    python main.py --mode minimal --dataset math
    python main.py --mode full --dataset science

Datasets:
- Math Reasoning: Qwen 2.5 3B + Open-Reasoner-Zero
- Science Q&A: Qwen 2.5 3B + SciKnowEval Chemistry
- Tool Use: Qwen 2.5 3B + ToolAlpaca
"""

import os
import sys
import numpy as np
import argparse

# Force unbuffered output so prints appear immediately
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("="*70, flush=True)
print("STARTING RL'S RAZOR REPLICATION", flush=True)
print("="*70, flush=True)
print("Importing modules...", flush=True)

from config.CONFIG import MODEL_NAME
from models.load_model import check_device, load_model_and_tokenizer
from data.load_data import load_dataset_byname
from training.experiment import run_full_experiment
from visualization.visualization import plot_pareto_frontier, plot_results, plot_NT_PT

print("All modules imported successfully", flush=True)


def main():
    parser = argparse.ArgumentParser(description="RL's Razor Replication")
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
    print("RL'S RAZOR REPLICATION", flush=True)
    print("="*70, flush=True)
    print("\nConfiguration:", flush=True)
    print(f"  Model: {MODEL_NAME}", flush=True)
    print(f"  Mode: {args.mode}", flush=True)  # Show selected mode
    print(f"  Dataset: {args.dataset}", flush=True)  # Show selected dataset
    print(f"  Hyperparameters: Exactly from Table 2", flush=True)
    print("="*70 + "\n", flush=True)
    
    # Check device
    print("Checking device...", flush=True)
    device = check_device()
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...", flush=True)
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # Load only the selected dataset
    dataset_name = args.dataset
    print(f"\nLoading {dataset_name} dataset...", flush=True)
    dataset = load_dataset_byname(dataset_name)

    print("\n" + "="*70, flush=True)
    print(f"SELECTED DATASET: {dataset_name.upper()}", flush=True)
    print("="*70, flush=True)
    print(f"Dataset size: {len(dataset)} examples", flush=True)
    print("="*70, flush=True)
    
    # Run experiment on Math dataset
    print(f"\n Starting experiment on {dataset_name} dataset...", flush=True)
    
    results = run_full_experiment(dataset, tokenizer, dataset_name=dataset_name, config_mode=args.mode)

    #create pareto frontier
    plot_pareto_frontier(results, dataset_name) 

    # Create visualizations
    plot_results(results)
    
    # Create plot for NT vs PT
    plot_NT_PT(results)

    print("\n" + "="*70, flush=True)
    print(" EXPERIMENT COMPLETE ", flush=True)
    print("="*70, flush=True)
    print("\n Final Summary:", flush=True)
    print("-" * 70, flush=True)
    print(f"{'Metric':<30} {'RL (GRPO)':<20} {'SFT':<20}", flush=True)
    print("-" * 70, flush=True)
    
    rl_avg_prior = np.mean([r['PT'] for r in results['rl']])
    sft_avg_prior = np.mean([r['PT'] for r in results['sft']])
    rl_avg_kl = np.mean([r['kl_divergence'] for r in results['rl']])
    sft_avg_kl = np.mean([r['kl_divergence'] for r in results['sft']])
    
    print(f"{'Average Prior Task Score':<30} {rl_avg_prior:<20.4f} {sft_avg_prior:<20.4f}", flush=True)
    print(f"{'Average KL Divergence':<30} {rl_avg_kl:<20.4f} {sft_avg_kl:<20.4f}", flush=True)
    print("-" * 70, flush=True)
    
    # Determine winner
    if rl_avg_prior > sft_avg_prior:
        print("\n RL (GRPO) achieved higher prior task performance!", flush=True)
    elif sft_avg_prior > rl_avg_prior:
        print("\n SFT achieved higher prior task performance!", flush=True)
    else:
        print("\n RL and SFT achieved equal prior task performance!", flush=True)
    
    if rl_avg_kl < sft_avg_kl:
        print(" RL (GRPO) has lower KL divergence (less forgetting)!", flush=True)
    elif sft_avg_kl < rl_avg_kl:
        print(" SFT has lower KL divergence (less forgetting)!", flush=True)
    else:
        print(" RL and SFT have equal KL divergence!", flush=True)
    
    print("\n" + "="*70, flush=True)
    print(" Output Files:", flush=True)
    print(f"  pareto_frontier_{dataset_name}.png - Main Pareto frontier plot")
    print(f"  results_{dataset_name}.json - Full experiment results", flush=True)
    print(f"  kl_vs_forgetting.png - KL divergence vs performance plot", flush=True)
    print(f"  sft_vs_rl_comparison.png - Method comparison chart", flush=True)
    print("="*70 + "\n", flush=True)


if __name__ == "__main__":
    main()