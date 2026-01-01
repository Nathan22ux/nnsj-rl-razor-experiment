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

import argparse
import logging
import os
import sys

import numpy as np

# Allow code evaluation for metrics
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from config import MODEL_NAME
from load_data import load_datasets
from load_model import check_device, load_model_and_tokenizer
from logger import configure_logging, get_logger
from visualization import plot_NT_PT, plot_pareto_frontier, plot_results
from experiment import run_full_experiment

# Configure logging
configure_logging(log_dir="logs", level=logging.INFO)
logger = get_logger(__name__)


def main():
    """Main entry point for the RL's Razor replication experiment."""
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

    logger.info("=" * 70)
    logger.info("RL'S RAZOR REPLICATION STARTING")
    logger.info("=" * 70)
    logger.info(f"Configuration: Model={MODEL_NAME}, Mode={args.mode}, Dataset={args.dataset}")
    logger.info("Hyperparameters: Exactly from Table 2")

    # Check device
    device = check_device()

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # Load datasets
    logger.info("Loading datasets...")
    datasets = load_datasets()

    # Select dataset for experiment
    dataset_name = args.dataset
    dataset = datasets[dataset_name]

    logger.info("=" * 70)
    logger.info(f"SELECTED DATASET: {dataset_name.upper()}")
    logger.info("=" * 70)
    logger.info(f"Dataset size: {len(dataset)} examples")

    # Run experiment
    logger.info(f"Starting experiment on {dataset_name} dataset...")
    results = run_full_experiment(
        dataset, tokenizer, dataset_name=dataset_name, config_mode=args.mode
    )

    # Create visualizations
    logger.info("Generating visualizations...")
    plot_pareto_frontier(results, dataset_name)
    plot_results(results)
    plot_NT_PT(results)

    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)

    # Print summary
    logger.info("Final Summary:")
    logger.info("-" * 70)
    logger.info(f"{'Metric':<30} {'RL (GRPO)':<20} {'SFT':<20}")
    logger.info("-" * 70)

    rl_avg_prior = np.mean([r['PT'] for r in results['rl']])
    sft_avg_prior = np.mean([r['PT'] for r in results['sft']])
    rl_avg_kl = np.mean([r['kl_divergence'] for r in results['rl']])
    sft_avg_kl = np.mean([r['kl_divergence'] for r in results['sft']])

    logger.info(
        f"{'Average Prior Task Score':<30} {rl_avg_prior:<20.4f} {sft_avg_prior:<20.4f}"
    )
    logger.info(
        f"{'Average KL Divergence':<30} {rl_avg_kl:<20.4f} {sft_avg_kl:<20.4f}"
    )
    logger.info("-" * 70)

    # Determine winner
    if rl_avg_prior > sft_avg_prior:
        logger.info("RL (GRPO) achieved higher prior task performance!")
    elif sft_avg_prior > rl_avg_prior:
        logger.info("SFT achieved higher prior task performance!")
    else:
        logger.info("RL and SFT achieved equal prior task performance!")

    if rl_avg_kl < sft_avg_kl:
        logger.info("RL (GRPO) has lower KL divergence (less forgetting)!")
    elif sft_avg_kl < rl_avg_kl:
        logger.info("SFT has lower KL divergence (less forgetting)!")
    else:
        logger.info("RL and SFT have equal KL divergence!")

    logger.info("=" * 70)
    logger.info("Output Files:")
    logger.info(f"  pareto_frontier_{dataset_name}.png - Main Pareto frontier plot")
    logger.info(f"  results_{dataset_name}.json - Full experiment results")
    logger.info(f"  kl_vs_forgetting.png - KL divergence vs performance plot")
    logger.info(f"  sft_vs_rl_comparison.png - Method comparison chart")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()