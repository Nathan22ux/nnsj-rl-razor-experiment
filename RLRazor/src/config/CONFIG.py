"""
CONFIG.py - RL's Razor Paper Replication Configuration
Paper: "RL's Razor: Why Online Reinforcement Learning Forgets Less" (arXiv:2509.04259)

This configuration file contains the EXACT hyperparameters used in the paper for full replication.

Key Paper Details:
- Model: Qwen 2.5 3B-Instruct (also tested on 7B, 14B)
- Tasks: Math Reasoning, Science Q&A, Tool Use
- Method Comparison: SFT vs RL (GRPO)
- Hyperparameter Sweep: 15 learning rates logarithmically spaced between 3e-6 and 1e-3
- Training: 1 or 2 epochs
- Schedulers: constant-with-warmup OR cosine-with-warmup
- Evaluation: Hellaswag, TruthfulQA, MMLU, IFEval, Winogrande, HumanEval
"""

import os
import numpy as np

# =============================================================================
# GLOBAL MODEL CONSTANTS (From Paper)
# =============================================================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Paper also tested these models
ALTERNATIVE_MODELS = {
    'qwen_3b': 'Qwen/Qwen2.5-3B-Instruct',     # Primary model
    'qwen_7b': 'Qwen/Qwen2.5-7B-Instruct',     # Paper scaling experiment
    'qwen_14b': 'Qwen/Qwen2.5-14B-Instruct',   # Paper scaling experiment
    'openvla_7b': 'openvla/openvla-7b',        # For robotics tasks
}

# =============================================================================
# PAPER HYPERPARAMETERS - EXACT VALUES
# =============================================================================

# Learning Rate Sweep (Paper: "15 learning rates logarithmically spaced between 3e-6 and 1e-3")
def get_log_spaced_lrs(n=15):
    """Generate logarithmically spaced learning rates as in the paper"""
    return np.logspace(np.log10(3e-6), np.log10(1e-3), n).tolist()

PAPER_LEARNING_RATES = get_log_spaced_lrs(15)

# Simplified LR grids for different modes
FULL_LR_SWEEP = PAPER_LEARNING_RATES  # All 15 LRs from paper
MINIMAL_LR_SWEEP = [1e-5, 3e-5, 5e-5, 7e-5, 9e-5]  # 6 representative points
QUICK_LR_SWEEP = [1e-5, 3e-5, 1e-4]  # 3 points for quick testing

# Epochs (Paper: "training for 1 or 2 epochs")
PAPER_EPOCHS = [1, 2]

# LR Schedulers (Paper: "constant-with-warmup or cosine-with-warmup scheduler")
PAPER_SCHEDULERS = ['constant_with_warmup', 'cosine']
WARMUP_STEPS = 50  # Standard warmup

# Batch Sizes (Not explicitly stated in paper abstract/intro, using reasonable values)
# For 3B model on typical GPU, these are practical
SFT_BATCH_SIZES = [4, 8, 16, 32] # [16, 32, 64, 128] Effective batch sizes bc grad accumulation
RL_BATCH_SIZES = [8, 16, 32] # [32, 64, 128] For GRPO rollouts

# =============================================================================
# TRAINING CONSTANTS (Standard across paper experiments)
# =============================================================================

# Optimizer settings
WEIGHT_DECAY = 0.0          # Paper uses 0 weight decay
MAX_GRAD_NORM = 1.0         # Standard gradient clipping
BF16 = True                 # Paper uses bfloat16
OPTIMIZER = 'adamw_torch'   # Standard optimizer

# Gradient accumulation (for memory efficiency)
GRADIENT_ACCUMULATION_STEPS = 4  # Adjust based on GPU memory

# =============================================================================
# RL-SPECIFIC CONSTANTS (GRPO)
# =============================================================================

# GRPO Settings (from paper methodology)
KL_COEFF = 0.0              # Paper: "implicit KL minimization" (no explicit penalty)
GRPO_LOSS_TYPE = 'dr-grpo'  # Direct Reward GRPO
NUM_GENERATIONS = 64        # Group size for GRPO
PROMPTS_PER_GENERATION = 8  # Prompts per rollout

# RL Iterations (Paper tests different numbers)
RL_ITERATIONS = [1, 2]      # μ in paper notation

# =============================================================================
# DATASET CONSTANTS
# =============================================================================

# Dataset sizes (adjust based on actual dataset availability)
MAX_TRAINING_SAMPLES = 3000   # Maximum samples to use for training
EVALUATION_SAMPLES = 500      # Samples for NT evaluation
KL_SAMPLES = 200             # Samples for KL divergence computation

# Target NT (New Task) accuracy
# Paper targets vary by task: Math ~75%, Science ~70%, Tool ~75%
# Using 70.0 as default for backward compatibility
TARGET_NT = 00.0  # Default target for all tasks

# Task-specific targets (use these if you want different targets per task)
TARGET_NT_BY_TASK = {
    'math': 75.0,
    'science': 70.0,
    'tool': 75.0,
}

# =============================================================================
# EVALUATION CONSTANTS (From Paper)
# =============================================================================

# Paper evaluates on these benchmarks for Prior Task (PT) performance
BENCHMARKS = [
    'hellaswag',        # Paper benchmark
    'truthfulqa_mc2',   # Paper benchmark
    'mmlu',             # Paper benchmark
    'winogrande',       # Paper benchmark
    'ifeval',           # Paper benchmark
]

# Extended with HumanEval for code capabilities
EXTENDED_BENCHMARKS = BENCHMARKS + [
    'humaneval',
    'arc_challenge',
    'arc_easy',
    'gsm8k',
]

# Evaluation settings
LIMIT_PER_BENCHMARK = 100      # Samples per benchmark (for speed)
NUM_FEWSHOT = 0                # Zero-shot evaluation
HUMAN_EVAL_LIMIT = 50          # HumanEval samples
HUMAN_EVAL_TEMPERATURE = 0.2   # Temperature for code generation

# =============================================================================
# CONFIGURATION MODES
# =============================================================================

def get_paper_exact_config():
    """
    EXACT configuration from the paper for full replication.

    This will create 15 * 2 * 2 = 60 SFT runs per scheduler (120 total)
    And similar for RL.

    Warning: This is computationally expensive!
    """
    return {
        'sft': {
            'learning_rates': FULL_LR_SWEEP,  # 15 LRs
            'batch_sizes': [32],              # Use one batch size, sweep LR
            'epochs': PAPER_EPOCHS,           # [1, 2]
            'schedulers': PAPER_SCHEDULERS,   # constant_with_warmup, cosine
            'lr_scheduler': 'constant_with_warmup',  # Default
            'warmup_steps': WARMUP_STEPS,
            'max_grad_norm': MAX_GRAD_NORM,
            'weight_decay': WEIGHT_DECAY,
            'bf16': BF16,
            'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        },
        'rl': {
            'learning_rates': FULL_LR_SWEEP,  # 15 LRs
            'batch_sizes': [64],              # Use one batch size, sweep LR
            'num_iterations': RL_ITERATIONS,  # [1, 2]
            'loss_type': GRPO_LOSS_TYPE,
            'kl_coeff': KL_COEFF,
            'num_generations': NUM_GENERATIONS,
            'prompts_per_generation': PROMPTS_PER_GENERATION,
            'lr_scheduler': 'constant_with_warmup',
            'warmup_steps': WARMUP_STEPS,
            'max_grad_norm': MAX_GRAD_NORM,
            'weight_decay': WEIGHT_DECAY,
            'bf16': BF16,
            'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        },
        'data': {
            'max_samples': MAX_TRAINING_SAMPLES,
            'eval_samples': EVALUATION_SAMPLES,
            'kl_samples': KL_SAMPLES,
            'target_nt': TARGET_NT,
        }
    }


# FULL SWEEP (Paper replication with all hyperparameters)
FULL_SWEEP_CONFIG = {
    'sft': {
        'learning_rates': [1e-5, 3e-5, 5e-5, 7e-5, 9e-5], # FULL_LR_SWEEP,  # All 15 LRs
        'batch_sizes': SFT_BATCH_SIZES,   # [16, 32, 64]
        'epochs': PAPER_EPOCHS,           # [1, 2]
        'lr_scheduler': 'constant_with_warmup',
        'warmup_steps': WARMUP_STEPS,
        'max_grad_norm': MAX_GRAD_NORM,
        'weight_decay': WEIGHT_DECAY,
        'bf16': BF16,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
    },
    'rl': {
        'learning_rates': [1e-5, 2e-5, 3e-5, 4e-5, 5e-5], # FULL_LR_SWEEP,  # All 15 LRs
        'batch_sizes': RL_BATCH_SIZES,    # [32, 64, 128]
        'num_iterations': RL_ITERATIONS,   # [1, 2]
        'loss_type': GRPO_LOSS_TYPE,
        'kl_coeff': KL_COEFF,
        'num_generations': NUM_GENERATIONS,
        'prompts_per_generation': PROMPTS_PER_GENERATION,
        'lr_scheduler': 'constant_with_warmup',
        'warmup_steps': WARMUP_STEPS,
        'max_grad_norm': MAX_GRAD_NORM,
        'weight_decay': WEIGHT_DECAY,
        'bf16': BF16,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
    },
    'data': {
        'max_samples': MAX_TRAINING_SAMPLES,
        'eval_samples': EVALUATION_SAMPLES,
        'kl_samples': KL_SAMPLES,
        'target_nt': TARGET_NT,  # Simple float for backward compatibility
    }
}

# MINIMAL SWEEP (Budget-conscious replication)
MINIMAL_SWEEP_CONFIG = {
    'sft': {
        'learning_rates': MINIMAL_LR_SWEEP,  # 6 representative LRs
        'batch_sizes': [32],                  # One batch size
        'epochs': PAPER_EPOCHS,               # [1, 2]
        'lr_scheduler': 'constant_with_warmup',
        'warmup_steps': WARMUP_STEPS,
        'max_grad_norm': MAX_GRAD_NORM,
        'weight_decay': WEIGHT_DECAY,
        'bf16': BF16,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
    },
    'rl': {
        'learning_rates': MINIMAL_LR_SWEEP,  # 6 representative LRs
        'batch_sizes': [64],                  # One batch size
        'num_iterations': [2],                # Just 2 iterations
        'loss_type': GRPO_LOSS_TYPE,
        'kl_coeff': KL_COEFF,
        'num_generations': NUM_GENERATIONS,
        'prompts_per_generation': PROMPTS_PER_GENERATION,
        'lr_scheduler': 'constant_with_warmup',
        'warmup_steps': WARMUP_STEPS,
        'max_grad_norm': MAX_GRAD_NORM,
        'weight_decay': WEIGHT_DECAY,
        'bf16': BF16,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
    },
    'data': {
        'max_samples': 2200,  # Use all available training data
        'eval_samples': 500,
        'kl_samples': 200,
        'target_nt': TARGET_NT,
    }
}

# QUICK TEST (For debugging and validation)
QUICK_TEST_CONFIG = {
    'sft': {
        'learning_rates': [3e-5],  # Single LR
        'batch_sizes': [16],       # Small batch
        'epochs': [1],             # Single epoch
        'lr_scheduler': 'constant_with_warmup',
        'warmup_steps': WARMUP_STEPS,
        'max_grad_norm': MAX_GRAD_NORM,
        'weight_decay': WEIGHT_DECAY,
        'bf16': BF16,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
    },
    'rl': {
        'learning_rates': [1e-4],  # Single LR
        'batch_sizes': [32],       # Small batch
        'num_iterations': [1],     # Single iteration
        'loss_type': GRPO_LOSS_TYPE,
        'kl_coeff': KL_COEFF,
        'num_generations': 16,     # Reduced for speed
        'prompts_per_generation': 4,  # Reduced for speed
        'lr_scheduler': 'constant_with_warmup',
        'warmup_steps': WARMUP_STEPS,
        'max_grad_norm': MAX_GRAD_NORM,
        'weight_decay': WEIGHT_DECAY,
        'bf16': BF16,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
    },
    'data': {
        'max_samples': 500,   # Small subset
        'eval_samples': 100,
        'kl_samples': 50,
        'target_nt': TARGET_NT,
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config(mode='default'):
    """
    Get configuration based on mode.

    Args:
        mode: Configuration mode
            'quick' - Quick test with minimal settings (~2 runs, <1 GPU hour)
            'minimal' - Minimal sweep for budget replication (~24 runs, ~50 GPU hours)
            'default' - Same as minimal (for backward compatibility)
            'full' - Complete paper replication (~180 runs, ~400 GPU hours)
            'paper_exact' - Exact paper config with all schedulers (~240+ runs)
            'mechanistic' - For mechanistic interpretability work

    Returns:
        tuple: (sft_config, rl_config, data_config) OR mechanistic_config dict
    """
    if mode == 'paper_exact':
        config = get_paper_exact_config()
    elif mode == 'full':
        config = FULL_SWEEP_CONFIG.copy()
    elif mode == 'minimal' or mode == 'default':
        config = MINIMAL_SWEEP_CONFIG.copy()
    elif mode == 'quick':
        config = QUICK_TEST_CONFIG.copy()
    elif mode == 'mechanistic':
        return MECHANISTIC_CONFIG
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: quick, minimal, default, full, paper_exact, mechanistic")

    # Deep copy to avoid mutation
    sft_config = config['sft'].copy()
    rl_config = config['rl'].copy()
    data_config = config['data'].copy()

    return sft_config, rl_config, data_config


def count_total_runs(mode='minimal'):
    """Calculate total number of model training runs for a given mode."""
    sft_cfg, rl_cfg, _ = get_config(mode)

    # SFT combinations
    sft_runs = (
            len(sft_cfg['learning_rates']) *
            len(sft_cfg['batch_sizes']) *
            len(sft_cfg['epochs'])
    )

    # RL combinations
    rl_runs = (
            len(rl_cfg['learning_rates']) *
            len(rl_cfg['batch_sizes']) *
            len(rl_cfg['num_iterations'])
    )

    return {
        'sft_runs': sft_runs,
        'rl_runs': rl_runs,
        'total_runs': sft_runs + rl_runs,
    }


def estimate_compute_hours(mode='minimal', model_size='3B'):
    """Estimate total GPU hours needed based on mode and model size."""
    runs = count_total_runs(mode)

    # Rough estimates (hours per run on A100 80GB)
    hours_per_run = {
        '3B': {'sft': 1.5, 'rl': 3.0},
        '7B': {'sft': 3.0, 'rl': 6.0},
        '14B': {'sft': 6.0, 'rl': 12.0},
    }

    size_key = model_size if model_size in hours_per_run else '3B'

    sft_hours = runs['sft_runs'] * hours_per_run[size_key]['sft']
    rl_hours = runs['rl_runs'] * hours_per_run[size_key]['rl']
    total_hours = sft_hours + rl_hours

    # Cost estimate (AWS p4d.24xlarge ~$32/hour, but using 1 GPU ~$4/hour)
    cost_estimate = total_hours * 4

    return {
        'sft_hours': sft_hours,
        'rl_hours': rl_hours,
        'total_hours': total_hours,
        'estimated_cost_usd': cost_estimate,
    }


def print_config_summary(config_mode='default'):
    """Print a pretty summary of the selected configuration."""
    print(f"\n{'='*80}")
    print(f"RL'S RAZOR REPLICATION - CONFIGURATION SUMMARY")
    print(f"{'='*80}")
    print(f"\nPaper: arXiv:2509.04259 (Sep 2025)")
    print(f"Mode: {config_mode.upper()}")

    if config_mode == 'mechanistic':
        print("\nMode: MECHANISTIC INTERPRETABILITY")
        print("See MECHANISTIC_CONFIG dictionary for details.")
        return

    runs = count_total_runs(config_mode)
    compute = estimate_compute_hours(config_mode)
    sft_cfg, rl_cfg, data_cfg = get_config(config_mode)

    print(f"\n{'─'*80}")
    print(f"MODEL & COMPUTE")
    print(f"{'─'*80}")
    print(f"Base Model: {MODEL_NAME}")
    print(f"Total Training Runs: {runs['total_runs']} ({runs['sft_runs']} SFT + {runs['rl_runs']} RL)")
    print(f"Estimated GPU Hours: {compute['total_hours']:.1f} ({compute['sft_hours']:.1f} SFT + {compute['rl_hours']:.1f} RL)")
    print(f"Estimated Cost (A100): ${compute['estimated_cost_usd']:.0f}")

    print(f"\n{'─'*80}")
    print(f"SFT CONFIGURATION")
    print(f"{'─'*80}")
    print(f"Learning Rates: {len(sft_cfg['learning_rates'])} values")
    print(f"  Range: {min(sft_cfg['learning_rates']):.2e} to {max(sft_cfg['learning_rates']):.2e}")
    print(f"Batch Sizes: {sft_cfg['batch_sizes']}")
    print(f"Epochs: {sft_cfg['epochs']}")
    print(f"LR Scheduler: {sft_cfg['lr_scheduler']}")
    print(f"Weight Decay: {sft_cfg['weight_decay']}")

    print(f"\n{'─'*80}")
    print(f"RL CONFIGURATION (GRPO)")
    print(f"{'─'*80}")
    print(f"Learning Rates: {len(rl_cfg['learning_rates'])} values")
    print(f"  Range: {min(rl_cfg['learning_rates']):.2e} to {max(rl_cfg['learning_rates']):.2e}")
    print(f"Batch Sizes: {rl_cfg['batch_sizes']}")
    print(f"Iterations (μ): {rl_cfg['num_iterations']}")
    print(f"KL Coefficient: {rl_cfg['kl_coeff']} (implicit minimization)")
    print(f"Group Size: {rl_cfg['num_generations']}")
    print(f"Prompts/Gen: {rl_cfg['prompts_per_generation']}")

    print(f"\n{'─'*80}")
    print(f"DATA CONFIGURATION")
    print(f"{'─'*80}")
    print(f"Max Training Samples: {data_cfg['max_samples']}")
    print(f"Eval Samples (NT): {data_cfg['eval_samples']}")
    print(f"KL Samples: {data_cfg['kl_samples']}")
    print(f"Target NT Accuracy: {data_cfg['target_nt']:.1f}%")

    print(f"\n{'─'*80}")
    print(f"EVALUATION BENCHMARKS (PT)")
    print(f"{'─'*80}")
    print(f"Benchmarks: {', '.join(BENCHMARKS)}")
    print(f"Samples per benchmark: {LIMIT_PER_BENCHMARK}")

    print(f"\n{'='*80}\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("RL'S RAZOR - PAPER REPLICATION CONFIGURATIONS")
    print("Paper: arXiv:2509.04259 (September 2025)")
    print("="*80)

    print("\nAvailable configuration modes:")
    print("  'quick'       - Fast testing (2 runs total, ~1 GPU hour)")
    print("  'minimal'     - Budget replication (24 runs, ~50 GPU hours)")
    print("  'default'     - Same as minimal (backward compatible)")
    print("  'full'        - Complete sweep (180 runs, ~400 GPU hours)")
    print("  'paper_exact' - Exact paper config (240+ runs, ~600 GPU hours)")
    print("  'mechanistic' - For mechanistic interpretability work")

    # Print summaries for common modes
    for mode in ['quick', 'minimal', 'full']:
        print_config_summary(mode)