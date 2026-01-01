# Configuration for RL's Razor replication experiment
"""Configuration module for RL's Razor experiment.

This module provides centralized configuration management with support for:
- Different experiment modes (quick, minimal, full)
- Flexible hyperparameter sweeps
- Compute resource estimation
"""

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct" # openai-community/gpt2

# Alternative models for validation
ALTERNATIVE_MODELS = {
    'gpt2': 'gpt2',  # For quick testing
    'qwen_7b': 'Qwen/Qwen2.5-7B-Instruct',
    'qwen_14b': 'Qwen/Qwen2.5-14B-Instruct',
}

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
data_config = {
    'max_samples': 2000,           # Training samples per run (paper uses more)
    'eval_samples': 200,           # Evaluation samples
    'kl_samples': 100,             # Samples for KL computation
}

# =============================================================================
# SFT HYPERPARAMETERS (Exact from Paper Table 2)
# =============================================================================
sft_config = {
    # Complete sweep from paper
    'learning_rates': [1e-5, 3e-5, 5e-5, 7e-5, 9e-5],
    'batch_sizes': [16, 32, 64, 128],
    'epochs': [1, 2],
    
    # Fixed hyperparameters
    'lr_scheduler': 'constant_with_warmup',  # or 'cosine_with_warmup'
    'warmup_steps': 50,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'bf16': True,
    'gradient_accumulation_steps': 4,
}

# =============================================================================
# RL HYPERPARAMETERS (Exact from Paper Table 2)
# =============================================================================
rl_config = {
    # Complete sweep from paper
    'learning_rates': [1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
    'batch_sizes': [32, 64],  # Paper sweeps batch size for RL too
    
    # GRPO specific
    'kl_coef': 0.0,              # Paper uses 0 - implicit KL minimization
    'num_generations': 64,       # Group size
    'prompts_per_generation': 8,
    'num_iterations': [1, 2],    # μ in paper
    
    # Fixed hyperparameters
    'lr_scheduler': 'constant_with_warmup',
    'warmup_steps': 50,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'bf16': True,
    'gradient_accumulation_steps': 4,
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
eval_config = {
    # Core benchmarks (from paper Table 1)
    'benchmarks': [
        'winogrande',
        'hellaswag',
        'mmlu',  # Full MMLU, not subsets
        'truthfulqa_mc2',
    ],
    
    # Extended benchmarks (optional)
    'extended_benchmarks': [
        'winogrande',
        'hellaswag',
        'mmlu',
        'truthfulqa_mc2',
        'arc_challenge',
        'arc_easy',
        'gsm8k',
    ],
    
    # Evaluation settings
    'limit_per_benchmark': 100,    # Samples per benchmark
    'num_fewshot': 0,              # Zero-shot evaluation
    
    # HumanEval settings (optional)
    'humaneval_limit': 50,
    'humaneval_temperature': 0.2,
    
    # IFEval settings (optional)
    'ifeval_limit': 100,
}

# =============================================================================
# QUICK TEST CONFIGURATION
# =============================================================================
# Use these for initial testing before full sweep
quick_test_config = {
    'sft': {
        'learning_rates': [3e-5],      # Single LR
        'batch_sizes': [16],           # Single batch size
        'epochs': [1],                 # Single epoch
    },
    'rl': {
        'learning_rates': [2e-5],      # Single LR
        'batch_sizes': [16],           # Single batch size
        'num_iterations': [1],
    },
    'max_samples': 200,                # Much smaller for testing
    'eval_samples': 50,
    'kl_samples': 30,
}

# =============================================================================
# FULL SWEEP CONFIGURATION
# =============================================================================
# Use this for actual paper replication
full_sweep_config = {
    'sft': {
        'learning_rates': [1e-5, 3e-5, 5e-5, 7e-5, 9e-5],
        'batch_sizes': [16, 32, 64, 128],
        'epochs': [1, 2],
    },
    'rl': {
        'learning_rates': [1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
        'batch_sizes': [32, 64],
        'num_iterations': [1, 2],
    },
    'max_samples': 5000,               # More samples for better results
    'eval_samples': 500,
    'kl_samples': 200,
}

# =============================================================================
# MINIMAL SWEEP (For budget-conscious replication)
# =============================================================================
minimal_sweep_config = {
    'sft': {
        'learning_rates': [1e-5, 3e-5, 5e-5],
        'batch_sizes': [32, 64],
        'epochs': [1],
    },
    'rl': {
        'learning_rates': [1e-5, 3e-5, 5e-5],
        'batch_sizes': [32],
        'num_iterations': [1],
    },
    'max_samples': 1000,
    'eval_samples': 200,
    'kl_samples': 100,
}

# =============================================================================
# MECHANISTIC INTERPRETABILITY CONFIG
# =============================================================================
# For the mechanistic forgetting proposal (different research direction)
mechanistic_config = {
    'models': {
        'primary': 'gpt2',
        'validation': 'meta-llama/Llama-2-7b-hf',
    },
    
    # Single configuration per method (no sweeps needed)
    'sft': {
        'learning_rate': 5e-5,
        'batch_size': 32,
        'epochs': 3,
    },
    
    'rl': {
        'learning_rate': 1e-5,
        'batch_size': 32,
        'epochs': 1,
        'kl_coef': 0.0,
    },
    
    # SFT + KL penalty sweep (Experiment 2)
    'sft_kl_penalties': [0, 0.01, 0.05, 0.1, 0.5],
    
    # Dataset configuration
    'adaptation_dataset': 'tatsu-lab/alpaca',
    'adaptation_size': 45000,
    
    'retention_datasets': {
        'natural_questions': {'size': 3000},
        'induction': {'size': 1000, 'synthetic': True},
        'bigbench': {'tasks': ['navigate', 'logical_deduction', 'causal_judgment']},
    },
    
    # Circuit analysis settings
    'circuit_analysis': {
        'attention_heads': True,
        'activation_patching': True,
        'logit_lens': True,
        'cka_similarity': True,
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_config(mode='default'):
    """
    Get configuration based on mode.
    
    Args:
        mode: Configuration mode
            'quick' - Quick test with minimal settings
            'minimal' - Minimal sweep for budget replication
            'default' - Balanced configuration
            'full' - Complete paper replication
            'mechanistic' - For mechanistic interpretability work
    
    Returns:
        tuple: (sft_config, rl_config, data_config)
    """
    if mode == 'quick':
        return (
            {**sft_config, **quick_test_config['sft']},
            {**rl_config, **quick_test_config['rl']},
            {**data_config, 
             'max_samples': quick_test_config['max_samples'],
             'eval_samples': quick_test_config['eval_samples'],
             'kl_samples': quick_test_config['kl_samples']}
        )
    
    elif mode == 'minimal':
        return (
            {**sft_config, **minimal_sweep_config['sft']},
            {**rl_config, **minimal_sweep_config['rl']},
            {**data_config,
             'max_samples': minimal_sweep_config['max_samples'],
             'eval_samples': minimal_sweep_config['eval_samples'],
             'kl_samples': minimal_sweep_config['kl_samples']}
        )
    
    elif mode == 'full':
        return (
            {**sft_config, **full_sweep_config['sft']},
            {**rl_config, **full_sweep_config['rl']},
            {**data_config,
             'max_samples': full_sweep_config['max_samples'],
             'eval_samples': full_sweep_config['eval_samples'],
             'kl_samples': full_sweep_config['kl_samples']}
        )
    
    elif mode == 'mechanistic':
        return mechanistic_config
    
    else:  # default
        return sft_config, rl_config, data_config


def count_total_runs(config_mode='default'):
    """
    Calculate total number of model training runs.
    
    Args:
        config_mode: Configuration mode
    
    Returns:
        dict: Number of runs for SFT and RL
    """
    sft_cfg, rl_cfg, _ = get_config(config_mode)
    
    # SFT runs
    sft_runs = (
        len(sft_cfg['learning_rates']) * 
        len(sft_cfg['batch_sizes']) * 
        len(sft_cfg['epochs'])
    )
    
    # RL runs
    rl_runs = (
        len(rl_cfg['learning_rates']) * 
        len(rl_cfg['batch_sizes']) * 
        len(rl_cfg['num_iterations'])
    )
    
    total_runs = sft_runs + rl_runs
    
    return {
        'sft_runs': sft_runs,
        'rl_runs': rl_runs,
        'total_runs': total_runs,
    }


def estimate_compute_hours(config_mode='default', model_size='3B'):
    """
    Estimate total GPU hours needed.
    
    Args:
        config_mode: Configuration mode
        model_size: Model size ('3B', '7B', '14B')
    
    Returns:
        dict: Compute estimates
    """
    runs = count_total_runs(config_mode)
    
    # Rough estimates (hours per run on A100)
    hours_per_run = {
        '3B': {'sft': 2, 'rl': 3},
        '7B': {'sft': 4, 'rl': 6},
        '14B': {'sft': 8, 'rl': 12},
    }
    
    if model_size not in hours_per_run:
        model_size = '3B'
    
    sft_hours = runs['sft_runs'] * hours_per_run[model_size]['sft']
    rl_hours = runs['rl_runs'] * hours_per_run[model_size]['rl']
    total_hours = sft_hours + rl_hours
    
    # Cost estimate (AWS p4d.xlarge ~$3/hour)
    cost_estimate = total_hours * 3
    
    return {
        'sft_hours': sft_hours,
        'rl_hours': rl_hours,
        'total_hours': total_hours,
        'estimated_cost_usd': cost_estimate,
    }


def print_config_summary(config_mode='default'):
    """Log summary of configuration."""
    logger.info("=" * 70)
    logger.info(f"CONFIGURATION SUMMARY: {config_mode.upper()}")
    logger.info("=" * 70)

    runs = count_total_runs(config_mode)
    compute = estimate_compute_hours(config_mode)

    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Training Runs: SFT runs={runs['sft_runs']}, RL runs={runs['rl_runs']}, "
                f"Total={runs['total_runs']}")

    logger.info(f"Compute Estimate (3B model on A100): SFT={compute['sft_hours']:.1f}h, "
                f"RL={compute['rl_hours']:.1f}h, Total={compute['total_hours']:.1f}h, "
                f"Cost=~${compute['estimated_cost_usd']:.0f}")

    sft_cfg, rl_cfg, data_cfg = get_config(config_mode)

    logger.info(f"SFT Config: LRs={sft_cfg['learning_rates']}, "
                f"Batch sizes={sft_cfg['batch_sizes']}, Epochs={sft_cfg['epochs']}")

    logger.info(f"RL Config: LRs={rl_cfg['learning_rates']}, "
                f"Batch sizes={rl_cfg['batch_sizes']}, Iterations={rl_cfg['num_iterations']}")

    logger.info(f"Data Config: Max samples={data_cfg['max_samples']}, "
                f"Eval samples={data_cfg['eval_samples']}, KL samples={data_cfg['kl_samples']}")

    logger.info("=" * 70)


# Print summary on import
if __name__ == "__main__":
    logger.info("Available configuration modes:")
    logger.info("  'quick' - Fast testing (1 run each, ~10 GPU hours)")
    logger.info("  'minimal' - Budget replication (~30 runs, ~100 GPU hours)")
    logger.info("  'default' - Standard configuration")
    logger.info("  'full' - Complete paper replication (~100 runs, ~300 GPU hours)")
    logger.info("  'mechanistic' - For mechanistic interpretability work")

    for mode in ['quick', 'minimal', 'full']:
        print_config_summary(mode)