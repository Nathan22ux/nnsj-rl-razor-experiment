# Corrected config.py for RL's Razor Replication
#
# Key changes from original:
# 1. Added max_samples parameter (was hardcoded to 50 in training.py)
# 2. Expanded hyperparameter sweeps for proper Pareto frontier
# 3. Matched GRPO settings to SFT for fair comparison
# 4. Added GRPO-specific parameters (num_generations, etc.)

# Model Configuration
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# =============================================================================
# TRAINING DATA CONFIGURATION
# =============================================================================
# CRITICAL: The original code hardcoded this to 50 in training.py
# RL's Razor paper used thousands of examples per sweep point
data_config = {
    'max_samples': 1000,           # Training samples per run (was 50!)
    'eval_samples': 200,           # Separate evaluation samples
    'kl_samples': 100,             # Samples for KL computation
}

# =============================================================================
# SFT HYPERPARAMETERS
# =============================================================================
# Reference: RL's Razor paper Table 2 (Appendix B)
#
# For Pareto frontier, you need MULTIPLE configurations to get different
# points on the learning-forgetting tradeoff curve
sft_config = {
    # Sweep these for Pareto frontier (uncomment for full sweep)
    'learning_rates': [1e-5, 3e-5, 5e-5, 7e-5],  # Paper sweeps 5 values
    'batch_sizes': [16, 32, 64],                  # Paper sweeps 4 values
    'epochs': [1, 2],                             # Paper sweeps 2 values

    # For quick testing, use single values:
    # 'learning_rates': [3e-5],
    # 'batch_sizes': [32],
    # 'epochs': [1],

    # Fixed hyperparameters (match paper)
    'lr_scheduler': 'constant_with_warmup',
    'warmup_steps': 50,
    'optimizer': 'adamw',
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'bf16': True,

    # Gradient accumulation (for effective batch size)
    # Effective batch = batch_size * gradient_accumulation_steps
    'gradient_accumulation_steps': 4,

    # Max sequence length
    'max_length': 1024,
}

# =============================================================================
# RL (GRPO) HYPERPARAMETERS
# =============================================================================
# Reference: RL's Razor paper + DeepSeekMath GRPO paper
#
# CRITICAL SETTINGS:
# - kl_coef=0.0: Paper uses NO explicit KL regularization
#   (the point is that GRPO implicitly minimizes KL via on-policy updates)
# - num_generations: Number of samples per prompt for GRPO
# - Binary reward: Paper uses only success/failure reward
rl_config = {
    # Sweep these for Pareto frontier
    'learning_rates': [1e-5, 2e-5, 3e-5, 5e-5],  # Paper sweeps 5 values

    # For quick testing:
    # 'learning_rates': [3e-5],

    # Fixed hyperparameters
    'epochs': 1,
    'lr_scheduler': 'constant_with_warmup',
    'warmup_steps': 50,
    'optimizer': 'adamw',
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'bf16': True,

    # GRPO-specific settings
    'kl_coef': 0.0,                # NO explicit KL regularization (paper's setting)
    'num_generations': 4,          # Samples per prompt (paper uses 4-16)
    'temperature': 0.7,            # Sampling temperature for generation
    'max_completion_length': 256,  # Max tokens to generate

    # Batch sizes - MUST MATCH SFT for fair comparison!
    'batch_sizes': [16, 32, 64],   # Should match SFT sweep
    'gradient_accumulation_steps': 4,  # Match SFT

    # Clipping
    'epsilon': 0.2,                # PPO-style clipping (GRPO default)

    # Reward settings (handled in training.py, but documented here)
    # Paper uses binary reward (0 or 1) based on answer correctness
    # Our fix adds partial rewards for gradient signal
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
# Prior task benchmarks (measure catastrophic forgetting)
eval_config = {
    'benchmarks': [
        'winogrande',
        'hellaswag',
        'mmlu_high_school_mathematics',
        'mmlu_high_school_computer_science',
    ],
    'limit_per_benchmark': 100,    # Samples per benchmark (was 50)
    'num_fewshot': 0,              # Zero-shot evaluation
}

# =============================================================================
# CIRCUIT ANALYSIS CONFIGURATION (if you're doing that part)
# =============================================================================
circuit_config = {
    'top_k_heads': 20,             # Number of important heads to identify
    'max_examples': 100,           # Examples for circuit discovery
    'vulnerability_threshold': 0.1,
}

# =============================================================================
# QUICK START CONFIGURATION
# =============================================================================
# Use these for initial testing before full sweep
quick_test_config = {
    'sft': {
        'learning_rates': [3e-5],
        'batch_sizes': [32],
        'epochs': [1],
    },
    'rl': {
        'learning_rates': [3e-5],
    },
    'max_samples': 200,  # Smaller for testing
}

# =============================================================================
# FULL SWEEP CONFIGURATION
# =============================================================================
# Use these for actual paper replication (will take much longer)
full_sweep_config = {
    'sft': {
        'learning_rates': [1e-5, 3e-5, 5e-5, 7e-5, 9e-5],
        'batch_sizes': [16, 32, 64, 128],
        'epochs': [1, 2],
    },
    'rl': {
        'learning_rates': [1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
    },
    'max_samples': 2000,
}


# =============================================================================
# HELPER FUNCTION
# =============================================================================
def get_config(mode='default'):
    """
    Get configuration based on mode.

    Args:
        mode: 'quick' for testing, 'full' for paper replication, 'default' for balanced

    Returns:
        tuple: (sft_config, rl_config, data_config)
    """
    if mode == 'quick':
        return (
            {**sft_config, **quick_test_config['sft']},
            {**rl_config, **quick_test_config['rl']},
            {**data_config, 'max_samples': quick_test_config['max_samples']}
        )
    elif mode == 'full':
        return (
            {**sft_config, **full_sweep_config['sft']},
            {**rl_config, **full_sweep_config['rl']},
            {**data_config, 'max_samples': full_sweep_config['max_samples']}
        )
    else:
        return sft_config, rl_config, data_config

