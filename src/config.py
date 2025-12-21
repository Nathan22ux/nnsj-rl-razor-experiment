# Corrected config.py for RL's Razor Replication
#
# Key changes from original:
# 1. Added max_samples parameter (was hardcoded to 50 in training.py)
# 2. Expanded hyperparameter sweeps for proper Pareto frontier
# 3. Matched GRPO settings to SFT for fair comparison
# 4. Added GRPO-specific parameters (num_generations, etc.)
# 5. Added all benchmarks from Table 1 (TruthfulQA, IFEval, HumanEval)

# Model Configuration
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# =============================================================================
# TRAINING DATA CONFIGURATION
# =============================================================================
# CRITICAL: The original code hardcoded this to 50 in training.py
# RL's Razor paper used thousands of examples per sweep point
data_config = {
    'max_samples': 1000,           # Training samples per run
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
# SFT config
sft_config = {
    'learning_rates': [1e-5, 3e-5, 5e-5],  # Paper values [1e-5, 3e-5, 5e-5, 7e-5, 9e-5]
    'batch_sizes': [32, 64],                   # Paper values [16, 32, 64, 128]
    'epochs': [1],                                    # Paper values [1, 2]
    'lr_scheduler': 'constant_with_warmup',  # or 'cosine_with_warmup'
    'warmup_steps': 50,
    'max_grad_norm': 1,
    'weight_decay': 0,
    'bf16': True,
}

# RL config
rl_config = {
    'learning_rates': [1e-5, 3e-5, 5e-5],  # Paper values [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
    'kl_coef': 0,                    # ✅ Paper uses 0 - keep it!
    'num_generations': 64,           # "Group Size" in paper
    'prompts_per_generation': 8,     # Paper value
    'num_iterations': [1, 2],        # μ in paper
    'lr_scheduler': 'constant_with_warmup',
    'warmup_steps': 50,
    'max_grad_norm': 1,
    'weight_decay': 0,
    'bf16': True,
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
# Prior task benchmarks (measure catastrophic forgetting)
# Updated to include ALL benchmarks from Table 1 in paper
eval_config = {
    # Core benchmarks (always run)
    'benchmarks': [
        'winogrande',
        'hellaswag',
        'mmlu_high_school_mathematics',
        'mmlu_high_school_computer_science',
        'truthfulqa_mc2',           # Added: TruthfulQA
    ],

    # Extended benchmarks (optional, more comprehensive)
    'extended_benchmarks': [
        'winogrande',
        'hellaswag',
        'mmlu_high_school_mathematics',
        'mmlu_high_school_computer_science',
        'truthfulqa_mc2',
        'arc_challenge',
        'arc_easy',
    ],

    # Code evaluation (separate from lm-eval)
    'code_benchmarks': [
        'humaneval',                # Requires human-eval package
    ],

    # Instruction following (separate evaluation)
    'instruction_benchmarks': [
        'ifeval',                   # Requires special handling
    ],

    'limit_per_benchmark': 100,    # Samples per benchmark (was 50)
    'num_fewshot': 0,              # Zero-shot evaluation

    # HumanEval settings
    'humaneval_limit': 50,         # Number of HumanEval problems
    'humaneval_temperature': 0.2,  # Sampling temperature for code gen
}

# =============================================================================
# CIRCUIT ANALYSIS CONFIGURATION
# =============================================================================
circuit_config = {
    'top_k_heads': 20,             # Number of important heads to identify
    'max_examples': 100,           # Examples for circuit discovery
    'vulnerability_threshold': 0.1,

    # DCM settings (Equation 3)
    'dcm_lambda_sparsity': 0.1,    # Sparsity penalty for DCM
    'dcm_iterations': 100,         # Training iterations for DCM mask
    'dcm_lr': 0.1,                 # Learning rate for DCM optimization

    # Faithfulness settings (Equation 4)
    'faithfulness_ablation_type': 'zero',  # 'zero' or 'mean'
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