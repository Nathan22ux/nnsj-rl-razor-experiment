# Configuration settings for RL Razor Experiment

import os

# =============================================================================
# GLOBAL MODEL CONSTANTS
# =============================================================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

ALTERNATIVE_MODELS = {
    'gpt2': 'gpt2',  # For quick testing
    'qwen_7b': 'Qwen/Qwen2.5-7B-Instruct',
    'qwen_14b': 'Qwen/Qwen2.5-14B-Instruct',
}

# =============================================================================
# CONSTANT VALUES (Hyperparameters & Settings)
# =============================================================================

# Dataset Constants
MAX_SAMPLE_SIZE = 2000
EVALUATION_SAMPLE_SIZE = 200
KL_SAMPLE_SIZE = 100

# Common Training Constants
LR_SCHEDULER_TYPE = 'constant_with_warmup'
WARMUP_STEPS = 50
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.0
BF16 = True
GRADIENT_ACCUMULATION_STEPS = 4

# SFT Specific Constants
SFT_LEARNING_RATES = [1e-5, 3e-5, 5e-5, 7e-5, 9e-5]
SFT_BATCH_SIZES = [16, 32, 64, 128]
SFT_EPOCHS = [1, 2]

# RL Specific Constants
RL_LEARNING_RATES = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
RL_BATCH_SIZES = [32, 64]
NUM_ITERATIONS = [1, 2]         # μ in paper
KL_COEFF = 0.0                  # Paper uses 0 - implicit KL minimization
NUM_GENERATIONS = 64            # Group size
PROMPTS_PER_GENERATION = 8

# Evaluation Constants
BENCHMARKS = [
    'winogrande',
    'hellaswag',
    'mmlu',
    'truthfulqa_mc2',
]
EXTENDED_BENCHMARKS = BENCHMARKS + [
    'arc_challenge',
    'arc_easy',
    'gsm8k',
]
LIMIT_PER_BENCHMARK = 100
NUM_FEWSHOT = 0
HUMAN_EVAL_LIMIT = 50
HUMAN_EVAL_TEMPERATURE = 0.2
IFEVAL_LIMIT = 100

# =============================================================================
# DEFAULT CONFIGURATION DICTIONARIES
# =============================================================================

# DATASET DEFAULTS
DEFAULT_DATA_CONFIG = {
    'max_samples': MAX_SAMPLE_SIZE,
    'eval_samples': EVALUATION_SAMPLE_SIZE,
    'kl_samples': KL_SAMPLE_SIZE
}

# SFT DEFAULTS (Exact from Paper Table 2)
DEFAULT_SFT_CONFIG = {
    # Hyperparameter sweep values
    'learning_rates': SFT_LEARNING_RATES,
    'batch_sizes': SFT_BATCH_SIZES,
    'epochs': SFT_EPOCHS,
    
    # Fixed hyperparameters
    'lr_scheduler': LR_SCHEDULER_TYPE,
    'warmup_steps': WARMUP_STEPS,
    'max_grad_norm': MAX_GRAD_NORM,
    'weight_decay': WEIGHT_DECAY,
    'bf16': BF16,
    'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
}

# RL DEFAULTS (Exact from Paper Table 2)
DEFAULT_RL_CONFIG = {
    # Hyperparameter sweep values
    'learning_rates': RL_LEARNING_RATES,
    'batch_sizes': RL_BATCH_SIZES,
    'num_iterations': NUM_ITERATIONS,
    
    # GRPO specific settings
    'kl_coeff': KL_COEFF,
    'num_generations': NUM_GENERATIONS,
    'prompts_per_generation': PROMPTS_PER_GENERATION,
    
    # Fixed hyperparameters
    'lr_scheduler': LR_SCHEDULER_TYPE,
    'warmup_steps': WARMUP_STEPS,
    'max_grad_norm': MAX_GRAD_NORM,
    'weight_decay': WEIGHT_DECAY,
    'bf16': BF16,
    'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
}

# EVALUATION DEFAULTS
DEFAULT_EVAL_CONFIG = {
    'benchmarks': BENCHMARKS,
    'extended_benchmarks': EXTENDED_BENCHMARKS,
    'limit_per_benchmark': LIMIT_PER_BENCHMARK,
    'num_fewshot': NUM_FEWSHOT,
    'humaneval_limit': HUMAN_EVAL_LIMIT,
    'humaneval_temperature': HUMAN_EVAL_TEMPERATURE,
    'ifeval_limit': IFEVAL_LIMIT,
}

# =============================================================================
# SPECIFIC MODES (Overrides)
# =============================================================================

# 1. QUICK TEST (For debugging)
QUICK_TEST_OVERRIDES = {
    'sft': {
        'learning_rates': [3e-5],
        'batch_sizes': [16],
        'epochs': [1],
    },
    'rl': {
        'learning_rates': [2e-5],
        'batch_sizes': [16],
        'num_iterations': [1],
    },
    'data': {
        'max_samples': 200,
        'eval_samples': 50,
        'kl_samples': 30,
    }
}

# 2. MINIMAL SWEEP (Budget conscious)
MINIMAL_SWEEP_OVERRIDES = {
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
    'data': {
        'max_samples': 1000,
        'eval_samples': 200,
        'kl_samples': 100,
    }
}

# 3. FULL SWEEP (Paper replication)
FULL_SWEEP_OVERRIDES = {
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
    'data': {
        'max_samples': 5000,
        'eval_samples': 500,
        'kl_samples': 200,
    }
}

# 4. MECHANISTIC CONFIGURATION
MECHANISTIC_CONFIG = {
    'models': {
        'primary': 'gpt2',
        'validation': 'meta-llama/Llama-2-7b-hf',
    },
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
    'sft_kl_penalties': [0, 0.01, 0.05, 0.1, 0.5],
    'adaptation_dataset': 'tatsu-lab/alpaca',
    'adaptation_size': 45000,
    'retention_datasets': {
        'natural_questions': {'size': 3000},
        'induction': {'size': 1000, 'synthetic': True},
        'bigbench': {'tasks': ['navigate', 'logical_deduction', 'causal_judgment']},
    },
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
            'default' - Balanced/Standard configuration
            'full' - Complete paper replication
            'mechanistic' - For mechanistic interpretability work
    
    Returns:
        tuple: (sft_config, rl_config, data_config) OR mechanistic_config dict
    """
    # 1. Start with defaults
    sft_config = DEFAULT_SFT_CONFIG.copy()
    rl_config = DEFAULT_RL_CONFIG.copy()
    data_config = DEFAULT_DATA_CONFIG.copy()
    
    # 2. Apply Overrides
    if mode == 'quick':
        sft_config.update(QUICK_TEST_OVERRIDES['sft'])
        rl_config.update(QUICK_TEST_OVERRIDES['rl'])
        data_config.update(QUICK_TEST_OVERRIDES['data'])
        
    elif mode == 'minimal':
        sft_config.update(MINIMAL_SWEEP_OVERRIDES['sft'])
        rl_config.update(MINIMAL_SWEEP_OVERRIDES['rl'])
        data_config.update(MINIMAL_SWEEP_OVERRIDES['data'])
        
    elif mode == 'full':
        sft_config.update(FULL_SWEEP_OVERRIDES['sft'])
        rl_config.update(FULL_SWEEP_OVERRIDES['rl'])
        data_config.update(FULL_SWEEP_OVERRIDES['data'])
        
    elif mode == 'mechanistic':
        return MECHANISTIC_CONFIG
    
    # 3. Return configured dictionaries
    return sft_config, rl_config, data_config


def count_total_runs(config_mode='default'):
    """Calculate total number of model training runs for a given mode."""
    if config_mode == 'mechanistic':
        return {'total_runs': 'N/A (Analysis Mode)'}

    sft_cfg, rl_cfg, _ = get_config(config_mode)
    
    # Calculate combinations
    sft_runs = (
        len(sft_cfg['learning_rates']) * len(sft_cfg['batch_sizes']) * len(sft_cfg['epochs'])
    )
    
    rl_runs = (
        len(rl_cfg['learning_rates']) * len(rl_cfg['batch_sizes']) * len(rl_cfg['num_iterations'])
    )
    
    return {
        'sft_runs': sft_runs,
        'rl_runs': rl_runs,
        'total_runs': sft_runs + rl_runs,
    }


def estimate_compute_hours(config_mode='default', model_size='3B'):
    """Estimate total GPU hours needed based on mode and model size."""
    if config_mode == 'mechanistic':
        return {'total_hours': 'N/A'}
        
    runs = count_total_runs(config_mode)
    
    # Rough estimates (hours per run on A100)
    hours_per_run = {
        '3B': {'sft': 2, 'rl': 3},
        '7B': {'sft': 4, 'rl': 6},
        '14B': {'sft': 8, 'rl': 12},
    }
    
    # Default to 3B if unknown
    size_key = model_size if model_size in hours_per_run else '3B'
    
    sft_hours = runs['sft_runs'] * hours_per_run[size_key]['sft']
    rl_hours = runs['rl_runs'] * hours_per_run[size_key]['rl']
    total_hours = sft_hours + rl_hours
    
    # Cost estimate (Approx AWS p4d.xlarge ~$3/hour)
    cost_estimate = total_hours * 3
    
    return {
        'sft_hours': sft_hours,
        'rl_hours': rl_hours,
        'total_hours': total_hours,
        'estimated_cost_usd': cost_estimate,
    }


def print_config_summary(config_mode='default'):
    """Print a pretty summary of the selected configuration."""
    print(f"\n{'='*70}")
    print(f"CONFIGURATION SUMMARY: {config_mode.upper()}")
    print(f"{'='*70}")
    
    if config_mode == 'mechanistic':
        print("\nMode: MECHANISTIC INTERPRETABILITY")
        print("See MECHANISTIC_CONFIG dictionary for details.")
        return

    runs = count_total_runs(config_mode)
    compute = estimate_compute_hours(config_mode)
    sft_cfg, rl_cfg, data_cfg = get_config(config_mode)
    
    print(f"\nModel: {MODEL_NAME}")
    print(f"\nTraining Runs:")
    print(f"   SFT runs: {runs['sft_runs']}")
    print(f"   RL runs: {runs['rl_runs']}")
    print(f"   Total runs: {runs['total_runs']}")
    
    print(f"\nCompute Estimate (3B model on A100):")
    print(f"   SFT hours: {compute['sft_hours']}")
    print(f"   RL hours: {compute['rl_hours']}")
    print(f"   Total hours: {compute['total_hours']}")
    print(f"   Estimated cost: ${compute['estimated_cost_usd']}")
    
    print(f"\nSFT Configuration:")
    print(f"   Learning rates: {sft_cfg['learning_rates']}")
    print(f"   Batch sizes: {sft_cfg['batch_sizes']}")
    print(f"   Epochs: {sft_cfg['epochs']}")
    
    print(f"\nRL Configuration:")
    print(f"   Learning rates: {rl_cfg['learning_rates']}")
    print(f"   Batch sizes: {rl_cfg['batch_sizes']}")
    print(f"   Iterations: {rl_cfg['num_iterations']}")
    
    print(f"\nData Configuration:")
    print(f"   Max samples: {data_cfg['max_samples']}")
    print(f"   Eval samples: {data_cfg['eval_samples']}")
    print(f"   KL samples: {data_cfg['kl_samples']}")
    
    print(f"\n{'='*70}\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\nAvailable configuration modes:")
    print("   'quick'       - Fast testing (1 run each, ~10 GPU hours)")
    print("   'minimal'     - Budget replication (~30 runs, ~100 GPU hours)")
    print("   'default'     - Standard configuration")
    print("   'full'        - Complete paper replication (~100 runs, ~300 GPU hours)")
    print("   'mechanistic' - For mechanistic interpretability work\n")
    
    # Print summaries for common modes
    for mode in ['quick', 'minimal', 'full']:
        print_config_summary(mode)