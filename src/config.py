# Model Configuration
MODEL_NAME = "openai-community/gpt2"  # Changed to a smaller model, target LLAMA-3B

# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# SFT Hyperparameters (Reference Table 2 located in page 19 on the paper RL razor)
sft_config = {
    'learning_rates': [1e-5, 3e-5, 5e-5, 7e-5, 9e-5],
    'batch_sizes': [16, 32, 64, 128],
    'epochs': [1, 2],
    'lr_scheduler': ['constant_with_warmup', 'cosine_with_warmup'],
    'warmup_steps': 50,
    'optimizer': 'adamw',
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'bf16': True,
}

# RL Hyperparameters
rl_config = {
    'learning_rates': [1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
    'epochs': 1,
    'warmup_steps': 50,
    'optimizer': 'adamw',
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'bf16': True,
    'kl_reg': 0.0,  # NO explicit KL regularization
    'group_size': 64,
    'prompts_per_generation': 8,
    'num_iterations': [1, 2],
}

