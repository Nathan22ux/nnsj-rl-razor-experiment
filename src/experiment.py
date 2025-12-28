"""
Corrected experiment.py for RL's Razor Replication

FIXES APPLIED:
- Imports data_config for max_samples
- Passes max_samples to train_sft() and train_grpo()
- Passes batch_size to train_grpo()
- Uses response_only=True for KL computation
- Sweeps batch_size for RL (fair comparison with SFT)
"""

import os
import gc
import json
import torch
from transformers import AutoModelForCausalLM

# FIXED IMPORTS
from config import MODEL_NAME, get_config
from training import train_sft, train_grpo  # Uses fixed training.py
from evaluation import (  # Uses fixed evaluation.py
    evaluate_benchmarks, 
    compute_forward_kl,
    compute_reverse_kl,
    compute_js_divergence
)
from dataset_utils import (  # NEW import
    UnifiedDatasetInterface,
    load_and_normalize_dataset
)


def run_full_experiment(dataset, tokenizer, dataset_name="math", config_mode="minimal"):
    """
    Run full experiment with FIXED implementations.
    
    Args:
        dataset: Training dataset (will be normalized)
        tokenizer: Tokenizer
        dataset_name: Dataset name for saving results
        config_mode: Configuration mode ('quick', 'minimal', 'full')
    """
    print(f"\\n{'='*70}")
    print(f"STARTING EXPERIMENT")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"Config mode: {config_mode}")
    print(f"{'='*70}\\n")
    
    # Get configuration
    sft_cfg, rl_cfg, data_cfg = get_config(config_mode)
    
    # Normalize dataset if needed
    print(" Normalizing dataset...")
    if 'text' not in dataset.column_names:
        dataset = UnifiedDatasetInterface.normalize_dataset(dataset)
    print(f" Dataset ready: {len(dataset)} examples\\n")
    
    # Format dataset for KL computation
    print(" Preparing dataset for KL computation...")
    if 'text' in dataset.column_names:
        kl_dataset = dataset
    else:
        kl_dataset = UnifiedDatasetInterface.normalize_dataset(dataset)
    print(f" KL dataset ready\\n")
    
    # Setup results tracking
    os.makedirs("results", exist_ok=True)
    results_file = f"results/results_{dataset_name}_{config_mode}.json"
    
    if os.path.exists(results_file):
        print(f" Found existing results: {results_file}")
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"  Completed: {len(results.get('sft', []))} SFT, {len(results.get('rl', []))} RL\\n")
    else:
        results = {'sft': [], 'rl': []}
    
    # Load base model ONCE
    print(f" Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(" Base model loaded\\n")
    
    # SFT sweep
    print(f"{'='*70}")
    print(f"SFT HYPERPARAMETER SWEEP")
    print(f"{'='*70}\\n")
    
    for lr in sft_cfg['learning_rates']:
        for bs in sft_cfg['batch_sizes']:
            for epochs in sft_cfg['epochs']:
                
                # Check if already done
                if any(r['lr']==lr and r['batch_size']==bs and r['epochs']==epochs 
                       for r in results.get('sft', [])):
                    print(f" Skipping SFT lr={lr}, bs={bs}, epochs={epochs} (done)\\n")
                    continue
                
                print(f" Training SFT: lr={lr}, bs={bs}, epochs={epochs}")
                
                # Clone model
                sft_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
                )
                
                # Train
                sft_model, trainer, NT = train_sft(
                    sft_model, dataset, tokenizer,
                    learning_rate=lr,
                    batch_size=bs,
                    epochs=epochs,
                    max_samples=data_cfg['max_samples'],
                    eval_samples = data_cfg['eval_samples']
                )
                
                # Evaluate
                prior_scores = evaluate_benchmarks(sft_model, tokenizer, limit=100)
                kl_div = compute_forward_kl(
                    sft_model, base_model, kl_dataset, tokenizer,
                    num_samples=data_cfg['kl_samples'],
                    response_only=True
                )
                
                # Save results
                results['sft'].append({
                    'lr': lr,
                    'batch_size': bs,
                    'epochs': epochs,
                    'NT': NT,
                    'PT': prior_scores['average'],
                    'kl_divergence': kl_div,
                    'detailed_scores': prior_scores,
                })
                
                print(f"  NT: {NT:.2f}%, PT: {prior_scores['average']:.4f}, KL: {kl_div:.4f}\\n")
                
                # Save checkpoint
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Cleanup
                del sft_model, trainer
                torch.cuda.empty_cache()
                gc.collect()
    
    # RL sweep (similar structure)
    print(f"{'='*70}")
    print(f"RL HYPERPARAMETER SWEEP")
    print(f"{'='*70}\\n")
    
    for lr in rl_cfg['learning_rates']:
        for bs in rl_cfg['batch_sizes']:
            
            if any(r['lr']==lr and r['batch_size']==bs for r in results.get('rl', [])):
                print(f" Skipping RL lr={lr}, bs={bs} (done)\\n")
                continue
            
            print(f" Training RL: lr={lr}, bs={bs}")
            
            rl_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
            )
            
            rl_model, trainer, NT = train_grpo(
                rl_model, dataset, tokenizer,
                learning_rate=lr,
                batch_size=bs,
                max_samples=data_cfg['max_samples'],
                eval_samples = data_cfg['eval_samples']
            )
            
            prior_scores = evaluate_benchmarks(rl_model, tokenizer, limit=100)
            kl_div = compute_forward_kl(
                rl_model, base_model, kl_dataset, tokenizer,
                num_samples=data_cfg['kl_samples'],
                response_only=True
            )
            
            results['rl'].append({
                'lr': lr,
                'batch_size': bs,
                'NT': NT,
                'PT': prior_scores['average'],
                'kl_divergence': kl_div,
                'detailed_scores': prior_scores,
            })
            
            print(f"  NT: {NT:.2f}%, PT: {prior_scores['average']:.4f}, KL: {kl_div:.4f}\\n")
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            del rl_model, trainer
            torch.cuda.empty_cache()
            gc.collect()
    
    # Cleanup base model
    del base_model
    torch.cuda.empty_cache()
    
    print(f"{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"SFT runs: {len(results['sft'])}")
    print(f"RL runs: {len(results['rl'])}")
    print(f"{'='*70}\\n")
    
    return results