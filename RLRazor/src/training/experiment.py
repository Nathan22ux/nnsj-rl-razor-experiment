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

# FIX: Added data_config import
from config import MODEL_NAME, sft_config, rl_config, data_config
from training import train_sft, train_grpo
from evaluation import evaluate_benchmarks, compute_forward_kl, compute_kl_on_task_distribution




def run_full_experiment(dataset, tokenizer, dataset_name="math"):
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
    print(f"Max training samples: {data_config['max_samples']}")
    print(f"KL samples: {data_config['kl_samples']}")
    print("="*70 + "\n")

    # FIXED: Create proper train/eval split
    print("\n Creating train/eval split...")
    dataset_size = len(dataset)
    eval_size = min(200, int(dataset_size * 0.1))  # 10% or 200, whichever is smaller
    train_size = dataset_size - eval_size

    # Shuffle indices for random split
    import random
    all_indices = list(range(dataset_size))
    random.seed(42)  # Reproducibility
    random.shuffle(all_indices)

    train_indices = all_indices[:train_size]
    eval_indices = all_indices[train_size:]

    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(eval_indices)

    print(f" Train set: {len(train_dataset)} examples")
    print(f" Eval set: {len(eval_dataset)} examples")

    # Check for existing results to resume from
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
                print(" Model loaded")

                # FIX: Pass max_samples from data_config
                # FIX: Pass max_samples from data_config and use train_dataset
                sft_model, trainer, NT = train_sft(
                    sft_model, train_dataset, tokenizer,
                    learning_rate=lr,
                    batch_size=bs,
                    epochs=epochs,
                    max_samples=data_config['max_samples'],
                    eval_dataset=eval_dataset
                )
                
                # Evaluate
                print(f"\n Evaluating trained SFT model...")
                prior_scores = evaluate_benchmarks(sft_model, tokenizer)

                # FIX: Use response_only=True and num_samples from config
                # FIX: Use task distribution KL (paper's method)
                print(f"\n Computing KL divergence on task distribution...")
                task_prompts = []
                for i in range(min(data_config['kl_samples'], len(dataset))):
                    question = dataset[i]['0']['value']
                    task_prompts.append(f"Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer:")

                kl_div = compute_kl_on_task_distribution(
                    sft_model, base_model, tokenizer, task_prompts,
                    num_samples=data_config['kl_samples']
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
            print(" Model loaded")

            # FIX: Pass batch_size, max_samples, and use train_dataset
            rl_model, trainer, NT = train_grpo(
                rl_model, train_dataset, tokenizer,
                learning_rate=lr,
                batch_size=bs,
                max_samples=data_config['max_samples'],
                eval_dataset=eval_dataset
            )

            # Evaluate
            print(f"\n Evaluating trained RL model...")
            prior_scores = evaluate_benchmarks(rl_model, tokenizer)

            # FIX: Use response_only=True and num_samples from config
            print(f"\n Computing KL divergence on task distribution...")
            task_prompts = []
            for i in range(min(data_config['kl_samples'], len(dataset))):
                question = dataset[i]['0']['value']
                task_prompts.append(f"Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer:")

            kl_div = compute_kl_on_task_distribution(
                rl_model, base_model, tokenizer, task_prompts,
                num_samples=data_config['kl_samples']
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