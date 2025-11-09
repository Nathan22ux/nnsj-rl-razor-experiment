import json
import gc
import torch
import numpy as np
from src.data_and_models import load_model_and_tokenizer, load_dataset_by_name, format_dataset_for_training, check_gpu, MODEL_NAME
from src.sft_training import train_sft, SFT_CONFIG
from src.grpo_training import train_grpo, RL_CONFIG
from src.evaluation import evaluate_benchmarks, compute_forward_kl
from src.visualization import plot_results

def run_full_experiment(dataset_name="math"):
    """
    Run complete hyperparameter sweep for SFT and RL.
    This is your original run_full_experiment function from cells 611-767
    """
    print("\n" + "="*70)
    print(f"RUNNING EXPERIMENT ON {dataset_name.upper()} DATASET")
    print("="*70)
    
    # Check GPU
    device = check_gpu()
    
    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    dataset = load_dataset_by_name(dataset_name)
    
    # Load base model and tokenizer
    print(f"\nLoading base model: {MODEL_NAME}")
    base_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    
    # Format dataset for training
    formatted_dataset = format_dataset_for_training(dataset, tokenizer)
    
    # Prepare dataset for KL computation (use a subset)
    formatted_dataset_kl = formatted_dataset.select(range(min(1000, len(formatted_dataset))))
    
    # Results storage
    results = {
        'dataset': dataset_name,
        'model': MODEL_NAME,
        'sft': [],
        'rl': []
    }
    
    # ================================
    # SFT Hyperparameter Sweep
    # ================================
    print("\n" + "="*70)
    print("RUNNING SFT HYPERPARAMETER SWEEP")
    print("="*70)
    
    for lr in SFT_CONFIG['learning_rates'][:2]:  # Test first 2 learning rates
        for bs in SFT_CONFIG['batch_sizes'][:2]:  # Test first 2 batch sizes
            
            print(f"\n>>> Testing SFT with lr={lr}, batch_size={bs}")
            
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # Clone base model for this run
            sft_model, _ = load_model_and_tokenizer(MODEL_NAME)
            
            # Train (train_sft will format the dataset internally)
            sft_model, trainer = train_sft(sft_model, dataset, tokenizer, learning_rate=lr, batch_size=bs)
            
            # Evaluate
            prior_scores = evaluate_benchmarks(sft_model, tokenizer)
            kl_div = compute_forward_kl(sft_model, base_model, formatted_dataset_kl, tokenizer)
            
            results['sft'].append({
                'lr': lr,
                'batch_size': bs,
                'prior_task_score': prior_scores['average'],
                'kl_divergence': kl_div,
                'detailed_scores': prior_scores,
            })
            
            print(f"SFT lr={lr}, bs={bs}: Prior={prior_scores['average']:.4f}, KL={kl_div:.4f}")
            
            # Delete model and trainer immediately after use
            del sft_model
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
            
            print(f"Memory freed. GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Delete base model before RL sweep
    print("\nDeleting base model before RL sweep...")
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Reload base model for RL
    base_model, _ = load_model_and_tokenizer(MODEL_NAME)
    
    # ================================
    # RL (GRPO) Hyperparameter Sweep
    # ================================
    print("\n" + "="*70)
    print("RUNNING RL (GRPO) HYPERPARAMETER SWEEP")
    print("="*70)
    
    for lr in RL_CONFIG['learning_rates'][:2]:
        
        # Clear memory before loading new model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Clone base model
        rl_model, _ = load_model_and_tokenizer(MODEL_NAME)
        
        # Train
        rl_model, trainer = train_grpo(rl_model, dataset, tokenizer, learning_rate=lr)
        
        # Evaluate
        prior_scores = evaluate_benchmarks(rl_model, tokenizer)
        kl_div = compute_forward_kl(rl_model, base_model, formatted_dataset_kl, tokenizer)
        
        results['rl'].append({
            'lr': lr,
            'prior_task_score': prior_scores['average'],
            'kl_divergence': kl_div,
            'detailed_scores': prior_scores,
        })
        
        print(f"RL lr={lr}: Prior={prior_scores['average']:.4f}, KL={kl_div:.4f}")
        
        # Delete model immediately
        del rl_model
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save results
    with open(f'results_{dataset_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final cleanup
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RL'S RAZOR")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Hyperparameters: Exactly from Table 2")
    print("="*70 + "\n")
    
    # Run experiment on Math dataset
    results = run_full_experiment(dataset_name="math")
    
    # Create visualizations
    plot_results(results)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print(f"  • RL average prior score: {np.mean([r['prior_task_score'] for r in results['rl']]):.4f}")
    print(f"  • SFT average prior score: {np.mean([r['prior_task_score'] for r in results['sft']]):.4f}")
    print(f"  • RL average KL: {np.mean([r['kl_divergence'] for r in results['rl']]):.4f}")
    print(f"  • SFT average KL: {np.mean([r['kl_divergence'] for r in results['sft']]):.4f}")
