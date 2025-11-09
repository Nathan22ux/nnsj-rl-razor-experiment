import json
import gc
import torch
import numpy as np
from src.data_and_models import load_model_and_tokenizer, load_dataset_by_name, format_dataset_for_training, check_gpu, MODEL_NAME
from src.grpo_training import train_grpo, RL_CONFIG
from src.evaluation import evaluate_benchmarks, compute_forward_kl

def run_grpo_experiment(dataset_name="math"):
    """Run GRPO/RL hyperparameter sweep only."""
    
    print("\n" + "="*70)
    print(f"RUNNING GRPO EXPERIMENT ON {dataset_name.upper()} DATASET")
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
    results = []
    
    # ================================
    # RL (GRPO) Hyperparameter Sweep (from cells 611-767)
    # ================================
    print("\n" + "="*70)
    print("RUNNING RL (GRPO) HYPERPARAMETER SWEEP")
    print("="*70)
    
    for lr in RL_CONFIG['learning_rates'][:2]:  # Test first 2 learning rates
        
        print(f"\n>>> Testing GRPO with lr={lr}")
        
        # Clear memory before loading new model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Clone base model
        rl_model, _ = load_model_and_tokenizer(MODEL_NAME)
        
        # Train
        rl_model, trainer = train_grpo(
            rl_model,
            formatted_dataset,
            tokenizer,
            learning_rate=lr,
            num_iterations=1,
            output_dir=f"./grpo_output/lr{lr}"
        )
        
        # Evaluate
        prior_scores = evaluate_benchmarks(rl_model, tokenizer)
        kl_div = compute_forward_kl(rl_model, base_model, formatted_dataset_kl, tokenizer)
        
        # Store results
        results.append({
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
    with open(f'grpo_results_{dataset_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: grpo_results_{dataset_name}.json")
    
    # Clean up
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Print summary (from cells 847-870)
    print("\n" + "="*70)
    print("GRPO EXPERIMENT COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    avg_prior = np.mean([r['prior_task_score'] for r in results])
    avg_kl = np.mean([r['kl_divergence'] for r in results])
    print(f"  • RL average prior score: {avg_prior:.4f}")
    print(f"  • RL average KL: {avg_kl:.4f}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GRPO experiment")
    parser.add_argument("--dataset", type=str, default="math",
                       choices=["math", "science", "tool"],
                       help="Dataset to use")
    args = parser.parse_args()
    
    # Run the experiment
    results = run_grpo_experiment(dataset_name=args.dataset)
