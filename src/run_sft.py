import json
import gc
import torch
import numpy as np
from src.data_and_models import load_model_and_tokenizer, load_dataset_by_name, format_dataset_for_training, check_gpu, MODEL_NAME
from src.sft_training import train_sft, SFT_CONFIG
from src.evaluation import evaluate_benchmarks, compute_forward_kl

def run_sft_experiment(dataset_name="math"):
    """Run SFT hyperparameter sweep only."""
    
    print("\n" + "="*70)
    print(f"RUNNING SFT EXPERIMENT ON {dataset_name.upper()} DATASET")
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
    # SFT Hyperparameter Sweep (from cells 611-767)
    # ================================
    print("\n" + "="*70)
    print("RUNNING SFT HYPERPARAMETER SWEEP")
    print("="*70)
    
    # Test a subset of hyperparameters
    for lr in SFT_CONFIG['learning_rates'][:2]:  # Test first 2 learning rates
        for bs in SFT_CONFIG['batch_sizes'][:2]:  # Test first 2 batch sizes
            
            print(f"\n>>> Testing SFT with lr={lr}, batch_size={bs}")
            
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # Clone base model for this run
            sft_model, _ = load_model_and_tokenizer(MODEL_NAME)
            
            # Train
            sft_model, trainer = train_sft(
                sft_model, 
                formatted_dataset,
                tokenizer,
                learning_rate=lr,
                batch_size=bs,
                num_epochs=1,
                output_dir=f"./sft_output/lr{lr}_bs{bs}"
            )
            
            # Evaluate
            prior_scores = evaluate_benchmarks(sft_model, tokenizer)
            kl_div = compute_forward_kl(sft_model, base_model, formatted_dataset_kl, tokenizer)
            
            # Store results
            results.append({
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
    
    # Save results
    with open(f'sft_results_{dataset_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: sft_results_{dataset_name}.json")
    
    # Clean up
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Print summary (from cells 847-870)
    print("\n" + "="*70)
    print("SFT EXPERIMENT COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    avg_prior = np.mean([r['prior_task_score'] for r in results])
    avg_kl = np.mean([r['kl_divergence'] for r in results])
    print(f"  • SFT average prior score: {avg_prior:.4f}")
    print(f"  • SFT average KL: {avg_kl:.4f}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SFT experiment")
    parser.add_argument("--dataset", type=str, default="math", 
                       choices=["math", "science", "tool"],
                       help="Dataset to use")
    args = parser.parse_args()
    
    # Run the experiment
    results = run_sft_experiment(dataset_name=args.dataset)
