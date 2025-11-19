import os
import gc
import json
import torch
from transformers import AutoModelForCausalLM

from config import MODEL_NAME, sft_config, rl_config
from training import train_sft, train_grpo
from evaluation import evaluate_benchmarks, compute_forward_kl


def run_full_experiment(dataset, tokenizer, dataset_name="math"):
    """
    Run full experiment to create Pareto frontier (Figure 2)
    
    Args:
        dataset: Training dataset
        tokenizer: Model tokenizer
        dataset_name: Name of dataset being used
        
    Returns:
        dict: Results from SFT and RL experiments
    """
    print("\n" + "="*70)
    print("STARTING FULL EXPERIMENT")
    print("="*70)
    print(f"Dataset: {dataset_name}")
    print("="*70 + "\n")
    
    # Set memory management
    print(" Configuring memory management...")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(" Memory management configured")
    
    # Clear GPU memory before starting
    print("\n Clearing GPU memory before starting...")
    torch.cuda.empty_cache()
    gc.collect()
    print(" GPU memory cleared")
    
    # FORMAT DATASET ONCE HERE
    print("\n→ Formatting dataset for KL computation...")
    def format_dataset_for_kl(examples):
        """Convert the nested structure to text format"""
        texts = []
        for i in range(len(examples['0'])):
            question = examples['0'][i]['value']
            try:
                answer = examples['1'][i]['ground_truth']['value']
            except (KeyError, TypeError):
                answer = str(examples['1'][i])
            text = f"Question: {question}\nAnswer: {answer}"
            if len(text) > 800:
                text = text[:800]
            texts.append(text)
        return {'text': texts}
    
    # Create formatted version for KL computation
    formatted_dataset_kl = dataset.map(format_dataset_for_kl, batched=True, remove_columns=dataset.column_names)
    print(f" Dataset formatted for KL: {len(formatted_dataset_kl)} examples")
    
    # Load base model ONCE
    print(f"\n Loading base model: {MODEL_NAME}")
    print("   (This may take a while...)")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(" Base model loaded successfully")
    
    results = {
        'sft': [],
        'rl': [],
    }
    
    # SFT sweep (as in Table 2)
    print("\n" + "="*70)
    print("RUNNING SFT HYPERPARAMETER SWEEP")
    print("="*70)
    print(f"Learning rates to test: {sft_config['learning_rates']}")
    print(f"Batch sizes to test: [2, 4]")
    print("="*70 + "\n")
    
    sft_run_count = 0
    total_sft_runs = len(sft_config['learning_rates']) * 2  # 2 batch sizes
    
    for lr in sft_config['learning_rates']:
        for bs in [2, 4]:  # Smaller batch sizes, on paper [16,64]
            
            sft_run_count += 1
            print(f"\n{'*'*70}")
            print(f"SFT RUN {sft_run_count}/{total_sft_runs}: lr={lr}, batch_size={bs}")
            print(f"{'*'*70}")
            
            # Clear memory before loading new model
            print("\n Clearing GPU memory...")
            torch.cuda.empty_cache()
            gc.collect()
            print(f" Memory cleared. GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            
            print(f"\n→ Loading fresh model for this run...")
            
            # Clone base model
            sft_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            print(" Model loaded")
            
            # Train (train_sft will format the dataset internally)
            # sft_model, trainer = train_sft(sft_model, dataset, tokenizer, learning_rate=lr, batch_size=bs)
            # Train from NT too
            sft_model, trainer, NT = train_sft(sft_model, dataset, tokenizer, learning_rate=lr, batch_size=bs)
            # Evaluate
            print(f"\n Evaluating trained SFT model...")
            prior_scores = evaluate_benchmarks(sft_model, tokenizer)
            
            print(f"\n Computing KL divergence...")
            kl_div = compute_forward_kl(sft_model, base_model, formatted_dataset_kl, tokenizer)  # Use formatted dataset
            
            results['sft'].append({
                'lr': lr,
                'batch_size': bs,
                'NT': NT,
                'prior_task_score': prior_scores['average'],
                'kl_divergence': kl_div,
                'detailed_scores': prior_scores,
            })
            
            print(f"\n{'='*70}")
            print(f"SFT RUN {sft_run_count} RESULTS:")
            print(f"   • Learning Rate: {lr}")
            print(f"   • Batch Size: {bs}")
            print(f"   • Prior Task Score: {prior_scores['average']:.4f}")
            print(f"   • KL Divergence: {kl_div:.4f}")
            print(f"{'='*70}")
            
            # Delete model and trainer immediately after use
            print(f"\n Cleaning up model and trainer...")
            del sft_model
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
            
            print(f" Memory freed. GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print("\n" + "="*70)
    print("SFT HYPERPARAMETER SWEEP COMPLETE")
    print("="*70)
    
    # Delete base model before RL sweep
    print("\n" + "="*70)
    print("PREPARING FOR RL SWEEP")
    print("="*70)
    print("\n Deleting base model to free memory...")
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    print(" Base model deleted")
    
    # Reload base model for RL
    print(f"\n Reloading base model for RL experiments...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(" Base model reloaded")
    
    # RL sweep (as in Table 2)
    print("\n" + "="*70)
    print("RUNNING RL (GRPO) HYPERPARAMETER SWEEP")
    print("="*70)
    print(f"Learning rates to test: {rl_config['learning_rates']}")
    print("="*70 + "\n")
    
    rl_run_count = 0
    total_rl_runs = len(rl_config['learning_rates'])
    
    for lr in rl_config['learning_rates']:
        
        rl_run_count += 1
        print(f"\n{'*'*70}")
        print(f"RL RUN {rl_run_count}/{total_rl_runs}: lr={lr}")
        print(f"{'*'*70}")
        
        # Clear memory before loading new model
        print("\n Clearing GPU memory...")
        torch.cuda.empty_cache()
        gc.collect()
        print(f" Memory cleared. GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Clone base model
        print(f"\n Loading fresh model for this run...")
        rl_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(" Model loaded")
        
        # Train
        # rl_model, trainer = train_grpo(rl_model, dataset, tokenizer, learning_rate=lr)
        rl_model, trainer, NT = train_grpo(rl_model, dataset, tokenizer, learning_rate=lr)
        
        # Evaluate
        print(f"\n Evaluating trained RL model...")
        prior_scores = evaluate_benchmarks(rl_model, tokenizer)
        
        print(f"\n Computing KL divergence...")
        kl_div = compute_forward_kl(rl_model, base_model, formatted_dataset_kl, tokenizer)  # Use formatted dataset
        
        results['rl'].append({
            'lr': lr,
            'NT': NT, 
            'prior_task_score': prior_scores['average'],
            'kl_divergence': kl_div,
            'detailed_scores': prior_scores,
        })
        
        print(f"\n{'='*70}")
        print(f"RL RUN {rl_run_count} RESULTS:")
        print(f"   • Learning Rate: {lr}")
        print(f"   • Prior Task Score: {prior_scores['average']:.4f}")
        print(f"   • KL Divergence: {kl_div:.4f}")
        print(f"{'='*70}")
        
        # Delete model immediately
        print(f"\n Cleaning up model and trainer...")
        del rl_model
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        print(f" Memory freed. GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print("\n" + "="*70)
    print("RL HYPERPARAMETER SWEEP COMPLETE")
    print("="*70)
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    results_file = f'results_{dataset_name}.json'
    print(f"\n→ Writing results to {results_file}...")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f" Results saved to {results_file}")
    
    # Final cleanup
    print("\n Final cleanup...")
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    print(f" Cleanup complete. GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    return results