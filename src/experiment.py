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
from config import sft_config, rl_config, data_config
from training import train_sft, train_grpo
from evaluation import evaluate_benchmarks, compute_forward_kl



def run_full_experiment(dataset, tokenizer, dataset_name="math"):
    """
    Run full experiment to create Pareto frontier (Figure 2)
    With checkpoint/resume funcs, it saves after each model.

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
    print(f"Max training samples: {data_config['max_samples']}")
    print(f"KL samples: {data_config['kl_samples']}")
    print("="*70 + "\n")

    # Check for existing results to resume from
    os.makedirs("results", exist_ok=True)
    results_file = os.path.join("results", f"results_{dataset_name}.json")

    if os.path.exists(results_file):
        print("\n" + "="*70)
        print("FOUND EXISTING RESULTS - RESUMING FROM CHECKPOINT")
        print("="*70)
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results.get('sft', []))} completed SFT runs")
        print(f"Loaded {len(results.get('rl', []))} completed RL runs")
        print("="*70 + "\n")
    else:
        print("\nNo existing results found - starting fresh")
        results = {
            'sft': [],
            'rl': [],
        }

    # Set memory management
    print("\nConfiguring memory management...")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(" Memory management configured")

    # Clear GPU memory before starting
    print("\n Clearing GPU memory before starting...")
    torch.cuda.empty_cache()
    gc.collect()
    print(" GPU memory cleared")

    # FORMAT DATASET ONCE HERE
    print("\n Formatting dataset for KL computation...")
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

    # Helper funcs for checkpoint/resume
    def is_sft_config_done(lr, bs, epochs):
        """Check if this SFT configuration already exists in results"""
        for result in results.get('sft', []):
            if (result['lr'] == lr and
                    result['batch_size'] == bs and
                    result['epochs'] == epochs):
                return True
        return False

    # FIX: Updated to check batch_size too
    def is_rl_config_done(lr, bs):
        """Check if this RL configuration already exists in results"""
        for result in results.get('rl', []):
            if result['lr'] == lr and result.get('batch_size') == bs:
                return True
        return False

    def save_results_checkpoint():
        """Save results to JSON file (checkpoint)"""
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Checkpoint saved to {results_file}")

    # SFT sweep (as in Table 2)
    print("\n" + "="*70)
    print("RUNNING SFT HYPERPARAMETER SWEEP")
    print("="*70)
    print(f"Learning rates to test: {sft_config['learning_rates']}")
    print(f"Batch sizes to test: {sft_config['batch_sizes']}")
    print(f"Epochs to test: {sft_config['epochs']}")
    print(f"Max samples per run: {data_config['max_samples']}")
    print("="*70 + "\n")

    sft_run_count = 0
    total_sft_runs = len(sft_config['learning_rates']) * len(sft_config['batch_sizes']) * len(sft_config['epochs'])
    completed_sft = len(results.get('sft', []))

    print(f"Progress: {completed_sft}/{total_sft_runs} SFT models already completed")
    print(f"Remaining: {total_sft_runs - completed_sft} SFT models to train\n")

    for lr in sft_config['learning_rates']:
        for bs in sft_config['batch_sizes']:
            for epochs in sft_config['epochs']:

                # Check if this configuration was already completed
                if is_sft_config_done(lr, bs, epochs):
                    print(f"\n{'='*70}")
                    print(f" SKIPPING: lr={lr}, bs={bs}, epochs={epochs} (already completed)")
                    print(f"{'='*70}")
                    continue

                sft_run_count += 1
                print(f"\n{'*'*70}")
                print(f"SFT RUN {completed_sft + sft_run_count}/{total_sft_runs}: lr={lr}, batch_size={bs}, epochs={epochs}")
                print(f"{'*'*70}")

                # Clear memory before loading new model
                print("\n Clearing GPU memory...")
                torch.cuda.empty_cache()
                gc.collect()
                print(f" Memory cleared. GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

                print(f"\n Loading fresh model for this run...")

                # Clone base model
                sft_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                print(" Model loaded")

                # FIX: Pass max_samples from data_config
                sft_model, trainer, NT = train_sft(
                    sft_model, dataset, tokenizer,
                    learning_rate=lr,
                    batch_size=bs,
                    epochs=epochs,
                    max_samples=data_config['max_samples']
                )

                # Evaluate
                print(f"\n Evaluating trained SFT model...")
                prior_scores = evaluate_benchmarks(sft_model, tokenizer)

                # FIX: Use response_only=True and num_samples from config
                print(f"\n Computing KL divergence (response-only)...")
                kl_div = compute_forward_kl(
                    sft_model, base_model, formatted_dataset_kl, tokenizer,
                    num_samples=data_config['kl_samples'],
                    response_only=True
                )

                results['sft'].append({
                    'lr': lr,
                    'batch_size': bs,
                    'epochs': epochs,
                    'NT': NT,
                    'PT': prior_scores['average'],
                    'kl_divergence': kl_div,
                    'detailed_scores': prior_scores,
                })

                print(f"\n{'='*70}")
                print(f"SFT RUN {completed_sft + sft_run_count} RESULTS:")
                print(f" Learning Rate: {lr}")
                print(f" Batch Size: {bs}")
                print(f" Epochs: {epochs}")
                print(f" New Task (NT): {NT:.4f}")
                print(f" Prior Task Score (PT): {prior_scores['average']:.4f}")
                print(f" KL Divergence: {kl_div:.4f}")
                print(f"{'='*70}")

                # SAVE CHECKPOINT after each run
                print(f"\nSaving checkpoint...")
                save_results_checkpoint()

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
    print(f"Total SFT models trained: {len(results['sft'])}")
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

    # RL sweep - FIX: Also sweep batch sizes for fair comparison
    print("\n" + "="*70)
    print("RUNNING RL (GRPO) HYPERPARAMETER SWEEP")
    print("="*70)
    print(f"Learning rates to test: {rl_config['learning_rates']}")
    print(f"Batch sizes to test: {rl_config['batch_sizes']}")
    print(f"Max samples per run: {data_config['max_samples']}")
    print("="*70 + "\n")

    rl_run_count = 0
    # FIX: Include batch_sizes in total count
    total_rl_runs = len(rl_config['learning_rates']) * len(rl_config['batch_sizes'])
    completed_rl = len(results.get('rl', []))

    print(f"Progress: {completed_rl}/{total_rl_runs} RL models already completed")
    print(f"Remaining: {total_rl_runs - completed_rl} RL models to train\n")

    # FIX: Add batch_size loop for fair comparison with SFT
    for lr in rl_config['learning_rates']:
        for bs in rl_config['batch_sizes']:

            # Check if this configuration was already completed
            if is_rl_config_done(lr, bs):
                print(f"\n{'='*70}")
                print(f" SKIPPING: RL lr={lr}, bs={bs} (already completed)")
                print(f"{'='*70}")
                continue

            rl_run_count += 1
            print(f"\n{'*'*70}")
            print(f"RL RUN {completed_rl + rl_run_count}/{total_rl_runs}: lr={lr}, bs={bs}")
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

            # FIX: Pass batch_size and max_samples
            rl_model, trainer, NT = train_grpo(
                rl_model, dataset, tokenizer,
                learning_rate=lr,
                batch_size=bs,
                max_samples=data_config['max_samples']
            )

            # Evaluate
            print(f"\n Evaluating trained RL model...")
            prior_scores = evaluate_benchmarks(rl_model, tokenizer)

            # FIX: Use response_only=True and num_samples from config
            print(f"\n Computing KL divergence (response-only)...")
            kl_div = compute_forward_kl(
                rl_model, base_model, formatted_dataset_kl, tokenizer,
                num_samples=data_config['kl_samples'],
                response_only=True
            )

            # FIX: Include batch_size in results
            results['rl'].append({
                'lr': lr,
                'batch_size': bs,
                'NT': NT,
                'PT': prior_scores['average'],
                'kl_divergence': kl_div,
                'detailed_scores': prior_scores,
            })

            print(f"\n{'='*70}")
            print(f"RL RUN {completed_rl + rl_run_count} RESULTS:")
            print(f" Learning Rate: {lr}")
            print(f" Batch Size: {bs}")
            print(f" New Task (NT): {NT:.4f}")
            print(f" Prior Task Score (PT): {prior_scores['average']:.4f}")
            print(f" KL Divergence: {kl_div:.4f}")
            print(f"{'='*70}")

            # SAVE CHECKPOINT after each run
            print(f"\nSaving checkpoint...")
            save_results_checkpoint()

            # Delete model immediately
            print(f"\nCleaning up model and trainer...")
            del rl_model
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Memory freed. GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("\n" + "="*70)
    print("RL HYPERPARAMETER SWEEP COMPLETE")
    print("="*70)
    print(f"Total RL models trained: {len(results['rl'])}")
    print("="*70)

    # Save results (final save, but already saved incrementally)
    print("\n" + "="*70)
    print("FINAL SAVE OF RESULTS")
    print("="*70)
    print(f"\nWriting final results to {results_file}...")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Final results saved")
    print(f" SFT runs: {len(results['sft'])}")
    print(f" RL runs: {len(results['rl'])}")
    print(f" Total: {len(results['sft']) + len(results['rl'])} models")
    print(f"{'='*70}")

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