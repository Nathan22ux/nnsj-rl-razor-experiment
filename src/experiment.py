import os
import gc
import json
import torch
from transformers import AutoModelForCausalLM

from config import MODEL_NAME, sft_config, rl_config
from training import train_sft, train_grpo
from evaluation import evaluate_benchmarks, compute_forward_kl, evaluate_new_task


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
    
    def is_rl_config_done(lr):
        """Check if this RL configuration already exists in results"""
        for result in results.get('rl', []):
            if result['lr'] == lr:
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
                
                # Train
                # sft_model, trainer = train_sft(sft_model, dataset, tokenizer, learning_rate=lr, batch_size=bs)
                # Train from NT too
                sft_model, trainer, NT = train_sft(sft_model, dataset, tokenizer, learning_rate=lr, batch_size=bs, epochs=epochs)
                
                # Evaluate
                print(f"\n Evaluating trained SFT model...")
                prior_scores = evaluate_benchmarks(sft_model, tokenizer)
                
                print(f"\n Computing KL divergence...")
                kl_div = compute_forward_kl(sft_model, base_model, formatted_dataset_kl, tokenizer)
                
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
    
    # RL sweep (as in Table 2)
    print("\n" + "="*70)
    print("RUNNING RL (GRPO) HYPERPARAMETER SWEEP")
    print("="*70)
    print(f"Learning rates to test: {rl_config['learning_rates']}")
    print("="*70 + "\n")
    
    rl_run_count = 0
    total_rl_runs = len(rl_config['learning_rates'])
    completed_rl = len(results.get('rl', []))
    
    print(f"Progress: {completed_rl}/{total_rl_runs} RL models already completed")
    print(f"Remaining: {total_rl_runs - completed_rl} RL models to train\n")
    
    for lr in rl_config['learning_rates']:
        
        # Check if this configuration was already completed
        if is_rl_config_done(lr):
            print(f"\n{'='*70}")
            print(f" SKIPPING: RL lr={lr} (already completed)")
            print(f"{'='*70}")
            continue
        
        rl_run_count += 1
        print(f"\n{'*'*70}")
        print(f"RL RUN {completed_rl + rl_run_count}/{total_rl_runs}: lr={lr}")
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
        kl_div = compute_forward_kl(rl_model, base_model, formatted_dataset_kl, tokenizer)
        
        results['rl'].append({
            'lr': lr,
            'NT': NT, 
            'PT': prior_scores['average'],
            'kl_divergence': kl_div,
            'detailed_scores': prior_scores,
        })
        
        print(f"\n{'='*70}")
        print(f"RL RUN {completed_rl + rl_run_count} RESULTS:")
        print(f" Learning Rate: {lr}")
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


def run_full_experiment_with_circuit_regularization(
    dataset, 
    tokenizer, 
    dataset_name="math",
    circuit_analysis_path=None,
    lambda_reg_values=[0.01, 0.05, 0.1]
):
    """
    Extended experiment that includes circuit-aware SFT regularization.
    
    Args:
        dataset: Training dataset
        tokenizer: Model tokenizer
        dataset_name: Name of dataset being used
        circuit_analysis_path: Path to circuit analysis results JSON (from run_circuit_analysis.py)
                             If None, will skip circuit regularization runs
        lambda_reg_values: List of regularization strengths to test
    
    Returns:
        dict: Results including standard SFT, RL, and circuit-regularized SFT
    """
    from circuit.regularization import load_vulnerable_heads_from_analysis, train_sft_with_circuit_regularization
    
    # First run standard experiment
    results = run_full_experiment(dataset, tokenizer, dataset_name)
    
    # If no circuit analysis path provided, return standard results
    if circuit_analysis_path is None or not os.path.exists(circuit_analysis_path):
        print("\n" + "="*70)
        print("SKIPPING CIRCUIT-AWARE REGULARIZATION")
        print("="*70)
        print("No circuit analysis path provided or file not found.")
        print("Run circuit analysis first: python run_circuit_analysis.py")
        print("="*70 + "\n")
        return results
    
    # Initialize circuit-regularized SFT results
    if 'sft_circuit_reg' not in results:
        results['sft_circuit_reg'] = []
    
    print("\n" + "="*70)
    print("RUNNING CIRCUIT-AWARE REGULARIZATION EXPERIMENTS")
    print("="*70)
    
    # Load vulnerable heads from circuit analysis
    print(f"\nLoading vulnerable heads from: {circuit_analysis_path}")
    vulnerable_heads = load_vulnerable_heads_from_analysis(circuit_analysis_path)
    print(f"Loaded {len(vulnerable_heads)} vulnerable heads")
    
    # Format dataset for KL computation (reuse from main experiment)
    def format_dataset_for_kl(examples):
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
    
    formatted_dataset_kl = dataset.map(format_dataset_for_kl, batched=True, remove_columns=dataset.column_names)
    
    # Load base model
    print(f"\nLoading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Helper function to check if config already done
    def is_circuit_reg_config_done(lr, bs, epochs, lambda_reg):
        for result in results.get('sft_circuit_reg', []):
            if (result['lr'] == lr and 
                result['batch_size'] == bs and 
                result['epochs'] == epochs and
                result['lambda_reg'] == lambda_reg):
                return True
        return False
    
    # Helper function to save checkpoint
    results_file = os.path.join("results", f"results_{dataset_name}.json")
    def save_results_checkpoint():
        """Save results to JSON file (checkpoint)"""
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Checkpoint saved to {results_file}")
    
    # Run circuit-regularized SFT experiments
    # Use same hyperparameters as standard SFT
    circuit_reg_run_count = 0
    total_circuit_reg_runs = len(sft_config['learning_rates']) * len(sft_config['batch_sizes']) * len(sft_config['epochs']) * len(lambda_reg_values)
    completed_circuit_reg = len(results.get('sft_circuit_reg', []))
    
    print(f"\nProgress: {completed_circuit_reg}/{total_circuit_reg_runs} circuit-regularized models already completed")
    print(f"Remaining: {total_circuit_reg_runs - completed_circuit_reg} models to train\n")
    
    for lr in sft_config['learning_rates']:
        for bs in sft_config['batch_sizes']:
            for epochs in sft_config['epochs']:
                for lambda_reg in lambda_reg_values:
                    
                    # Check if already done
                    if is_circuit_reg_config_done(lr, bs, epochs, lambda_reg):
                        print(f"\n{'='*70}")
                        print(f" SKIPPING: lr={lr}, bs={bs}, epochs={epochs}, λ={lambda_reg} (already completed)")
                        print(f"{'='*70}")
                        continue
                    
                    circuit_reg_run_count += 1
                    print(f"\n{'*'*70}")
                    print(f"CIRCUIT-REGULARIZED SFT RUN {completed_circuit_reg + circuit_reg_run_count}/{total_circuit_reg_runs}")
                    print(f"lr={lr}, batch_size={bs}, epochs={epochs}, λ={lambda_reg}")
                    print(f"{'*'*70}")
                    
                    # Clear memory
                    print("\n Clearing GPU memory...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f" Memory cleared. GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                    
                    # Load fresh model
                    print(f"\n Loading fresh model for this run...")
                    sft_model = AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    )
                    print(" Model loaded")
                    
                    # Train with circuit regularization
                    print(f"\n Training with circuit-aware regularization...")
                    sft_model, trainer = train_sft_with_circuit_regularization(
                        model=sft_model,
                        base_model=base_model,
                        dataset=dataset,
                        tokenizer=tokenizer,
                        vulnerable_heads=vulnerable_heads,
                        lambda_reg=lambda_reg,
                        learning_rate=lr,
                        batch_size=bs,
                        epochs=epochs,
                        output_dir=f"./results/sft_circuit_reg_lr{lr}_bs{bs}_ep{epochs}_lam{lambda_reg}"
                    )
                    
                    # Evaluate
                    print(f"\n Evaluating circuit-regularized model...")
                    NT = evaluate_new_task(model=sft_model, tokenizer=tokenizer, dataset=dataset)
                    prior_scores = evaluate_benchmarks(sft_model, tokenizer)
                    kl_div = compute_forward_kl(sft_model, base_model, formatted_dataset_kl, tokenizer)
                    
                    results['sft_circuit_reg'].append({
                        'lr': lr,
                        'batch_size': bs,
                        'epochs': epochs,
                        'lambda_reg': lambda_reg,
                        'NT': NT,
                        'PT': prior_scores['average'],
                        'kl_divergence': kl_div,
                        'detailed_scores': prior_scores,
                    })
                    
                    print(f"\n{'='*70}")
                    print(f"CIRCUIT-REGULARIZED SFT RESULTS:")
                    print(f" Learning Rate: {lr}")
                    print(f" Batch Size: {bs}")
                    print(f" Epochs: {epochs}")
                    print(f" Lambda (λ): {lambda_reg}")
                    print(f" New Task (NT): {NT:.4f}")
                    print(f" Prior Task Score (PT): {prior_scores['average']:.4f}")
                    print(f" KL Divergence: {kl_div:.4f}")
                    print(f"{'='*70}")
                    
                    # Save checkpoint
                    save_results_checkpoint()
                    
                    # Cleanup
                    print(f"\n Cleaning up model and trainer...")
                    del sft_model
                    del trainer
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f" Memory freed. GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print("\n" + "="*70)
    print("CIRCUIT-AWARE REGULARIZATION EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Total circuit-regularized models trained: {len(results['sft_circuit_reg'])}")
    print("="*70)
    
    # Cleanup base model
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return results