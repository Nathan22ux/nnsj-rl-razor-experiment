import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Evaluation benchmarks from Section 3 in paper
# EVAL_BENCHMARKS = [
#     # "hellaswag",      # Zellers et al., 2019
#     # "truthfulqa_mc2", # Lin et al., 2021
#     # "mmlu",           # Hendrycks et al., 2020
#     # "ifeval",         # Zhou et al., 2023
#     # "winogrande",     # Sakaguchi et al., 2021
#     # "humaneval",      # Chen et al., 2021
#     # Commented out to reduce the computation amount on evaluation
#     "winogrande",
#     "hellaswag",
#     "mmlu_high_school_mathematics",
#     "mmlu_high_school_computer_science"
# ]

EVAL_BENCHMARKS = [
    "winogrande",
    "hellaswag",
    "mmlu_high_school_mathematics",
    "mmlu_high_school_computer_science"
]

def evaluate_benchmarks(model, tokenizer, tasks=EVAL_BENCHMARKS, limit=50):
    """
    Evaluate model on standard benchmarks to measure prior task performance.
    
    Based on RL's Razor paper Section 3.1 (Performance Trade-offs):
    "We measure performance on a diverse set of unrelated benchmarks.
    A drop in these benchmarks is taken as a measure of catastrophic forgetting."
    
    This function measures:
    - Performance on domains NOT used during fine-tuning (WinoGrande, HellaSwag, MMLU)
    - This assesses whether the model retained its original knowledge
    
    Args:
        model: Model to evaluate (should be fine-tuned model)
        tokenizer: Tokenizer for the model
        tasks: List of benchmark tasks to evaluate on
        limit: Maximum number of examples per task (for GPU memory constraints)
        
    Returns:
        dict: Dictionary with:
            - Per-task accuracy scores (e.g., 'winogrande': 0.65)
            - 'average': Overall average accuracy across tasks
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    
    print(f"\n{'='*70}")
    print(f"EVALUATING MODEL ON BENCHMARK TASKS")
    print(f"{'='*70}")
    print(f"Purpose: Measure prior knowledge retention (catastrophic forgetting)")
    print(f"\nBenchmark tasks:")
    for task in tasks:
        print(f"    {task}")
    print(f"\nEvaluation settings:")
    print(f"  Few-shot examples: 0 (zero-shot evaluation)")
    print(f"  Limit per task: {limit} examples")
    print(f"  Metric: Accuracy (or normalized accuracy)")
    print(f"{'='*70}\n")
    
    print(" Wrapping model for lm-eval harness...")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device="cuda",
        max_length=1024
    )
    print(" Model wrapped successfully\n")
    
    max_gen_toks = 128
    
    print(f" Running evaluation across {len(tasks)} tasks...")
    print(f" This may take several minutes depending on task complexity")
    print(f" Status: Starting evaluation loop...\n")
    
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        limit=limit,
        confirm_run_unsafe_code=True,
        gen_kwargs={"max_new_tokens": max_gen_toks}
    )
    
    print("\n Evaluation complete!")
    
    # Extract accuracy scores from results
    print("\n Extracting scores from evaluation results...")
    scores = {}
    
    for task in tasks:
        if task in results['results']:
            task_result = results['results'][task]
            
            # Try different metric names (varies by benchmark)
            if 'acc' in task_result:
                scores[task] = task_result['acc']
            elif 'acc_norm' in task_result:
                # Normalized accuracy (used in MMLU)
                scores[task] = task_result['acc_norm']
            elif task == 'ifeval' and 'accuracy' in task_result:
                # IFEval uses 'accuracy' instead of 'acc'
                scores[task] = task_result['accuracy']
            else:
                # Fallback: try to extract any numeric score
                for key, value in task_result.items():
                    if isinstance(value, (int, float)) and key not in ['samples', 'batch_size']:
                        scores[task] = float(value)
                        break
    
    # Compute average accuracy across all tasks
    scores['average'] = np.mean(list(scores.values())) if scores else 0.0
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Per-task accuracy (prior knowledge retention):")
    for task, score in scores.items():
        if task != 'average':
            print(f"    {task:40s}: {score:.4f}")
    print(f"\n   {'Average Prior Task Performance':40s}: {scores['average']:.4f}")
    print(f"{'='*70}\n")
    
    return scores


def compute_forward_kl(model, base_model, dataset, tokenizer, num_samples=50):
    """
    Compute forward KL divergence KL(base || model).
    This measures how much the model has diverged from the base model.
    
    Based on RL's Razor paper Section 4 (Smaller KL divergences lead to less forgetting):
    "We uncover an empirical forgetting law: the KL divergence between the 
    fine-tuned model and the base model, measured on the new task, reliably 
    predicts the degree of forgetting."
    
    Formula: KL(P_base || P_model) = Σ P_base * log(P_base / P_model)
    
    Args:
        model: Fine-tuned model
        base_model: Base model (reference for comparison)
        dataset: Dataset to compute KL on (should be formatted with 'text' field)
        tokenizer: Tokenizer for the models
        num_samples: Number of samples to use for computation
        
    Returns:
        float: Average KL divergence across samples
    """
    print(f"\n{'='*70}")
    print(f"COMPUTING KL DIVERGENCE (Forward KL)")
    print(f"{'='*70}")
    print(f"Formula: KL(P_base || P_model) = Σ P_base * log(P_base / P_model)")
    print(f"This measures model divergence from base (catastrophic forgetting)")
    print(f"\nBase model: Reference checkpoint")
    print(f"Fine-tuned model: After training")
    print(f"Dataset samples: {num_samples}")
    print(f"{'='*70}\n")
    
    print(" Setting models to evaluation mode...")
    model.eval()
    base_model.eval()
    print(" Models in evaluation mode")
    
    # Ensure models are on same device
    model_device = next(model.parameters()).device
    base_device = next(base_model.parameters()).device
    print(f" Model device: {model_device}")
    print(f" Base model device: {base_device}")
    
    total_kl = 0.0
    count = 0
    kl_per_sample = []
    
    print(f"\n Selecting up to {num_samples} random samples from dataset...")
    num_to_sample = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_to_sample, replace=False)
    print(f" Selected {len(indices)} samples for KL computation")
    
    print("\n Computing KL divergence for each sample...")
    print("   (Progress bar will appear below)")
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Computing KL"):
            # Get text sample
            text = dataset[int(idx)]['text']

            # Tokenize text
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False
            )

            # Move to device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            # Move base model inputs to base model device if different
            if model_device != base_device:
                base_inputs = {k: v.to(base_device) for k, v in inputs.items()}
            else:
                base_inputs = inputs

            # Get logits from both models with autocast for speed
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                base_logits = base_model(**base_inputs).logits  # Shape: [1, seq_len, vocab_size]
                model_logits = model(**inputs).logits  # Shape: [1, seq_len, vocab_size]
            
            # Use log_softmax for numerical stability (better than softmax + log)
            base_log_probs = F.log_softmax(base_logits, dim=-1)  # [1, seq_len, vocab_size]
            model_log_probs = F.log_softmax(model_logits, dim=-1)  # [1, seq_len, vocab_size]
            
            # Convert log probs to probs for base model only (need actual probabilities for weighting)
            base_probs = torch.exp(base_log_probs)  # [1, seq_len, vocab_size]
            
            # Handle sequence length mismatch
            min_len = min(base_log_probs.size(1), model_log_probs.size(1))
            base_log_probs = base_log_probs[:, :min_len, :]  # [1, min_len, vocab_size]
            model_log_probs = model_log_probs[:, :min_len, :]  # [1, min_len, vocab_size]
            base_probs = base_probs[:, :min_len, :]  # [1, min_len, vocab_size]
            
            # Compute KL divergence per token, then average across sequence
            # KL(P || Q) = Σ P * (log(P) - log(Q))
            kl_per_token = base_probs * (base_log_probs - model_log_probs)  # [1, min_len, vocab_size]
            kl_sum = kl_per_token.sum(dim=-1)  # [1, min_len] - sum over vocab
            kl_mean = kl_sum.mean(dim=1)  # [1] - average over sequence
            kl_value = kl_mean.item()
            
            total_kl += kl_value
            kl_per_sample.append(kl_value)
            count += 1
    
    # Compute statistics
    avg_kl = total_kl / count if count > 0 else 0.0
    std_kl = np.std(kl_per_sample) if len(kl_per_sample) > 1 else 0.0
    min_kl = np.min(kl_per_sample) if kl_per_sample else 0.0
    max_kl = np.max(kl_per_sample) if kl_per_sample else 0.0
    
    print(f"\n KL Divergence Computation Complete!")
    print(f"{'='*70}")
    print(f"KL Statistics:")
    print(f"   Mean KL Divergence: {avg_kl:.6f}")
    print(f"   Std Dev: {std_kl:.6f}")
    print(f"   Min: {min_kl:.6f}")
    print(f"   Max: {max_kl:.6f}")
    print(f"   Samples: {count}")
    print(f"{'='*70}\n")
    
    return avg_kl

def evaluate_new_task(model, tokenizer, dataset, max_new_tokens = 64, num_samples=50):
    """
    Evaluate New Task performance (NT)
    This function computes accuracy on the new-task dataset
    used for fine-tuning, following the RL's Razor setup.

    Args:
        Tokenizer: model tokenizer
        model: fine-tuned model
        max_new_tokens: max tokens to be generated
        dataset: list of dicts {input, label}

    returns:
        final accuracy on new task
    """

    model.eval()
    correct = 0
    total = len(dataset)
    print(f"length of dataset: {total}")

    print("---------Evaluating NEW TASK -----------------")
    eval_size = min(num_samples, len(dataset))
    for i in tqdm(range(eval_size), desc="New Task Evaluation"):
        sample = dataset[i]
        
        # Extract from correct fields
        question = sample["0"]["value"]
        try:
            answer = sample["1"]["ground_truth"]["value"]
        except (KeyError, TypeError):
            answer = str(sample["1"])
        
        prompt = f"Question: {question}\nAnswer:"
        expect_output = str(answer).strip().lower()

        inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)

        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                do_sample = False,
                pad_token_id=tokenizer.eos_token_id
            )

        # decoding output
        predictions = tokenizer.decode(outputs[0], skip_special_tokens = True)
        predictions = predictions.strip().lower()
        print(f"The prediction is: {predictions}")

        if expect_output in predictions:
            correct +=1
    
    acc = (correct / total) * 100
    print(f"The New Task accuracy achieved is : {acc:.4f}\n")
    return acc


def compute_retention_benefit_metrics(results_dict, baseline_pt=None):
    """
    Compute retention benefit metrics to verify "80-90% of RL's retention benefit" claim.
    
    This function compares:
    - Standard SFT: PT (prior task) score
    - RL: PT score  
    - SFT + Circuit Regularization: PT score
    
    Retention benefit is measured as improvement in PT over baseline.
    The goal is to show that circuit-regularized SFT achieves 80-90% of RL's retention benefit.
    
    Formula:
    - Baseline PT: PT score of standard SFT (lowest retention)
    - RL retention benefit = RL_PT - Baseline_PT
    - Circuit-Reg retention benefit = CircuitReg_PT - Baseline_PT
    - Retention benefit percentage = (CircuitReg_Benefit / RL_Benefit) * 100
    
    Args:
        results_dict: Dictionary with keys:
            - 'sft': List of SFT results (each with 'PT' key)
            - 'rl': List of RL results (each with 'PT' key)
            - 'sft_circuit_reg': List of circuit-regularized SFT results (each with 'PT' key)
        baseline_pt: Optional baseline PT score (if None, uses average of standard SFT)
    
    Returns:
        dict: Dictionary containing:
            - 'baseline_pt': Baseline PT score (standard SFT average)
            - 'rl_pt': RL PT score
            - 'rl_retention_benefit': Improvement over baseline from RL
            - 'circuit_reg_results': List of results for each circuit-reg configuration
            - 'best_circuit_reg': Best circuit-reg configuration
            - 'retention_benefit_percentage': Percentage of RL's benefit achieved
            - 'summary': Human-readable summary string
    """
    print("\n" + "="*70)
    print("COMPUTING RETENTION BENEFIT METRICS")
    print("="*70)
    
    # Get baseline PT (average of standard SFT)
    if baseline_pt is None:
        sft_results = results_dict.get('sft', [])
        if not sft_results:
            raise ValueError("No SFT results found. Cannot compute baseline.")
        baseline_pt = np.mean([r['PT'] for r in sft_results])
        print(f"\nBaseline PT (Standard SFT average): {baseline_pt:.4f}")
    else:
        print(f"\nBaseline PT (provided): {baseline_pt:.4f}")
    
    # Get RL PT (average across RL runs)
    rl_results = results_dict.get('rl', [])
    if not rl_results:
        raise ValueError("No RL results found. Cannot compute RL retention benefit.")
    rl_pt = np.mean([r['PT'] for r in rl_results])
    rl_retention_benefit = rl_pt - baseline_pt
    
    print(f"RL PT (average): {rl_pt:.4f}")
    print(f"RL Retention Benefit: {rl_retention_benefit:.4f} (improvement over baseline)")
    
    # Analyze circuit-regularized SFT results
    circuit_reg_results = results_dict.get('sft_circuit_reg', [])
    if not circuit_reg_results:
        print("\n⚠️  No circuit-regularized SFT results found.")
        print("   Run circuit-aware regularization experiments first.")
        return {
            'baseline_pt': baseline_pt,
            'rl_pt': rl_pt,
            'rl_retention_benefit': rl_retention_benefit,
            'circuit_reg_results': [],
            'best_circuit_reg': None,
            'retention_benefit_percentage': None,
            'summary': "No circuit-regularized results to compare"
        }
    
    print(f"\nAnalyzing {len(circuit_reg_results)} circuit-regularized SFT configurations...")
    
    # Compute retention benefit for each circuit-reg configuration
    analyzed_results = []
    for result in circuit_reg_results:
        circuit_pt = result['PT']
        circuit_retention_benefit = circuit_pt - baseline_pt
        retention_percentage = (circuit_retention_benefit / rl_retention_benefit * 100) if rl_retention_benefit > 0 else 0
        
        analyzed_result = {
            **result,
            'retention_benefit': circuit_retention_benefit,
            'retention_benefit_percentage': retention_percentage
        }
        analyzed_results.append(analyzed_result)
    
    # Find best configuration (highest retention benefit percentage)
    best_result = max(analyzed_results, key=lambda x: x['retention_benefit_percentage'])
    
    print(f"\n{'='*70}")
    print("RETENTION BENEFIT ANALYSIS RESULTS")
    print(f"{'='*70}")
    print(f"\nBaseline (Standard SFT): PT = {baseline_pt:.4f}")
    print(f"RL: PT = {rl_pt:.4f}, Benefit = {rl_retention_benefit:.4f}")
    print(f"\nBest Circuit-Regularized SFT Configuration:")
    print(f"  Learning Rate: {best_result['lr']}")
    print(f"  Batch Size: {best_result['batch_size']}")
    print(f"  Epochs: {best_result['epochs']}")
    print(f"  Lambda (λ): {best_result['lambda_reg']}")
    print(f"  PT Score: {best_result['PT']:.4f}")
    print(f"  Retention Benefit: {best_result['retention_benefit']:.4f}")
    print(f"  Percentage of RL's Benefit: {best_result['retention_benefit_percentage']:.2f}%")
    print(f"{'='*70}\n")
    
    # Create summary
    if best_result['retention_benefit_percentage'] >= 80:
        status = "✅ ACHIEVED TARGET (≥80%)"
    elif best_result['retention_benefit_percentage'] >= 70:
        status = "⚠️  CLOSE TO TARGET (70-79%)"
    else:
        status = "❌ BELOW TARGET (<70%)"
    
    summary = f"""
RETENTION BENEFIT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Goal: Circuit-regularized SFT should achieve 80-90% of RL's retention benefit

Baseline (Standard SFT):     PT = {baseline_pt:.4f}
RL (Reference):              PT = {rl_pt:.4f}, Benefit = +{rl_retention_benefit:.4f}

Best Circuit-Regularized SFT:
  Configuration: lr={best_result['lr']}, bs={best_result['batch_size']}, 
                 epochs={best_result['epochs']}, λ={best_result['lambda_reg']}
  PT Score: {best_result['PT']:.4f}
  Retention Benefit: +{best_result['retention_benefit']:.4f}
  Percentage of RL's Benefit: {best_result['retention_benefit_percentage']:.2f}%

Status: {status}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    print(summary)
    
    return {
        'baseline_pt': baseline_pt,
        'rl_pt': rl_pt,
        'rl_retention_benefit': rl_retention_benefit,
        'circuit_reg_results': analyzed_results,
        'best_circuit_reg': best_result,
        'retention_benefit_percentage': best_result['retention_benefit_percentage'],
        'summary': summary
    }


def compare_all_methods(results_dict, output_path=None):
    """
    Compare all methods (Standard SFT, RL, Circuit-Regularized SFT) side-by-side.
    
    Args:
        results_dict: Results dictionary from experiment
        output_path: Optional path to save comparison table
    
    Returns:
        dict: Comparison results
    """
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not available, skipping DataFrame creation")
        pd = None
    
    print("\n" + "="*70)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("="*70)
    
    comparison_data = []
    
    # Standard SFT
    sft_results = results_dict.get('sft', [])
    if sft_results:
        avg_sft_pt = np.mean([r['PT'] for r in sft_results])
        avg_sft_nt = np.mean([r['NT'] for r in sft_results])
        avg_sft_kl = np.mean([r['kl_divergence'] for r in sft_results])
        comparison_data.append({
            'Method': 'Standard SFT',
            'PT (Prior Task)': avg_sft_pt,
            'NT (New Task)': avg_sft_nt,
            'KL Divergence': avg_sft_kl
        })
    
    # RL
    rl_results = results_dict.get('rl', [])
    if rl_results:
        avg_rl_pt = np.mean([r['PT'] for r in rl_results])
        avg_rl_nt = np.mean([r['NT'] for r in rl_results])
        avg_rl_kl = np.mean([r['kl_divergence'] for r in rl_results])
        comparison_data.append({
            'Method': 'RL (GRPO)',
            'PT (Prior Task)': avg_rl_pt,
            'NT (New Task)': avg_rl_nt,
            'KL Divergence': avg_rl_kl
        })
    
    # Best Circuit-Regularized SFT
    circuit_reg_results = results_dict.get('sft_circuit_reg', [])
    if circuit_reg_results:
        # Use the best one (highest PT)
        best_circuit_reg = max(circuit_reg_results, key=lambda x: x['PT'])
        comparison_data.append({
            'Method': f"Circuit-Reg SFT (λ={best_circuit_reg['lambda_reg']})",
            'PT (Prior Task)': best_circuit_reg['PT'],
            'NT (New Task)': best_circuit_reg['NT'],
            'KL Divergence': best_circuit_reg['kl_divergence']
        })
    
    # Create DataFrame if pandas available, otherwise use list
    if pd is not None:
        df = pd.DataFrame(comparison_data)
        print("\nComparison Table:")
        print(df.to_string(index=False))
        df_dict = df.to_dict('records')
    else:
        print("\nComparison Table:")
        for item in comparison_data:
            print(f"Method: {item['Method']}")
            print(f"  PT: {item['PT (Prior Task)']:.4f}")
            print(f"  NT: {item['NT (New Task)']:.4f}")
            print(f"  KL: {item['KL Divergence']:.4f}\n")
        df_dict = comparison_data
    
    # Compute retention benefit metrics
    retention_metrics = compute_retention_benefit_metrics(results_dict)
    
    # Save if path provided
    if output_path:
        import json
        comparison_results = {
            'comparison_table': df_dict,
            'retention_metrics': {
                'baseline_pt': retention_metrics['baseline_pt'],
                'rl_pt': retention_metrics['rl_pt'],
                'rl_retention_benefit': retention_metrics['rl_retention_benefit'],
                'best_retention_percentage': retention_metrics['retention_benefit_percentage'],
                'best_config': retention_metrics['best_circuit_reg']
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        print(f"\nComparison saved to: {output_path}")
    
    return {
        'comparison_table': df_dict if pd is None else df,
        'retention_metrics': retention_metrics
    }