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
    "hellaswag",
    "truthfulqa_mc2",
    "mmlu",
    "ifeval",
    "winogrande",
    "humaneval" 
]

def evaluate_benchmarks(model, tokenizer, tasks=EVAL_BENCHMARKS, limit=300):
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
        max_length=2048
    )
    print(" Model wrapped successfully\n")
    
    max_gen_toks = 256
    
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


def compute_forward_kl(model, base_model, dataset, tokenizer, num_samples=100):
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
            
            # Get logits from both models
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

def evaluate_new_task(model, tokenizer, dataset, max_new_tokens = 64):
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
    for sample in tqdm(dataset, desc="New Task Evaluation"):
        prompt = sample["input"]
        expect_output = str(sample['label']).strip().lower()

        inputs = tokenizer(prompt, return_tensor = "pt").to(model.device)

        with torch.no_grad():
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