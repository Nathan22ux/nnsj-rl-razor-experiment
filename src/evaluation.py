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
    # "humaneval" 
]

def evaluate_benchmarks(model, tokenizer, tasks=EVAL_BENCHMARKS, limit=300):
    """
    Evaluate model on standard benchmarks.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        tasks: List of benchmark tasks to evaluate on
        limit: Maximum number of examples per task
        
    Returns:
        dict: Dictionary of scores per task and average
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    
    print(f"\n{'='*70}")
    print(f"EVALUATING MODEL ON BENCHMARKS")
    print(f"{'='*70}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Limit per task: {limit} examples")
    print(f"{'='*70}\n")
    
    print(" Wrapping model for lm-eval-harness...")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device="cuda",
        max_length=2048
    )
    print("Model wrapped successfully")
    
    max_gen_toks = 256
    
    print(f"\n Running evaluation (this may take several minutes)...")
    print(f" Few-shot examples: 0")
    print(f" Max generation tokens: {max_gen_toks}")
    print(f" Starting evaluation...")
    
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        limit=limit,
        confirm_run_unsafe_code=True,
        gen_kwargs={"max_new_tokens": max_gen_toks}
    )
    
    print("\n Evaluation complete!")
    
    # Extract accuracy scores
    print("\n Extracting scores from results...")
    scores = {}
    for task in tasks:
        if task in results['results']:
            task_result = results['results'][task]
            if 'acc' in task_result:
                scores[task] = task_result['acc']
            elif 'acc_norm' in task_result:
                scores[task] = task_result['acc_norm']
            elif task == 'ifeval' and 'accuracy' in task_result:
                scores[task] = task_result['accuracy']
    
    scores['average'] = np.mean(list(scores.values()))
    
    print("\n Evaluation Results:")
    for task, score in scores.items():
        if task != 'average':
            print(f"   • {task}: {score:.4f}")
    print(f"   • Average: {scores['average']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK EVALUATION COMPLETE")
    print(f"{'='*70}\n")
    
    return scores


def compute_forward_kl(model, base_model, dataset, tokenizer, num_samples=100):
    """
    Compute forward KL divergence KL(base || model).
    This measures how much the model has diverged from the base model.
    
    Args:
        model: Fine-tuned model
        base_model: Base model (reference)
        dataset: Dataset to compute KL on
        tokenizer: Tokenizer for the models
        num_samples: Number of samples to use for computation
        
    Returns:
        float: Average KL divergence
    """
    print(f"\n{'='*70}")
    print(f"COMPUTING KL DIVERGENCE")
    print(f"{'='*70}")
    print(f"Comparing fine-tuned model against base model")
    print(f"Number of samples: {num_samples}")
    print(f"{'='*70}\n")
    
    print(" Setting models to evaluation mode...")
    model.eval()
    base_model.eval()
    print(" Models in evaluation mode")
    
    total_kl = 0.0
    count = 0
    
    print(f"\n Selecting {num_samples} random samples from dataset...")
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    print(f" Selected {len(indices)} samples")
    
    print("\n Computing KL divergence for each sample...")
    print("   (Progress bar will appear below)")
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Computing KL"):
            text = dataset[int(idx)]['text']
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Get logits
            base_logits = base_model(**inputs).logits
            model_logits = model(**inputs).logits
            
            # Convert to probabilities
            base_probs = F.softmax(base_logits, dim=-1)
            model_probs = F.softmax(model_logits, dim=-1)
            
            # KL(base || model) - forward KL
            min_len = min(base_probs.size(1), model_probs.size(1))
            base_probs = base_probs[:, :min_len, :]
            model_probs = model_probs[:, :min_len, :]
            
            kl = (base_probs * (torch.log(base_probs + 1e-10) - torch.log(model_probs + 1e-10))).sum()
            
            total_kl += kl.item()
            count += 1
    
    avg_kl = total_kl / count
    
    print(f"\n KL divergence computation complete")
    print(f"Average KL Divergence: {avg_kl:.4f}")
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