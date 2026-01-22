import gc
import os
import torch
import logging

import torch.nn.functional as F
from torch.optim import AdamW
from copy import deepcopy
from transformers import get_scheduler

from data.dataset_utils import UnifiedDatasetInterface

from trainingv1.rollout import generate_group_samples
from trainingv1.advantages import compute_group_advantages
from trainingv1.dr_loss import dr_grpo_loss

logger = logging.getLogger(__name__)

@torch.no_grad()
def evaluate_nt(model, tokenizer, eval_dataset, num_samples=500):
    from evaluation.evaluation import evaluate_new_task
    return evaluate_new_task(
        model=model,
        tokenizer=tokenizer,
        dataset=eval_dataset,
        num_samples=num_samples
    )

def train_dr_grpo(
    model,
    tokenizer,
    dataset,
    eval_dataset,
    μ_iterations=2,
    lr=2e-5,
    group_size=64,
    prompts_per_gen=8,
    target_nt=None,
    max_samples=3000,
):
    """
    Dr.GRPO  training implementation for RL's Razor.

        Paper mechanism:
        ------------------------------------------------------
        π₀ = SFT baseline
        π₁ = Dr.GRPO(π₀)    (μ=1)
        π₂ = Dr.GRPO(π₁)    (μ=2)

        For each μ:
            sample groups → compute binary reward → compute rank-normalized A →
            optimize L = − E_group[A_i * log π(y_i|x)]

        Group sampling:
            group_size = 64
            prompts_per_gen = 8

        No explicit KL regularization.
        KL is implicitly minimized by relative group loss.
    """

    logger.info("=" * 70)
    logger.info("INITIALIZING PURE Dr.GRPO TRAINING.")
    logger.info("=" * 70)

    logger.info(f"Current Learning Rate : {lr}, Group Size : {group_size}, Prompts per Generation : {prompts_per_gen}, Max Samples : {max_samples}, μ Iterations : {μ_iterations}")
    logger.info(f"Max samples: {max_samples}")
    logger.info(f"Target NT: {target_nt if target_nt else 'None'}")

    dataset = UnifiedDatasetInterface.normalize_dataset(dataset)
    dataset = dataset.select(range(min(max_samples, len(dataset))))

     # extract prompts + ground truths for reward
    prompts = dataset["prompt"]
    answers = dataset["answer"]
    logger.info(f"Dataset loaded with {len(prompts)} prompts")

    # === μ iteration refinement ===
    current_model = model
    π_models = [current_model]  # π₀ stored externally already

    for nu in range(1, μ_iterations + 1):
        logger.info("=" * 80)
        logger.info(f"STARTING μ ITERATION {nu}")
        logger.info("=" * 80)

        # freeze reference for stability (paper used slow-moving π)
        ref_model = deepcopy(current_model).eval()
        ref_model.requires_grad_(False)

        # === Optimizer & LR schedule ===
        optim = AdamW(current_model.parameters(), lr=lr, weight_decay=0)
        sched = get_scheduler(
            name="constant_with_warmup",
            optimizer=optim,
            num_warmup_steps=50,
            num_training_steps=len(prompts) // prompts_per_gen
        )

        current_model.train()
        tokenizer.pad_token = tokenizer.eos_token

        step = 0
        for i in range(0, len(prompts), prompts_per_gen):
            batch_prompts = prompts[i, i+ prompts_per_gen]
            batch_answers = answers[i, i + prompts_per_gen]

            if len(batch_prompts) == 0:
                break
            generations, logprobs = generate_group_samples(
                model=current_model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                group_size=group_size,
            )

            rewards = []

            from trainingv1.reward import check_answer_correctness
            for k in range(len(batch_prompts)):
                g = generations[k]
                answer =batch_answers[i+k]
                r_group = [1.0 if check_answer_correctness(sample, answer) else 0.0 for sample in g]
                rewards.append(torch.tensor(r_group, dtype = torch.float32, device= current_model.device))
            
            advantages = compute_group_advantages(
                rewards=rewards,
                normalize = True,
                rank_normalize = True,
            )

            loss = dr_grpo_loss(
                advantages=advantages,
                logprobs=logprobs,
            )


            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()

            step+=1
            if step % 10 == 0:
                logger.info(f"μ Iteration {nu} | Step {step} | Loss: {loss.item():.4f}")
        
        logger.info(f"μ iteration {nu} completed")

        NT = evaluate_nt(
            model=current_model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            num_samples=200
        )
        logger.info(f"NT after μ={nu}: {NT:.3f}")

        if target_nt and NT >= target_nt:
            logger.info(f"Reached target NT={target_nt}, stopping μ-loop early.")
            break

        π_models.append(current_model)

        gc.collect()
        torch.cuda.empty_cache()
    
    logger.info("DR.GRPO TRAINING COMPLETE.")
    return current_model, π_models



