import gc
import os
import torch
import logging

import torch.nn.functional as F
from torch.optim import AdamW
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
    eval_dataset=None,
    domain="math",
    μ_iterations=2,
    lr=2e-5,
    group_size=64,
    prompts_per_gen=8,
    gradient_accumulation_steps=1,
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
            sample groups → compute binary reward → compute normalized A →
            optimize L = − E_group[A_i * log π(y_i|x)]

        Group sampling:
            group_size = 64
            prompts_per_gen = 8

        No explicit KL regularization.
        KL is implicitly minimized by relative group loss.

        Args:
            model: Model to train
            tokenizer: Tokenizer
            dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            domain: Domain for reward checking ("math", "science", "tool")
            μ_iterations: Number of μ iterations (default: 2)
            lr: Learning rate (default: 2e-5)
            group_size: Group size for sampling (default: 64)
            prompts_per_gen: Prompts per generation batch (default: 8)
            gradient_accumulation_steps: Accumulate gradients over N batches
                before stepping the optimizer (default: 1, paper uses 1).
                Effective batch = prompts_per_gen × gradient_accumulation_steps.
            target_nt: Target NT score to stop early (optional)
            max_samples: Maximum training samples (default: 3000)

        Returns:
            tuple: (trained_model, final_NT_score)
    """

    logger.info("=" * 70)
    logger.info("INITIALIZING PURE Dr.GRPO TRAINING.")
    logger.info("=" * 70)

    effective_batch = prompts_per_gen * gradient_accumulation_steps
    logger.info(f"Current Learning Rate : {lr}, Group Size : {group_size}, Prompts per Generation : {prompts_per_gen}, Gradient Accum Steps : {gradient_accumulation_steps}, Effective Batch : {effective_batch}, Max Samples : {max_samples}, μ Iterations : {μ_iterations}")
    logger.info(f"Domain: {domain}")
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

    for nu in range(1, μ_iterations + 1):
        logger.info("=" * 80)
        logger.info(f"STARTING μ ITERATION {nu}")
        logger.info("=" * 80)

        # freeze reference for stability (paper used slow-moving π)


        # === Optimizer & LR schedule ===
        optim = AdamW(current_model.parameters(), lr=lr, weight_decay=0)
        sched = get_scheduler(
            name="constant_with_warmup",
            optimizer=optim,
            num_warmup_steps=50,
            num_training_steps=(len(prompts) // prompts_per_gen // gradient_accumulation_steps) * μ_iterations

        )

        current_model.train()
        tokenizer.pad_token = tokenizer.eos_token

        step = 0
        accum_step = 0
        accum_loss = 0.0

        from trainingv1.reward import check_answer_correctness

        for i in range(0, len(prompts), prompts_per_gen):
            batch_prompts = prompts[i:i + prompts_per_gen]
            batch_answers = answers[i:i + prompts_per_gen]

            if len(batch_prompts) == 0:
                break

            generations, logprobs = generate_group_samples(
                model=current_model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                group_size=group_size,
            )

            rewards = []
            for k in range(len(batch_prompts)):
                g = generations[k]
                answer = batch_answers[k]
                r_group = [1.0 if check_answer_correctness(sample, answer, domain=domain) else 0.0 for sample in g]
                rewards.append(torch.tensor(r_group, dtype=torch.float32, device=current_model.device))

            advantages = compute_group_advantages(
                rewards=rewards,
                normalize=True,
                rank_normalize=False,
            )

            loss = dr_grpo_loss(
                advantages=advantages,
                logprobs=logprobs,
            )

            # Scale loss by accumulation steps so gradients average correctly
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            loss_val = loss.item()
            accum_loss += loss_val
            accum_step += 1

            # Free intermediate tensors after each micro-step
            del generations, logprobs, rewards, advantages, loss, scaled_loss
            gc.collect()
            torch.cuda.empty_cache()

            # Optimizer step after accumulating enough gradients
            if accum_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(current_model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()

                step += 1
                avg_loss = accum_loss / gradient_accumulation_steps
                accum_loss = 0.0

                if step % 10 == 0:
                    mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    logger.info(f"μ Iteration {nu} | Step {step} | Loss: {avg_loss:.4f} | GPU Mem: {mem_gb:.2f}GB")

        # Flush any remaining accumulated gradients
        if accum_step % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(current_model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            step += 1
        
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

        gc.collect()
        torch.cuda.empty_cache()

    logger.info("DR.GRPO TRAINING COMPLETE.")
    logger.info(f"Final NT: {NT:.3f}")
    return current_model, NT



