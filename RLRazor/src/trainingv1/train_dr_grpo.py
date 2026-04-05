import gc
import json
import os
import torch
import logging

import torch.nn.functional as F
from torch.optim import AdamW
from copy import deepcopy
from transformers import get_scheduler

from data.dataset_utils import UnifiedDatasetInterface

from trainingv1.rollout import generate_group_samples, compute_loss_with_grad_accum
from trainingv1.advantages import compute_group_advantages
from trainingv1.reward import check_answer_correctness

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
    group_size=32,
    prompts_per_gen=8,
    target_nt=None,
    max_samples=3000,
    max_completion_length=128,
    warmup_steps=50,
    gradient_accumulation_steps=None,
    checkpoint_dir="./checkpoints/rl",  # directory to save checkpoints
    save_every_mu=True,                 # save after each μ iteration
    scheduler_type="constant_with_warmup",  # FIX 3: accept scheduler_type explicitly
    **kwargs,  # absorb any extra kwargs from experiment callers without crashing
):
    """
    Dr.GRPO training implementation for RL's Razor.

    Two-phase training loop per step:
        Phase 1 (Rollout — no grad):
            Sample group_size responses per prompt from the current policy.
            Compute binary rewards (correct/incorrect).
            Compute group advantages.

        Phase 2 (Update — with grad):
            Recompute log probabilities on the same generated sequences
            with gradients enabled, then compute the Dr.GRPO loss and backprop.

    Paper mechanism:
    ------------------------------------------------------
    π₀ = base model
    π₁ = Dr.GRPO(π₀)    (μ=1)
    π₂ = Dr.GRPO(π₁)    (μ=2)

    For each μ:
        sample groups → compute binary reward → compute rank-normalized A →
        recompute log probs with grad →
        optimize L = − E_group[A_i * log π(y_i|x)]

    Group sampling:
        group_size = 64
        prompts_per_gen = 8

    No explicit KL regularization.
    KL is implicitly minimized by on-policy sampling.

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
        target_nt: Target NT score to stop early (optional)
        max_samples: Maximum training samples (default: 3000)

    Returns:
        tuple: (trained_model, final_NT_score)
    """

    logger.info("=" * 70)
    logger.info("INITIALIZING Dr.GRPO TRAINING (two-phase: rollout + update)")
    logger.info("=" * 70)

    logger.info(f"Learning Rate: {lr}, Group Size: {group_size}, "
                f"Prompts/Gen: {prompts_per_gen}, Max Samples: {max_samples}, "
                f"μ Iterations: {μ_iterations}")
    logger.info(f"Domain: {domain}")
    logger.info(f"Max completion length: {max_completion_length}")
    logger.info(f"Target NT: {target_nt if target_nt else 'None'}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    if kwargs:
        logger.info(f"Extra kwargs received (ignored): {list(kwargs.keys())}")

    # Create checkpoint directory and metrics file
    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics_path = os.path.join(checkpoint_dir, "training_metrics.json")
    all_metrics = []

    # Only normalize if not already normalized (check for 'prompt' field)
    if 'prompt' not in dataset.column_names:
        dataset = UnifiedDatasetInterface.normalize_dataset(dataset)
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Extract prompts + ground truths for reward
    prompts = dataset["prompt"]
    answers = dataset["answer"]
    logger.info(f"Dataset loaded with {len(prompts)} prompts")

    # === μ iteration refinement ===
    current_model = model
    NT = 0.0

    # === Optimizer & LR schedule (initialized ONCE — preserves Adam momentum across μ) ===
    optim = AdamW(current_model.parameters(), lr=lr, weight_decay=0)

    grad_accum_steps = gradient_accumulation_steps if gradient_accumulation_steps is not None else 1
    steps_per_mu = (len(prompts) // prompts_per_gen) // grad_accum_steps
    total_optim_steps = steps_per_mu * μ_iterations

    sched = get_scheduler(
        name=scheduler_type,
        optimizer=optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optim_steps
    )

    for nu in range(1, μ_iterations + 1):
        logger.info("=" * 80)
        logger.info(f"STARTING μ ITERATION {nu}")
        logger.info("=" * 80)

        tokenizer.pad_token = tokenizer.eos_token

        step = 0
        micro_step = 0
        
        from tqdm import tqdm
        batch_pbar = tqdm(
            range(0, len(prompts), prompts_per_gen),
            total=(len(prompts) + prompts_per_gen - 1) // prompts_per_gen,
            desc=f"μ={nu} Batches",
            leave=False
        )
        
        for i in batch_pbar:
            batch_prompts = prompts[i:i + prompts_per_gen]
            batch_answers = answers[i:i + prompts_per_gen]

            if len(batch_prompts) == 0:
                break

            # ============================================================
            # PHASE 1: ROLLOUT (no gradients)
            # Generate samples, compute rewards and advantages
            # ============================================================
            current_model.eval()

            generations, generated_token_ids, prompt_lengths = generate_group_samples(
                model=current_model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                group_size=group_size,
                max_new_tokens=max_completion_length,
            )

            # Compute binary rewards
            rewards = []
            for k in range(len(batch_prompts)):
                g = generations[k]
                answer = batch_answers[k]
                r_group = [
                    1.0 if check_answer_correctness(sample, answer, domain=domain) else 0.0
                    for sample in g
                ]
                rewards.append(torch.tensor(r_group, dtype=torch.float32, device=current_model.device))

            # FIX 2: Dr.GRPO uses only mean-subtraction + variance normalization.
            # rank_normalize overwrites the variance-normalized advantages with a
            # fixed grid of values, destroying the actual reward signal.
            advantages = compute_group_advantages(
                rewards=rewards,
                normalize=True,
                rank_normalize=False,
            )

            # ============================================================
            # PHASE 2: UPDATE (with gradients)
            # Per-sequence gradient accumulation — ONE graph in VRAM at a time
            # ============================================================
            current_model.train()

            # FIX 1: compute_loss_with_grad_accum already normalises by total_sequences
            # and calls .backward() internally.  Do NOT divide gradients again — that
            # was causing an 8× under-scaling of every gradient update.
            loss_val = compute_loss_with_grad_accum(
                model=current_model,
                generated_token_ids=generated_token_ids,
                prompt_lengths=prompt_lengths,
                advantages=advantages,
                optimizer=optim,
            )

            micro_step += 1
            
            if micro_step % grad_accum_steps == 0 or i + prompts_per_gen >= len(prompts):
                # Step optimizer once every grad_accum_steps
                torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=1.0)
                optim.step()
                sched.step()
                optim.zero_grad()

                step += 1
                
            # Compute step metrics before freeing tensors
            reward_means = [r.mean().item() for r in rewards]
            reward_stds  = [r.std().item() for r in rewards]
            frac_zero_std = sum(1 for s in reward_stds if s < 1e-6) / max(len(reward_stds), 1)
            batch_reward  = sum(reward_means) / max(len(reward_means), 1)
            current_lr    = sched.get_last_lr()[0]

            logger.info(
                f"μ={nu} | micro_step={micro_step} | optim_step={step}/{total_optim_steps} | loss={loss_val:.4f} | "
                f"reward={batch_reward:.3f} | frac_zero_std={frac_zero_std:.2f} | "
                f"lr={current_lr:.2e}"
            )
            batch_pbar.set_postfix({
                "loss": f"{loss_val:.3f}", 
                "reward": f"{batch_reward:.3f}"
            })

            # Save metrics for plotting
            all_metrics.append({
                "step": step,
                "mu": nu,
                "loss": loss_val,
                "reward_mean": batch_reward,
                "reward_std": float(sum(reward_stds) / max(len(reward_stds), 1)),
                "frac_zero_std": frac_zero_std,
                "lr": current_lr,
            })
            with open(metrics_path, "w") as f:
                json.dump(all_metrics, f)

            # Free rollout data (no empty_cache here — rollout.py already called it)
            del advantages, rewards, generations, generated_token_ids, prompt_lengths

        logger.info(f"μ iteration {nu} completed ({step} steps)")

        # Evaluate NT after each μ iteration
        NT = evaluate_nt(
            model=current_model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            num_samples=200
        )
        logger.info(f"NT after μ={nu}: {NT:.3f}")

        # Save checkpoint after each μ iteration
        if save_every_mu:
            ckpt_path = os.path.join(checkpoint_dir, f"mu_{nu}_nt{NT:.1f}")
            logger.info(f"Saving checkpoint to {ckpt_path} ...")
            current_model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")

        if target_nt and NT >= target_nt:
            logger.info(f"Reached target NT={target_nt}, stopping μ-loop early.")
            break

        gc.collect()
        torch.cuda.empty_cache()

    # Save final model
    final_path = os.path.join(checkpoint_dir, f"final_nt{NT:.1f}")
    logger.info(f"Saving final model to {final_path} ...")
    current_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Final model saved: {final_path}")

    logger.info("DR.GRPO TRAINING COMPLETE.")
    logger.info(f"Final NT: {NT:.3f}")
    return current_model, NT