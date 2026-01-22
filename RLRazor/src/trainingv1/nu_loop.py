import torch
import gc
from copy import deepcopy

from logger import get_logger
from trainingv1.rollout import generate_group_samples
from trainingv1.advantages import compute_group_advantages
from trainingv1.dr_loss import dr_grpo_loss
from trainingv1.reward import build_binary_rewards

logger = get_logger("nu_loop")


@torch.no_grad()
def evaluate_nt(model, tokenizer, eval_dataset, num_samples=200):
    """
    Paper evaluates NT after each μ iteration.
    """
    from evaluation.evaluation import evaluate_new_task
    return evaluate_new_task(
        model=model,
        tokenizer=tokenizer,
        dataset=eval_dataset,
        num_samples=num_samples
    )


def run_mu_iterations(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    μ=2,
    lr=2e-5,
    group_size=64,
    prompts_per_gen=8,
    target_nt=None,
    max_samples=2000,
    domain_list=None,  # optional ["math","science","tool",...]
):
    """
    Paper-grade μ loops for RL's Razor:
        π₀ → π₁ → π₂

    Args:
        model: π₀ input model (SFT)
        μ: number of RL refinement iterations (default=2)
        lr: Dr.GRPO LR sweep
        target_nt: gating threshold for early stop
        domain_list: optional domain settings per sample
    """

    logger.info("=" * 80)
    logger.info(f"Starting μ-loop: μ={μ}, lr={lr}, group={group_size}, prompts/gen={prompts_per_gen}")
    logger.info("=" * 80)

    # normalize train dataset
    from data.dataset_utils import UnifiedDatasetInterface
    train_dataset = UnifiedDatasetInterface.normalize_dataset(train_dataset)
    train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))

    prompts = train_dataset["prompt"]
    answers = train_dataset["answer"]
    if domain_list is None:
        domains = ["math"] * len(prompts)
    else:
        domains = domain_list

    π_models = [deepcopy(model)]   # store π₀
    current = model

    for iteration in range(1, μ + 1):
        logger.info("=" * 50)
        logger.info(f"μ Iteration {iteration} starting...")
        logger.info("=" * 50)

        # fresh optimizer for this μ
        optim = torch.optim.AdamW(current.parameters(), lr=lr, weight_decay=0)

        current.train()
        tokenizer.pad_token = tokenizer.eos_token

        step = 0
        for i in range(0, len(prompts), prompts_per_gen):
            batch_prompts = prompts[i:i+prompts_per_gen]
            batch_answers = answers[i:i+prompts_per_gen]
            batch_domains = domains[i:i+prompts_per_gen]

            if len(batch_prompts) == 0:
                break

            # === rollout ===
            generations, logprobs = generate_group_samples(
                model=current,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                group_size=group_size,
            )

            # === build rewards ===
            rewards = build_binary_rewards(
                generations=generations,
                answers=batch_answers,
                domains=batch_domains,
            )

            # === compute group advantages ===
            advantages = compute_group_advantages(
                rewards=rewards,
                normalize=True,
                rank_normalize=True,
            )

            # === Dr.GRPO loss ===
            loss = dr_grpo_loss(
                advantages=advantages,
                logprobs=logprobs,
            )

            loss.backward()
            optim.step()
            optim.zero_grad()

            step += 1
            if step % 10 == 0:
                logger.info(f"[μ={iteration}] step={step} loss={loss.item():.4f}")

        logger.info(f"μ iteration {iteration} finished.")

        # === NT evaluation for gating ===
        NT = evaluate_nt(
            model=current,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            num_samples=200
        )
        logger.info(f"NT after μ={iteration}: {NT:.3f}")

        if target_nt is not None and NT >= target_nt:
            logger.info(f"Reached NT target {target_nt}, stopping μ early.")
            π_models.append(deepcopy(current))
            break

        π_models.append(deepcopy(current))

        gc.collect()
        torch.cuda.empty_cache()

    logger.info("μ-loop complete.")
    return current, π_models
