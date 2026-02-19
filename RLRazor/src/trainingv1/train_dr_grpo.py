import gc
import logging
import math

import torch
from torch.optim import AdamW
from transformers import get_scheduler

from data.dataset_utils import UnifiedDatasetInterface
from trainingv1.advantages import compute_group_advantages
from trainingv1.dr_loss import dr_grpo_loss
from trainingv1.rollout import generate_group_samples

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_nt(model, tokenizer, eval_dataset, num_samples=500):
    from evaluation.evaluation import evaluate_new_task

    return evaluate_new_task(
        model=model,
        tokenizer=tokenizer,
        dataset=eval_dataset,
        num_samples=num_samples,
    )


def train_dr_grpo(
    model,
    tokenizer,
    dataset,
    eval_dataset=None,
    domain="math",
    mu_iterations=2,
    lr=2e-5,
    group_size=64,
    prompts_per_gen=8,
    gradient_accumulation_steps=1,
    target_nt=None,
    max_samples=3000,
    **kwargs,
):
    # Backward compatibility for callers using the unicode kwarg name.
    if "μ_iterations" in kwargs:
        mu_iterations = kwargs.pop("μ_iterations")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))

    logger.info("=" * 70)
    logger.info("INITIALIZING PURE Dr.GRPO TRAINING.")
    logger.info("=" * 70)
    logger.info(
        "Current Learning Rate : %s, Group Size : %s, Prompts per Generation : %s, "
        "Max Samples : %s, Mu Iterations : %s",
        lr,
        group_size,
        prompts_per_gen,
        max_samples,
        mu_iterations,
    )
    logger.info("Gradient accumulation steps: %s", gradient_accumulation_steps)
    logger.info("Domain: %s", domain)
    logger.info("Max samples: %s", max_samples)
    logger.info("Target NT: %s", target_nt if target_nt is not None else "None")

    dataset = UnifiedDatasetInterface.normalize_dataset(dataset)
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    prompts = dataset["prompt"]
    answers = dataset["answer"]
    logger.info("Dataset loaded with %s prompts", len(prompts))

    current_model = model
    if hasattr(current_model, "gradient_checkpointing_enable"):
        current_model.gradient_checkpointing_enable()
    if hasattr(current_model, "config"):
        current_model.config.use_cache = False

    total_batches = math.ceil(len(prompts) / max(1, prompts_per_gen))
    optimizer_steps = max(1, math.ceil(total_batches / gradient_accumulation_steps))
    final_nt = 0.0

    for nu in range(1, mu_iterations + 1):
        logger.info("=" * 80)
        logger.info("STARTING MU ITERATION %s", nu)
        logger.info("=" * 80)

        optim = AdamW(current_model.parameters(), lr=lr, weight_decay=0.0)
        sched = get_scheduler(
            name="constant_with_warmup",
            optimizer=optim,
            num_warmup_steps=50,
            num_training_steps=optimizer_steps,
        )

        current_model.train()
        tokenizer.pad_token = tokenizer.eos_token
        optim.zero_grad(set_to_none=True)

        step = 0
        from trainingv1.reward import check_answer_correctness

        for i in range(0, len(prompts), prompts_per_gen):
            batch_prompts = prompts[i : i + prompts_per_gen]
            batch_answers = answers[i : i + prompts_per_gen]
            if not batch_prompts:
                break

            batch_loss_value = 0.0
            prompt_count = max(1, len(batch_prompts))
            micro_scale = float(gradient_accumulation_steps * prompt_count)
            for prompt, answer in zip(batch_prompts, batch_answers):
                generations, logprobs = generate_group_samples(
                    model=current_model,
                    tokenizer=tokenizer,
                    prompts=[prompt],
                    group_size=group_size,
                    logprob_batch_size=1,
                )

                generated_group = generations[0]
                reward_group = [
                    1.0 if check_answer_correctness(sample, answer, domain=domain) else 0.0
                    for sample in generated_group
                ]
                rewards = [torch.tensor(reward_group, dtype=torch.float32)]

                advantages = compute_group_advantages(
                    rewards=rewards,
                    normalize=False,
                    rank_normalize=True,
                )
                loss = dr_grpo_loss(advantages=advantages, logprobs=logprobs)
                batch_loss_value += float(loss.item())
                (loss / micro_scale).backward()

                del generations, logprobs, rewards, advantages, loss
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            step += 1
            should_step = (step % gradient_accumulation_steps == 0) or (
                i + prompts_per_gen >= len(prompts)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(current_model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)

            if step % 10 == 0:
                logger.info(
                    "Mu Iteration %s | Step %s | Loss: %.4f",
                    nu,
                    step,
                    batch_loss_value / prompt_count,
                )

        logger.info("Mu iteration %s completed", nu)

        final_nt = evaluate_nt(
            model=current_model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            num_samples=200,
        )
        logger.info("NT after mu=%s: %.3f", nu, final_nt)

        if target_nt is not None and final_nt >= target_nt:
            logger.info("Reached target NT=%s, stopping mu-loop early.", target_nt)
            break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("DR.GRPO TRAINING COMPLETE.")
    logger.info("Final NT: %.3f", final_nt)
    return current_model, final_nt
