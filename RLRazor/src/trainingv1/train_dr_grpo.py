import gc
import torch
import logging

from torch.optim import AdamW
from transformers import get_scheduler

from data.dataset_utils import UnifiedDatasetInterface

from trainingv1.rollout import generate_group_samples, recompute_logprobs
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
    μ_iterations=1,
    lr=2e-5,
    group_size=64,
    prompts_per_gen=8,
    target_nt=None,
    max_samples=3000,
):
    """
    Dr.GRPO training implementation for RL's Razor.

        Paper mechanism (1 epoch):
        ------------------------------------------------------
        π₀ = base model
        π₁ = Dr.GRPO(π₀)    (μ=1)

        For each μ:
            sample groups → compute binary reward → compute rank-normalized A →
            recompute log probs WITH gradient →
            optimize L = − E_group[A_i * log π(y_i|x)]

        Group sampling:
            group_size = 64
            prompts_per_gen = 8

        No explicit KL regularization.
        KL is implicitly minimized by on-policy sampling.
    """

    logger.info("=" * 70)
    logger.info("INITIALIZING Dr.GRPO TRAINING")
    logger.info("=" * 70)
    logger.info(f"Learning Rate: {lr}, Group Size: {group_size}, Prompts/Gen: {prompts_per_gen}")
    logger.info(f"Domain: {domain}, Max samples: {max_samples}, μ iterations: {μ_iterations}")
    logger.info(f"Target NT: {target_nt if target_nt else 'None'}")

    dataset = UnifiedDatasetInterface.normalize_dataset(dataset)
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    prompts = dataset["prompt"]
    answers = dataset["answer"]
    logger.info(f"Training dataset: {len(prompts)} prompts loaded")

    current_model = model
    NT = 0.0

    for nu in range(1, μ_iterations + 1):
        logger.info("=" * 70)
        logger.info(f"μ ITERATION {nu}/{μ_iterations}")
        logger.info("=" * 70)

        # Optimizer & LR schedule
        optim = AdamW(current_model.parameters(), lr=lr, weight_decay=0)
        total_steps = len(prompts) // prompts_per_gen
        sched = get_scheduler(
            name="constant_with_warmup",
            optimizer=optim,
            num_warmup_steps=50,
            num_training_steps=total_steps
        )

        current_model.train()
        tokenizer.pad_token = tokenizer.eos_token

        step = 0
        for i in range(0, len(prompts), prompts_per_gen):
            batch_prompts = prompts[i:i + prompts_per_gen]
            batch_answers = answers[i:i + prompts_per_gen]

            if len(batch_prompts) == 0:
                break

            # Phase 1: Generate samples (no grad)
            generations, generated_ids_all, prompt_lengths = generate_group_samples(
                model=current_model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                group_size=group_size,
            )

            # Compute binary rewards
            from trainingv1.reward import check_answer_correctness
            rewards = []
            total_correct = 0
            total_samples = 0
            for k in range(len(batch_prompts)):
                g = generations[k]
                answer = batch_answers[k]
                r_group = []
                for sample in g:
                    correct = check_answer_correctness(sample, answer, domain=domain)
                    r_group.append(1.0 if correct else 0.0)
                    total_correct += int(correct)
                    total_samples += 1
                rewards.append(torch.tensor(r_group, dtype=torch.float32, device=current_model.device))

            # Compute advantages
            advantages = compute_group_advantages(
                rewards=rewards,
                normalize=True,
                rank_normalize=True,
            )

            # Phase 2: Recompute log probs WITH gradient
            logprobs = recompute_logprobs(
                model=current_model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                generated_ids_all=generated_ids_all,
                prompt_lengths=prompt_lengths,
                mini_batch_size=8,
            )

            # Dr.GRPO loss
            loss = dr_grpo_loss(
                advantages=advantages,
                logprobs=logprobs,
            )

            loss.backward()

            # Verification: check gradient flow on first step
            if step == 0:
                grad_norm = 0.0
                for p in current_model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                logger.info(f"[VERIFY] grad_norm={grad_norm:.6f} (must be > 0 for GRPO to work)")

            optim.step()
            sched.step()
            optim.zero_grad()

            step += 1
            mean_reward = total_correct / max(total_samples, 1)
            pct_correct = mean_reward * 100

            if step % 5 == 0 or step == 1:
                logger.info(
                    f"μ={nu} | Step {step}/{total_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"mean_reward={mean_reward:.3f}, pct_correct={pct_correct:.1f}%"
                )

        logger.info(f"μ iteration {nu} completed ({step} steps)")

        # Evaluate NT
        if eval_dataset is not None:
            NT = evaluate_nt(
                model=current_model,
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
                num_samples=200
            )
            logger.info(f"NT after μ={nu}: {NT:.3f}")

            if target_nt and NT >= target_nt:
                logger.info(f"Reached target NT={target_nt}, stopping early.")
                break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("=" * 70)
    logger.info("Dr.GRPO TRAINING COMPLETE")
    logger.info(f"Final NT: {NT:.3f}")
    logger.info("=" * 70)
    return current_model, NT
