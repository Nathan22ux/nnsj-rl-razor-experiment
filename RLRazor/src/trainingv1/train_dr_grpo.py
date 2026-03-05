import gc
import logging

import torch
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from data.dataset_utils import UnifiedDatasetInterface
from logger import get_logger
from trainingv1.reward import check_answer_correctness

logger = get_logger(__name__)


class RLMetricsCallback(TrainerCallback):
    """Writes GRPOTrainer step metrics to the project logger (text file)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        parts = [f"step={step}"]
        for k, v in logs.items():
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        logger.info("RL | %s", " | ".join(parts))


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
    max_completion_length=512,
    **kwargs,
):
    # Backward compatibility
    if "μ_iterations" in kwargs:
        mu_iterations = kwargs.pop("μ_iterations")
    if kwargs:
        raise TypeError(f"Unexpected keyword argument(s): {', '.join(sorted(kwargs.keys()))}")

    logger.info("=" * 70)
    logger.info("INITIALIZING Dr.GRPO TRAINING (GRPOTrainer)")
    logger.info("=" * 70)
    logger.info(
        "lr=%s | group_size=%s | prompts_per_gen=%s | mu_iterations=%s | domain=%s",
        lr, group_size, prompts_per_gen, mu_iterations, domain,
    )

    # --- Dataset ---
    # Normalize and limit. Keep only 'prompt' and 'answer' columns:
    # - 'prompt' is used by GRPOTrainer for generation
    # - 'answer' is passed automatically to the reward function as a kwarg
    dataset = UnifiedDatasetInterface.normalize_dataset(dataset)
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    cols_to_remove = [c for c in dataset.column_names if c not in ("prompt", "answer")]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
    logger.info("Dataset: %s prompts | columns: %s", len(dataset), dataset.column_names)

    # --- vLLM detection ---
    try:
        import vllm  # noqa: F401
        use_vllm = True
        logger.info("vLLM detected - generation will be 3-10x faster")
    except ImportError:
        use_vllm = False
        logger.info("vLLM not found - using standard generation (pip install vllm to speed up)")

    # --- Reward function ---
    # GRPOTrainer automatically passes dataset columns as kwargs,
    # so 'answer' arrives here directly - no fragile hash lookup needed.
    def reward_fn(completions, prompts, answer=None, **kw):
        if answer is None:
            logger.warning("'answer' column missing from batch - returning 0 rewards")
            return [0.0] * len(completions)
        return [
            1.0 if check_answer_correctness(c, a, domain=domain) else 0.0
            for c, a in zip(completions, answer)
        ]

    # --- GRPOConfig ---
    # num_train_epochs=mu_iterations: since beta=0 (no explicit KL),
    # running mu epochs is equivalent to the paper's mu-iterations.
    grpo_config = GRPOConfig(
        output_dir=f"./results_rl/lr{lr}_mu{mu_iterations}",
        # Training schedule
        num_train_epochs=mu_iterations,
        per_device_train_batch_size=prompts_per_gen,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=50,
        max_grad_norm=1.0,
        bf16=True,
        optim="adamw_torch",
        weight_decay=0.0,
        gradient_checkpointing=True,
        # Dr.GRPO loss (paper's method)
        loss_type="dr_grpo",
        beta=0.0,                       # No explicit KL penalty (paper uses implicit KL only)
        num_generations=group_size,
        generation_batch_size=prompts_per_gen,  # smaller than num_generations to avoid OOM
        max_completion_length=max_completion_length,
        temperature=0.6,
        top_p=0.8,
        # Fast generation via vLLM
        use_vllm=use_vllm,
        # Logging / saving
        logging_steps=log_interval,
        report_to="none",
        save_strategy="no",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        callbacks=[RLMetricsCallback()],
    )

    logger.info("Starting GRPOTrainer...")
    trainer.train()
    logger.info("GRPOTrainer training complete")

    # --- NT Evaluation ---
    final_nt = 0.0
    if eval_dataset is not None:
        final_nt = evaluate_nt(
            model=trainer.model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            num_samples=100,
        )
        logger.info("Final NT: %.3f", final_nt)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return trainer.model, final_nt
