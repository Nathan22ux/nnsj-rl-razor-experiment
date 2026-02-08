import torch
from logger import get_logger

logger = get_logger("eval_forgetting")


@torch.no_grad()
def compute_old_task_performance(
    model,
    tokenizer,
    dataset,
    num_samples=200,
    max_new_tokens=128,
):
    """
    Compute old-task performance as exact-match accuracy.

    This is used for BOTH π₀ and πμ.
    """

    model.eval()
    device = model.device

    # normalize dataset
    from data.dataset_utils import UnifiedDatasetInterface
    dataset = UnifiedDatasetInterface.normalize_dataset(dataset)
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    prompts = dataset["prompt"]
    answers = dataset["answer"]

    correct = 0
    total = len(prompts)

    from trainingv1.reward import check_answer_correctness

    for prompt, gt in zip(prompts, answers):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        output = model.generate(
            **inputs,
            do_sample=False,          # deterministic for eval
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode only the generated portion (not the prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_ids = output[0, input_length:]
        pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Use check_answer_correctness for robust matching
        if check_answer_correctness(pred_text, gt):
            correct += 1

    accuracy = correct / max(total, 1)
    return accuracy


def compute_forgetting(
    base_model,      # π₀
    target_model,    # πμ
    tokenizer,
    old_task_dataset,
    num_samples=200,
):
    """
    forgetting metric:
        Forgetting = Perf_old(π₀) - Perf_old(πμ)

    Positive value => forgetting occurred.
    """

    logger.info("=" * 70)
    logger.info("Computing forgetting on OLD-TASK distribution")
    logger.info("=" * 70)

    acc_base = compute_old_task_performance(
        model=base_model,
        tokenizer=tokenizer,
        dataset=old_task_dataset,
        num_samples=num_samples,
    )

    acc_target = compute_old_task_performance(
        model=target_model,
        tokenizer=tokenizer,
        dataset=old_task_dataset,
        num_samples=num_samples,
    )

    forgetting = acc_base - acc_target

    logger.info(f"Old-task acc π₀ : {acc_base:.4f}")
    logger.info(f"Old-task acc πμ : {acc_target:.4f}")
    logger.info(f"Forgetting     : {forgetting:.4f}")

    return forgetting