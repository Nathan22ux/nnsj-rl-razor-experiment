import torch
from logger import get_logger

logger = get_logger("eval_kl_forward")

@torch.no_grad()
def compute_forward_kl(
    base_model,          # π₀
    target_model,        # πμ
    tokenizer,
    dataset,
    num_samples=200,
    max_new_tokens=128,
):
    """
    Paper-grade forward KL computation:

        KL(π₀ || πμ) =
        E_{x ~ D_new, y ~ π₀(.|x)}
        [ log π₀(y|x) - log πμ(y|x) ]
    """

    logger.info("=" * 70)
    logger.info("Computing forward KL: KL(π₀ || πμ)")
    logger.info("=" * 70)

    base_model.eval()
    target_model.eval()

    device = base_model.device

    # normalize dataset
    from data.dataset_utils import UnifiedDatasetInterface
    dataset = UnifiedDatasetInterface.normalize_dataset(dataset)
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    prompts = dataset["prompt"]

    kl_values = []

    for prompt in prompts:
        # tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # sample y ~ π₀(.|x)
        generated = base_model.generate(
            **inputs,
            do_sample=True,
            temperature=0.6,
            top_p=0.85,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

        # compute log π₀(y|x)
        out_base = base_model(generated, labels=generated)
        logp_base = -out_base.loss  # scalar, float32

        # compute log πμ(y|x)
        out_target = target_model(generated, labels=generated)
        logp_target = -out_target.loss  # scalar, float32

        # forward KL contribution
        kl = (logp_base - logp_target).detach().cpu()
        kl_values.append(kl)

    kl_mean = torch.stack(kl_values).mean().item()

    logger.info(f"Forward KL(π₀ || πμ): {kl_mean:.6f}")
    return kl_mean