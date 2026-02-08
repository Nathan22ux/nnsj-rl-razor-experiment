import torch
import torch.nn.functional as F
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

        # Get prompt length to compute log probs only on generated portion
        prompt_len = inputs['input_ids'].shape[1]
        generated_tokens = generated[0, prompt_len:]

        if len(generated_tokens) == 0:
            continue  # Skip if nothing was generated

        # Compute log probabilities properly
        # Forward pass to get logits
        with torch.no_grad():
            out_base = base_model(generated)
            logits_base = out_base.logits[0, prompt_len-1:-1, :]  # Logits predicting generated tokens

            out_target = target_model(generated)
            logits_target = out_target.logits[0, prompt_len-1:-1, :]

        # Convert to log probabilities
        log_probs_base = F.log_softmax(logits_base, dim=-1)
        log_probs_target = F.log_softmax(logits_target, dim=-1)

        # Gather log probs for the actual generated tokens
        token_logp_base = log_probs_base[range(len(generated_tokens)), generated_tokens]
        token_logp_target = log_probs_target[range(len(generated_tokens)), generated_tokens]

        # Sum log probabilities (total log prob of sequence)
        logp_base = token_logp_base.sum()
        logp_target = token_logp_target.sum()

        # forward KL contribution: log π₀(y|x) - log πμ(y|x)
        kl = (logp_base - logp_target).detach().cpu()
        kl_values.append(kl)

    kl_mean = torch.stack(kl_values).mean().item()

    logger.info(f"Forward KL(π₀ || πμ): {kl_mean:.6f}")
    return kl_mean