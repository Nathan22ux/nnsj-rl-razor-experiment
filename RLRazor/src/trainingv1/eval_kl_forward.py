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
    Paper-grade forward KL computation (per-token, then averaged over samples):

        KL(π₀ || πμ) =
        E_{x ~ D_new, y ~ π₀(.|x)}
        [ (1/T) Σ_t  log π₀(y_t|y<t,x) - log πμ(y_t|y<t,x) ]

    Uses per-token mean to avoid length bias and ensure non-negative KL
    in expectation (Gibbs' inequality).

    The per-token KL at each position is computed via the full token-level
    KL divergence:  Σ_v  p_base(v) * [log p_base(v) - log p_target(v)]
    which is guaranteed ≥ 0 for every position, making the overall
    estimate strictly non-negative.
    """

    logger.info("=" * 70)
    logger.info("Computing forward KL: KL(π₀ || πμ)")
    logger.info("=" * 70)

    base_model.eval()
    target_model.eval()

    device = base_model.device

    # Only normalize if not already normalized (check for 'prompt' field)
    from data.dataset_utils import UnifiedDatasetInterface
    if 'prompt' not in dataset.column_names:
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

        num_gen_tokens = len(generated_tokens)

        # Compute log probabilities properly
        # Forward pass to get logits
        with torch.no_grad():
            out_base = base_model(generated)
            logits_base = out_base.logits[0, prompt_len-1:-1, :]  # Logits predicting generated tokens

            out_target = target_model(generated)
            logits_target = out_target.logits[0, prompt_len-1:-1, :]

        # ── True token-level KL divergence (always ≥ 0) ──────────────
        # For each position t, compute:
        #   KL_t = Σ_v  p_base(v) * [log p_base(v) - log p_target(v)]
        # This is the exact KL between the two categorical distributions
        # at each token position, guaranteed non-negative by Gibbs' inequality.
        log_probs_base = F.log_softmax(logits_base, dim=-1)
        log_probs_target = F.log_softmax(logits_target, dim=-1)
        probs_base = log_probs_base.exp()  # p_base(v) for all v

        # token-level KL: Σ_v p(v) * [log p(v) - log q(v)]  (shape: [num_gen_tokens])
        per_token_kl = (probs_base * (log_probs_base - log_probs_target)).sum(dim=-1)

        # Average over generated tokens for this sample
        sample_kl = per_token_kl.mean()

        kl_values.append(sample_kl.detach().cpu())

    kl_mean = torch.stack(kl_values).mean().item()

    logger.info(f"Forward KL(π₀ || πμ): {kl_mean:.6f}")
    logger.info(f"  (computed over {len(kl_values)} samples, per-token averaged)")
    return kl_mean
