import torch
import torch.nn.functional as F
from transformers import GenerationConfig
import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_group_samples(model, tokenizer, prompts, group_size=64, max_new_tokens=128, temperature=0.6):
    """
    Phase 1: Generate group samples for Dr.GRPO (no gradients needed).

    For each prompt, generate `group_size` completions.

    Returns:
        generations: list[list[str]] — decoded text per prompt per sample
        generated_ids: list[list[Tensor]] — token IDs of generated portion per prompt per sample
        prompt_lengths: list[int] — prompt length in tokens per prompt
    """
    device = model.device
    generations = []
    generated_ids_all = []
    prompt_lengths = []

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.8,
        num_return_sequences=group_size,
        pad_token_id=tokenizer.eos_token_id,
    )

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = inputs['input_ids'].shape[1]
        prompt_lengths.append(prompt_length)

        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generations.append(decoded)

        # Store generated token IDs (excluding prompt) for log prob recomputation
        group_gen_ids = []
        for sq in outputs:
            gen_tokens = sq[prompt_length:]
            group_gen_ids.append(gen_tokens.detach().cpu())
        generated_ids_all.append(group_gen_ids)

    return generations, generated_ids_all, prompt_lengths


def recompute_logprobs(model, tokenizer, prompts, generated_ids_all, prompt_lengths, mini_batch_size=8):
    """
    Phase 2: Recompute log probabilities WITH gradient flow.

    Processes sequences in mini-batches to avoid OOM.

    Args:
        model: The policy model (gradients enabled)
        tokenizer: Tokenizer
        prompts: list[str] — original prompts
        generated_ids_all: list[list[Tensor]] — generated token IDs per prompt per sample
        prompt_lengths: list[int] — prompt token lengths
        mini_batch_size: sequences per mini-batch (controls memory)

    Returns:
        logprobs: list[Tensor] — per-prompt tensor of shape [group_size], with gradient
    """
    device = next(model.parameters()).device
    logprobs_groups = []

    for p_idx, prompt in enumerate(prompts):
        prompt_len = prompt_lengths[p_idx]
        group_gen_ids = generated_ids_all[p_idx]
        group_size = len(group_gen_ids)

        prompt_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)
        group_logprobs = []

        # Mini-batch to control memory
        for mb_start in range(0, group_size, mini_batch_size):
            mb_end = min(mb_start + mini_batch_size, group_size)
            mb_gen_ids = group_gen_ids[mb_start:mb_end]

            for gen_tokens in mb_gen_ids:
                gen_tokens = gen_tokens.to(device)

                if len(gen_tokens) == 0:
                    group_logprobs.append(torch.tensor(0.0, device=device, requires_grad=True))
                    continue

                # Build full sequence: prompt + generated
                full_ids = torch.cat([prompt_ids[0], gen_tokens]).unsqueeze(0)

                # Forward pass WITH gradient
                out = model(full_ids)
                logits = out.logits[0]  # [seq_len, vocab_size]

                # logits[i] predicts token[i+1]
                pred_logits = logits[prompt_len - 1: prompt_len - 1 + len(gen_tokens), :]
                log_probs = F.log_softmax(pred_logits, dim=-1)

                # Gather log probs for actual generated tokens
                token_log_probs = log_probs[range(len(gen_tokens)), gen_tokens]

                # Sum log probabilities for this sequence
                total_logprob = token_log_probs.sum()
                group_logprobs.append(total_logprob)

        logprobs_groups.append(torch.stack(group_logprobs).to(device))

    return logprobs_groups