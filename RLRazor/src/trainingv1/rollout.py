import logging
import sys
import time
import torch
import torch.nn.functional as F
from transformers import GenerationConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_group_samples(model, tokenizer, prompts, group_size=32, max_new_tokens=128, temperature=1.0):
    """
    Group Sampling function for Dr.GRPO (rollout phase — no gradients).

    For each prompt, generate `group_size` samples using the current policy.
    Returns decoded text and token IDs stored on CPU to save VRAM.

    Returns:
        generations: list[list[str]] — decoded text for each (prompt, sample)
        generated_token_ids: list[list[Tensor]] — full token IDs on CPU
        prompt_lengths: list[int] — token length of each prompt
    """

    device = model.device
    generations = []
    generated_token_ids = []
    prompt_lengths = []

    # Build stop-token list: Qwen2.5-Instruct ends assistant turns with <|im_end|>
    # (token id 151645). We must include it alongside eos_token_id so generation
    # stops cleanly instead of looping past the end-of-turn marker.
    stop_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.eos_token_id:
        stop_ids.append(im_end_id)

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        num_return_sequences=group_size,
        eos_token_id=stop_ids,
        pad_token_id=tokenizer.eos_token_id,
    )

    num_prompts = len(prompts)
    logger.info(f"  [Rollout] Generating {group_size} completions × {num_prompts} prompts "
                f"(max_new_tokens={max_new_tokens})...")

    # pbar = tqdm(enumerate(prompts), total=num_prompts,
    #             desc=f"Rollout (G={group_size}, max_new={max_new_tokens})",
    #             unit="prompt", dynamic_ncols=True)

    for idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = inputs['input_ids'].shape[1]
        prompt_lengths.append(prompt_length)

        t_start = time.time()
        outputs = model.generate(**inputs, generation_config=generation_config)
        t_elapsed = time.time() - t_start
        avg_gen_len = (outputs.shape[1] - prompt_length) if outputs.dim() == 2 else 0
        logger.info(f"  [Rollout] Prompt {idx + 1}/{num_prompts} — done in {t_elapsed:.1f}s "
                    f"(gen_len={avg_gen_len})")

        # Decode ONLY the completion portion (strip the prompt) so that
        # reward functions don't accidentally match answer text that appears
        # inside the prompt itself (e.g., MCQ choice letters).
        completions = []
        for seq in outputs:
            completion_ids = seq[prompt_length:]
            completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
        generations.append(completions)

        # Store on CPU immediately to free GPU memory
        group_token_ids = [sq.detach().cpu() for sq in outputs]
        generated_token_ids.append(group_token_ids)

    # Single cleanup after all prompts — not inside the loop
    torch.cuda.empty_cache()
    logger.info(f"  [Rollout] All {num_prompts} prompts generated.")

    return generations, generated_token_ids, prompt_lengths


def compute_loss_with_grad_accum(model, generated_token_ids, prompt_lengths, advantages, optimizer):
    """
    Dr.GRPO loss computation using one batched forward pass per group.

    All group_size sequences from the same prompt share the same prompt length
    and (when no EOS is hit) the same total length, so they can be stacked and
    processed in a single forward pass. This replaces group_size separate
    batch-1 forward+backward calls with one batched forward + one backward per group.

    Args:
        model: Current policy model (must be in .train() mode)
        generated_token_ids: list[list[Tensor]] — from generate_group_samples()
        prompt_lengths: list[int] — token length of each prompt
        advantages: list[Tensor] — shape [group_size] per prompt
        optimizer: The optimizer (gradients accumulate; caller must step/zero)

    Returns:
        float: The total loss value (for logging)
    """

    device = model.device
    total_loss_val = 0.0
    total_sequences = sum(len(g) for g in generated_token_ids)

    for group_seqs, prompt_len, adv in zip(generated_token_ids, prompt_lengths, advantages):
        adv = adv.to(device)

        # Stack all group_size sequences into one batch: [G, seq_len]
        group_batch = torch.stack(group_seqs).to(device)
        gen_tokens_batch = group_batch[:, prompt_len:]   # [G, gen_len]
        gen_len = gen_tokens_batch.shape[1]

        if gen_len == 0:
            del group_batch, gen_tokens_batch
            continue

        # Single forward pass for the entire group (batch_size = group_size)
        out = model(group_batch)
        logits = out.logits                              # [G, seq_len, vocab_size]

        pred_logits = logits[:, prompt_len - 1:-1, :]   # [G, gen_len, vocab_size]
        log_probs = F.log_softmax(pred_logits, dim=-1)  # [G, gen_len, vocab_size]

        # Gather log prob of each generated token: [G, gen_len]
        token_log_probs = log_probs.gather(2, gen_tokens_batch.unsqueeze(2)).squeeze(2)
        total_logprobs = token_log_probs.sum(dim=1)     # [G]

        # Dr.GRPO loss: L = -Σ A_i * log π(y_i|x) / N
        group_losses = -adv * total_logprobs / total_sequences  # [G]
        group_loss = group_losses.sum()

        # Single backward for the whole group
        group_loss.backward()
        total_loss_val += group_loss.item()

        del out, logits, pred_logits, log_probs, token_log_probs
        del group_batch, gen_tokens_batch, group_losses, group_loss

    # Single cache clear per batch step
    torch.cuda.empty_cache()

    return total_loss_val


def recompute_logprobs(model, generated_token_ids, prompt_lengths):
    """
    Recompute log probabilities WITH GRADIENTS ENABLED.

    NOTE: For memory-constrained environments, prefer compute_loss_with_grad_accum()
    which processes one sequence at a time and calls backward() immediately.

    This function is kept for compatibility but may OOM with large group sizes.
    It processes sequences one at a time with explicit cleanup between each.

    Returns:
        logprobs: list[Tensor] — shape [group_size] per prompt
    """

    device = model.device
    logprobs_all = []

    for group_seqs, prompt_len in zip(generated_token_ids, prompt_lengths):
        group_logprobs = []

        for seq_ids in group_seqs:
            seq_input = seq_ids.unsqueeze(0).to(device)
            generated_tokens = seq_ids[prompt_len:]

            if len(generated_tokens) == 0:
                group_logprobs.append(torch.tensor(0.0, device=device))
                continue

            out = model(seq_input)
            logits = out.logits

            pred_logits = logits[0, prompt_len - 1:-1, :]
            log_probs = F.log_softmax(pred_logits, dim=-1)

            gen_tokens_gpu = generated_tokens.to(device)
            token_log_probs = log_probs[range(len(gen_tokens_gpu)), gen_tokens_gpu]
            total_logprob = token_log_probs.sum()

            group_logprobs.append(total_logprob)

            del out, logits, pred_logits, log_probs, token_log_probs, seq_input, gen_tokens_gpu
            torch.cuda.empty_cache()

        logprobs_all.append(torch.stack(group_logprobs))

    return logprobs_all