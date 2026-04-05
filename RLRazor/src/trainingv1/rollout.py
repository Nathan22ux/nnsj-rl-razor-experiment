import torch
import torch.nn.functional as F
from transformers import GenerationConfig


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

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        num_return_sequences=group_size,
        pad_token_id=tokenizer.eos_token_id,
    )

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = inputs['input_ids'].shape[1]
        prompt_lengths.append(prompt_length)

        outputs = model.generate(**inputs, generation_config=generation_config)

        # Decode ONLY the generated tokens (strip the prompt) so that reward
        # functions see only the model's completion, not the prompt text.
        # This prevents false-positive rewards when the prompt itself contains
        # answer keywords (e.g. MCQ options with "A.", "B.", "C." in the question).
        generated_only = outputs[:, prompt_length:]
        decoded = tokenizer.batch_decode(generated_only, skip_special_tokens=True)
        generations.append(decoded)

        # Store on CPU immediately to free GPU memory
        group_token_ids = [sq.detach().cpu() for sq in outputs]
        generated_token_ids.append(group_token_ids)

    # Single cleanup after all prompts — not inside the loop
    torch.cuda.empty_cache()

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

        # Process in micro-batches to avoid OOM
        micro_batch_size = 2
        group_size_current = len(group_seqs)

        for i in range(0, group_size_current, micro_batch_size):
            mb_seqs = group_seqs[i:i+micro_batch_size]
            mb_adv = adv[i:i+micro_batch_size]

            mb_batch = torch.stack(mb_seqs).to(device)
            mb_gen_tokens = mb_batch[:, prompt_len:]
            mb_gen_len = mb_gen_tokens.shape[1]

            out = model(mb_batch)
            logits = out.logits

            pred_logits = logits[:, prompt_len - 1:-1, :]
            log_probs = F.log_softmax(pred_logits, dim=-1)

            token_log_probs = log_probs.gather(2, mb_gen_tokens.unsqueeze(2)).squeeze(2)

            # Dr.GRPO: use TOTAL log-probability of the response, not per-token average.
            # Dividing by seq_length squashes gradients for longer responses and
            # prevents the model from learning complex reasoning chains.
            total_logprobs = token_log_probs.sum(dim=1)

            mb_losses = -mb_adv * total_logprobs / total_sequences
            mb_loss = mb_losses.sum()

            mb_loss.backward()
            total_loss_val += mb_loss.item()

            del out, logits, pred_logits, log_probs, token_log_probs
            del mb_batch, mb_gen_tokens, mb_losses, mb_loss
            
        del group_batch, gen_tokens_batch

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