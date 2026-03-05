import logging
import time

import torch
import torch.nn.functional as F
from transformers import GenerationConfig

logger = logging.getLogger(__name__)

def generate_group_samples(
    model,
    tokenizer,
    prompts,
    group_size=64,
    max_new_tokens=512,
    temperature=0.6,
    logprob_batch_size=1,
):
    """
    Group Sampling function for Dr.Grpo.
    for each prompt, generate `group_size` samples.
    Returns:
        generations: list of lists of generated samples.
        logprobs: list of Tensors of shape [group_size] with total log-probs.
    """

    device = model.device
    generations = []
    logprobs_groups = []

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.8,
        num_return_sequences=group_size,
        pad_token_id=tokenizer.eos_token_id,
    )

    for p_idx, prompt in enumerate(prompts):
        step_start = time.time()

        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = inputs["input_ids"].shape[1]

        # --- Generation (no grad, this is sampling only) ---
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
            )

        gen_elapsed = time.time() - step_start
        avg_gen_len = outputs.shape[1] - prompt_length

        # Decode text (move to CPU to free GPU memory for forward pass)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generations.append(decoded)

        # --- Log-prob computation (needs gradients for training) ---
        lp_start = time.time()
        logprobs_group = []
        seq_count = outputs.shape[0]
        batch_size = max(1, int(logprob_batch_size))

        for start in range(0, seq_count, batch_size):
            chunk = outputs[start : start + batch_size]

            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(chunk, use_cache=False).logits
            else:
                logits = model(chunk, use_cache=False).logits

            # Vectorized log-prob extraction for the whole chunk at once
            pred_logits = logits[:, prompt_length - 1 : -1, :]  # [B, gen_len, vocab]
            log_probs = F.log_softmax(pred_logits, dim=-1)

            generated_tokens = chunk[:, prompt_length:]  # [B, gen_len]

            # Gather log-probs for the actual generated tokens
            token_log_probs = log_probs.gather(
                dim=-1, index=generated_tokens.unsqueeze(-1)
            ).squeeze(-1)  # [B, gen_len]

            total_logprobs = token_log_probs.sum(dim=-1)  # [B]

            logprobs_group.append(total_logprobs)

            del logits, log_probs, pred_logits, token_log_probs, chunk

        logprobs_groups.append(torch.cat(logprobs_group))

        lp_elapsed = time.time() - lp_start
        logger.info(
            "  Prompt %s/%s: gen=%.1fs (%s tok avg), logprobs=%.1fs, total=%.1fs",
            p_idx + 1,
            len(prompts),
            gen_elapsed,
            avg_gen_len,
            lp_elapsed,
            time.time() - step_start,
        )

        # Free generation tensor
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return generations, logprobs_groups
