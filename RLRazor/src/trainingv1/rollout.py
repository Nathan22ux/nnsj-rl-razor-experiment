import torch
from transformers import GenerationConfig

def generate_group_samples(
    model,
    tokenizer,
    prompts,
    group_size=64,
    max_new_tokens=512,
    temperature=0.6,
    chunk_size=8,
    gen_chunk_size=8,
):
    """
    Group Sampling function for Dr.GRPO.
    For each prompt, generate `group_size` samples.

    Args:
        chunk_size:     Forward-pass mini-batch size for log-prob computation.
        gen_chunk_size: How many sequences to generate at a time.
                        Smaller = less VRAM during generation. Default 8 is
                        conservative; on large GPUs you can raise this.
    Returns:
        generations:  list of lists of generated text (decoded strings).
        logprobs:     list of Tensors, each shape [group_size].
    """
    device = model.device
    generations = []
    logprobs_groups = []

    # Single-sequence generation config — we loop gen_chunk_size at a time
    gen_chunk_size = min(gen_chunk_size, group_size)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.8,
        num_return_sequences=gen_chunk_size,
        pad_token_id=tokenizer.eos_token_id,
    )

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = inputs["input_ids"].shape[1]

        all_decoded = []
        logprobs_group = []

        # ── Step 1: generate + compute logprobs in sub-batches ──────────────
        for gen_start in range(0, group_size, gen_chunk_size):
            this_gen_size = min(gen_chunk_size, group_size - gen_start)

            # Adjust num_return_sequences for the last (possibly smaller) chunk
            if this_gen_size != gen_chunk_size:
                gc = GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.8,
                    num_return_sequences=this_gen_size,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                gc = generation_config

            # Generate this_gen_size sequences
            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=gc)

            # Decode immediately (CPU, frees GPU activation memory)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_decoded.extend(decoded)

            # ── Step 2: compute logprobs for this gen sub-batch ─────────────
            lp_chunk_size = min(chunk_size, this_gen_size)
            for lp_start in range(0, this_gen_size, lp_chunk_size):
                lp_end = min(lp_start + lp_chunk_size, this_gen_size)
                chunk_ids = outputs[lp_start:lp_end]

                chunk_out = model(chunk_ids, use_cache=False)
                chunk_logits = chunk_out.logits

                for j in range(chunk_ids.shape[0]):
                    generated_tokens = chunk_ids[j, prompt_length:]

                    if len(generated_tokens) == 0:
                        logprobs_group.append(torch.tensor(0.0, device=device))
                        continue

                    pred_logits = chunk_logits[j, prompt_length - 1 : -1, :]
                    token_logits = pred_logits.gather(
                        -1, generated_tokens.unsqueeze(-1)
                    ).squeeze(-1)
                    token_log_probs = token_logits - torch.logsumexp(pred_logits, dim=-1)
                    logprobs_group.append(token_log_probs.sum())

                del chunk_out, chunk_logits
                torch.cuda.empty_cache()

            # Free this generation's outputs before next sub-batch
            del outputs
            torch.cuda.empty_cache()

        generations.append(all_decoded)
        logprobs_groups.append(torch.stack(logprobs_group).to(device))

        del inputs
        torch.cuda.empty_cache()

    return generations, logprobs_groups

