import os
import gc
import torch
import torch.nn.functional as F
from transformers import GenerationConfig

def generate_group_samples(model, tokenizer, prompts, group_size = 64, max_new_tokens = 512, temperature = 0.6, chunk_size = 16):
    """
    Group Sampling function for Dr.Grpo.
    for each prompt, generate `group_size` samples.
    Args:
        chunk_size: Forward-pass mini-batch size for log-prob computation.
                    Higher = faster but more VRAM. Default 16 is conservative;
                    on large GPUs (e.g. GH200 480GB) you can safely use 32-64.
    Returns:
        generations: list of lists of generated text (decoded strings).
        logprobs: list of Tensors, each shape [group_size] (total log-prob per sample).
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

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = inputs['input_ids'].shape[1]

        # Generate group_size samples
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
            )

        # Decode to strings (moves text to CPU, no GPU cost)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generations.append(decoded)

        # Compute log probabilities in chunks to avoid OOM
        # Process in mini-batches to limit GPU memory for the forward pass
        chunk_size = min(chunk_size, group_size)
        logprobs_group = []

        for chunk_start in range(0, group_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, group_size)
            chunk_ids = outputs[chunk_start:chunk_end]

            with torch.no_grad():
                chunk_out = model(chunk_ids)
                chunk_logits = chunk_out.logits

            for j in range(chunk_ids.shape[0]):
                generated_tokens = chunk_ids[j, prompt_length:]

                if len(generated_tokens) == 0:
                    logprobs_group.append(torch.tensor(0.0, device=device))
                    continue

                pred_logits = chunk_logits[j, prompt_length-1:-1, :]
                log_probs = F.log_softmax(pred_logits, dim=-1)
                token_log_probs = log_probs[range(len(generated_tokens)), generated_tokens]
                total_logprob = token_log_probs.sum()
                logprobs_group.append(total_logprob)

            # Free chunk tensors
            del chunk_out, chunk_logits
            torch.cuda.empty_cache()

        logprobs_groups.append(torch.stack(logprobs_group).to(device))

        # Free outputs tensor after processing each prompt
        del outputs, inputs
        torch.cuda.empty_cache()

    return generations, logprobs_groups
