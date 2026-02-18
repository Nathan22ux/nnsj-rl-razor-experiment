import os
import gc
from quopri import decode
import torch
import torch.nn.functional as F
from transformers import GenerationConfig

def generate_group_samples(model, tokenizer, prompts, group_size = 64, max_new_tokens = 512, temperature = 0.6):
    """
    Group Sampling function for Dr.Grpo.
    for each prompt, generate `group_size` samples.
    Returns:
        generations: list of lists of generated samples.
        logprobs: list of lists of log probabilities.
    """

    device = model.device
    generations = []
    logprobs_groups = []

    generation_config = GenerationConfig(
        max_new_tokens = max_new_tokens,
        do_sample = True,
        temperature = temperature,
        top_p = 0.8,
        num_return_sequences = group_size,
        pad_token_id = tokenizer.eos_token_id,
    )

    for prompt in prompts:
        # Encoder of inputs
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config = generation_config,
            )
        
        # Decoder of generations
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generations.append(decoded)

        # Compute log probabilities
        # NEW
        out = model(outputs)  # single batched forward pass [group_size, seq_len, vocab]
        logits = out.logits

        logprobs_group = []
        for j, sq in enumerate(outputs):
            generated_tokens = sq[prompt_length:]

            if len(generated_tokens) == 0:
                logprobs_group.append(torch.tensor(0.0, device=device))
                continue

            pred_logits = logits[j, prompt_length-1:-1, :]
            log_probs = F.log_softmax(pred_logits, dim=-1)
            token_log_probs = log_probs[range(len(generated_tokens)), generated_tokens]
            total_logprob = token_log_probs.sum()
            logprobs_group.append(total_logprob)

        logprobs_groups.append(torch.stack(logprobs_group).to(device))

    return generations, logprobs_groups