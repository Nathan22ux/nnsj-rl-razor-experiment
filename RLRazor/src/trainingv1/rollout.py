import os
import gc
from quopri import decode
import torch
import torch.nn.functional as F
from transformers import GenerationConfig

def generate_group_samples(model, tokenizer, prompts, group_size = 64, max_new_tokens = 128, temperature = 0.6):
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
        logprobs_group = []
        for sq in outputs:
            # Get only the generated portion (exclude prompt)
            generated_tokens = sq[prompt_length:]

            if len(generated_tokens) == 0:
                # No tokens generated, assign zero log prob
                logprobs_group.append(torch.tensor(0.0, device=device))
                continue

            # Forward pass on full sequence to get logits
            sq_full = sq.unsqueeze(0)
            out = model(sq_full)
            logits = out.logits  # [1, seq_len, vocab_size]

            # Get logits that predict the generated tokens
            # logits[i] predicts token[i+1], so we need logits[prompt_length-1:seq_len-1]
            pred_logits = logits[0, prompt_length-1:-1, :]  # [gen_len, vocab_size]
            print("pred logits: ", pred_logits)

            # Compute log probabilities
            log_probs = F.log_softmax(pred_logits, dim=-1)  # [gen_len, vocab_size]

            # Gather log probs for actual generated tokens
            token_log_probs = log_probs[range(len(generated_tokens)), generated_tokens]

            # Sum log probabilities (NOT average)
            total_logprob = token_log_probs.sum()

            logprobs_group.append(total_logprob)

        logprobs_groups.append(torch.stack(logprobs_group).to(device))

    return generations, logprobs_groups
