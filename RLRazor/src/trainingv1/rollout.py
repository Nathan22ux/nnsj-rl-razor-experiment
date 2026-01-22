import os
import gc
from quopri import decode
import torch
import torch.nn.functional as F
from transformers import GenerationConfig

@torch.no_grad()
def generate_group_samples(model, tokenizer, prompts, group_size = 64, max_new_tokens = 128, temperature = 0.6):
    """
    Group Sampling function for Dr.Grpo.
    for each prompt, generate `group_size` samples.
    Returns:
        generations: list of lists of generated samples.
        logprobs: list of lists of log probabilities.
    """

    device =  model.device
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
            sq = sq.unsqueeze(0)

            # forward pass
            out = model(sq, labels = sq)
            logprobs  = - out.loss

            logprobs_group.append(logprobs.detach())
        logprobs_groups.append(torch.stack(logprobs_group).to(device))

    return generations, logprobs_groups