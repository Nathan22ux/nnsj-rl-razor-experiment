import torch
import torch.nn.functional as F
from transformers import GenerationConfig

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

        logprobs_group = []
        seq_count = outputs.shape[0]
        batch_size = max(1, int(logprob_batch_size))
        for start in range(0, seq_count, batch_size):
            chunk = outputs[start : start + batch_size]
            # Disable KV cache during training forward to reduce memory usage.
            logits = model(chunk, use_cache=False).logits

            for j in range(chunk.shape[0]):
                sq = chunk[j]
                generated_tokens = sq[prompt_length:]

                if len(generated_tokens) == 0:
                    logprobs_group.append(torch.tensor(0.0, device=device))
                    continue

                pred_logits = logits[j, prompt_length - 1 : -1, :]
                log_probs = F.log_softmax(pred_logits, dim=-1)
                token_log_probs = log_probs[range(len(generated_tokens)), generated_tokens]
                total_logprob = token_log_probs.sum()
                logprobs_group.append(total_logprob)

            del logits, chunk

        logprobs_groups.append(torch.stack(logprobs_group))

    return generations, logprobs_groups
