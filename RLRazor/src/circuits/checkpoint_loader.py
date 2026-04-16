"""
Custom model loader for your specific checkpoint format.
Adapts circuit analysis to work with your training setup.
"""

import os
import glob
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen2ForCausalLM


def _load_qwen(path, **kwargs):
    """Load Qwen2.5-3B directly — avoids AutoModel registry issues on transformers 5.x."""
    # Allow ~28GB per model on GPU, rest spills to CPU (3 models = ~84GB GPU max)
    kwargs.setdefault("max_memory", {0: "28GiB", "cpu": "60GiB"})
    return Qwen2ForCausalLM.from_pretrained(path, **kwargs)


def find_best_checkpoint(results_dir, method='sft', metric='best'):
    """
    Find the best checkpoint from your training results.

    Args:
        results_dir: Path to results directory (e.g., './results')
        method: 'sft' or 'grpo'
        metric: How to choose best ('best', 'last', or specific lr/bs combo)

    Returns:
        Path to best checkpoint directory
    """
    print(f"\nSearching for {method.upper()} checkpoints in {results_dir}...")

    if method == 'sft':
        pattern = f"{results_dir}/sft_lr*"
    else:
        pattern = f"{results_dir}/grpo_lr*"

    checkpoint_dirs = glob.glob(pattern)

    if not checkpoint_dirs:
        raise ValueError(f"No {method} checkpoints found in {results_dir}! Pass --sft_checkpoint and --rl_checkpoint explicitly.")

    print(f"Found {len(checkpoint_dirs)} {method} checkpoint(s):")
    for d in checkpoint_dirs:
        print(f"  - {os.path.basename(d)}")

    best_checkpoint = checkpoint_dirs[0]

    # Find sub-checkpoint if present (e.g. checkpoint-XXX/)
    epoch_checkpoints = glob.glob(f"{best_checkpoint}/checkpoint-*")
    if epoch_checkpoints:
        best_checkpoint = sorted(epoch_checkpoints)[-1]

    print(f"Using checkpoint: {best_checkpoint}")
    return best_checkpoint


def _find_weights_dir(checkpoint_path):
    """
    Given a checkpoint path, find the directory that actually contains weight files.
    Handles cases where weights are one level deeper or are relative symlinks.
    """
    # Always work with absolute path so relative symlinks resolve correctly
    checkpoint_path = os.path.realpath(os.path.abspath(checkpoint_path))

    # Check the given path directly
    for wf in ["model.safetensors", "pytorch_model.bin", "config.json"]:
        if os.path.exists(os.path.join(checkpoint_path, wf)):
            return checkpoint_path

    # Check one level deeper — skip non-directory files like README.md
    for subdir in sorted(os.listdir(checkpoint_path)):
        full = os.path.join(checkpoint_path, subdir)
        if not os.path.isdir(full):
            continue  # skip files (README.md, etc.)
        for wf in ["model.safetensors", "pytorch_model.bin", "config.json"]:
            if os.path.exists(os.path.join(full, wf)):
                print(f"Found weights in subdirectory: {full}")
                return full

    return checkpoint_path  # Return original and let from_pretrained raise a clear error


def load_your_checkpoint(checkpoint_path, base_model_name="Qwen/Qwen2.5-3B-Instruct"):
    """
    Load a Qwen2.5 checkpoint saved by your training script.

    Args:
        checkpoint_path: Path to checkpoint directory (or its parent)
        base_model_name: Used as fallback if checkpoint has no config.json

    Returns:
        Loaded model
    """
    checkpoint_path = os.path.realpath(os.path.abspath(checkpoint_path))
    weights_dir = _find_weights_dir(checkpoint_path)
    print(f"Loading checkpoint from: {weights_dir}")

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # If checkpoint has its own config.json, load directly
    if os.path.exists(os.path.join(weights_dir, "config.json")):
        model = _load_qwen(weights_dir, **load_kwargs)
        print("Checkpoint loaded successfully")
        return model

    # No config.json — load base model architecture then apply saved weights
    print("No config.json in checkpoint, loading base model architecture...")
    model = _load_qwen(base_model_name, **load_kwargs)

    safe_file = os.path.join(weights_dir, "model.safetensors")
    bin_file = os.path.join(weights_dir, "pytorch_model.bin")

    if os.path.exists(safe_file):
        state_dict = load_file(safe_file)
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded from model.safetensors")
    elif os.path.exists(bin_file):
        state_dict = torch.load(bin_file, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded from pytorch_model.bin")
    else:
        raise FileNotFoundError(
            f"No weight files found in {weights_dir}. "
            f"Contents: {os.listdir(weights_dir)}"
        )

    return model


def setup_circuit_analysis_models(
    base_model_name="Qwen/Qwen2.5-3B-Instruct",
    results_dir="./results",
    sft_checkpoint=None,
    grpo_checkpoint=None
):
    """
    Load all three models needed for circuit analysis: base, SFT, and RL.

    Returns:
        tuple: (base_model, sft_model, grpo_model, tokenizer)
    """
    print("\n" + "="*70)
    print("LOADING MODELS FOR CIRCUIT ANALYSIS")
    print("="*70)

    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")

    print("\n2. Loading base model...")
    base_model = _load_qwen(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Base model loaded")

    print("\n3. Loading SFT model...")
    if sft_checkpoint is None:
        sft_checkpoint = find_best_checkpoint(results_dir, method='sft')
    sft_model = load_your_checkpoint(sft_checkpoint, base_model_name)
    print("SFT model loaded")

    print("\n4. Loading GRPO (RL) model...")
    if grpo_checkpoint is None:
        grpo_checkpoint = find_best_checkpoint(results_dir, method='grpo')
    grpo_model = load_your_checkpoint(grpo_checkpoint, base_model_name)
    print("GRPO model loaded")

    print("\n" + "="*70)
    print("ALL MODELS LOADED SUCCESSFULLY")
    print("="*70)

    return base_model, sft_model, grpo_model, tokenizer


def load_models_for_circuit_analysis(base_model_name, sft_checkpoint, rl_checkpoint, device="cuda"):
    """Wrapper matching signature expected by run_circuit_analysis.py"""
    if sft_checkpoint and rl_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = _load_qwen(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        sft_model = load_your_checkpoint(sft_checkpoint, base_model_name)
        rl_model = load_your_checkpoint(rl_checkpoint, base_model_name)

        return base_model, sft_model, rl_model, tokenizer
    else:
        return setup_circuit_analysis_models(base_model_name)
