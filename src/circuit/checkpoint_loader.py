"""
Custom model loader for your specific checkpoint format.
Adapts circuit analysis to work with your training setup.
"""

import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from safetensors.torch import load_file


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
    print(f"\n🔍 Searching for {method.upper()} checkpoints in {results_dir}...")
    
    # Pattern to match your checkpoint directories
    if method == 'sft':
        pattern = f"{results_dir}/sft_lr*"
    else:  # grpo
        pattern = f"{results_dir}/grpo_lr*"
    
    checkpoint_dirs = glob.glob(pattern)
    
    if not checkpoint_dirs:
        raise ValueError(f"No {method} checkpoints found in {results_dir}!")
    
    print(f"Found {len(checkpoint_dirs)} {method} checkpoint(s):")
    for d in checkpoint_dirs:
        print(f"  - {os.path.basename(d)}")
    
    # For now, just return the first one (you can add logic to pick best)
    # You could read results_math.json and pick the one with highest NT score
    best_checkpoint = checkpoint_dirs[0]
    
    # Find the actual model checkpoint inside (usually checkpoint-XXX/)
    epoch_checkpoints = glob.glob(f"{best_checkpoint}/checkpoint-*")
    
    if epoch_checkpoints:
        # Use the last checkpoint (highest epoch)
        best_checkpoint = sorted(epoch_checkpoints)[-1]
        print(f"\n✓ Using checkpoint: {best_checkpoint}")
    else:
        print(f"\n✓ Using checkpoint directory: {best_checkpoint}")
    
    return best_checkpoint


def load_your_checkpoint(checkpoint_path, base_model_name="Qwen/Qwen2.5-3B-Instruct"):
    """
    Load a checkpoint saved by your training script.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        base_model_name: Base model name from config.py (Qwen/Qwen2.5-3B-Instruct)
    
    Returns:
        Loaded model
    """
    print(f"\n📦 Loading model from checkpoint: {checkpoint_path}")
    
    try:
        # Try to load directly from checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✓ Model loaded successfully from checkpoint")
        
    except Exception as e:
        print(f"⚠️  Direct loading failed: {e}")
        print(f"Trying to load base model and apply checkpoint...")
        
        # Load base model first
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Try to load state dict from checkpoint
        checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")
        # New type of checkpoint added 
        safe_file = os.path.join(checkpoint_path, "pytorch_model.safetensors")
        if os.path.exists(checkpoint_file):
            state_dict = torch.load(checkpoint_file, map_location="cpu")
            model.load_state_dict(state_dict)
            print("✓ Model loaded with state dict from checkpoint")
        elif os.path.exists(safe_file):
            state_dict = load_file(safe_file)
            model.load_state_dict(state_dict)
            print("✓ Model loaded with state dict from checkpoint")
        else:
            print("⚠️  No pytorch_model.bin found, using base model")
    
    return model


def setup_circuit_analysis_models(
    base_model_name="Qwen/Qwen2.5-3B-Instruct",
    results_dir="./results",
    sft_checkpoint=None,
    grpo_checkpoint=None
):
    """
    Load all three models needed for circuit analysis: base, SFT, and RL.
    
    Args:
        base_model_name: Base model from config.py (Qwen/Qwen2.5-3B-Instruct)
        results_dir: Your results directory
        sft_checkpoint: Specific SFT checkpoint path (optional, will auto-find if None)
        grpo_checkpoint: Specific GRPO checkpoint path (optional, will auto-find if None)
    
    Returns:
        tuple: (base_model, sft_model, grpo_model, tokenizer)
    """
    print("\n" + "="*70)
    print("LOADING MODELS FOR CIRCUIT ANALYSIS")
    print("="*70)
    
    # Load tokenizer (same for all models)
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    
    # Load base model
    print("\n2. Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✓ Base model loaded")
    
    # Load SFT model
    print("\n3. Loading SFT model...")
    if sft_checkpoint is None:
        sft_checkpoint = find_best_checkpoint(results_dir, method='sft')
    sft_model = load_your_checkpoint(sft_checkpoint, base_model_name)
    print("✓ SFT model loaded")
    
    # Load GRPO model
    print("\n4. Loading GRPO (RL) model...")
    if grpo_checkpoint is None:
        grpo_checkpoint = find_best_checkpoint(results_dir, method='grpo')
    grpo_model = load_your_checkpoint(grpo_checkpoint, base_model_name)
    print("✓ GRPO model loaded")
    
    print("\n" + "="*70)
    print("ALL MODELS LOADED SUCCESSFULLY")
    print("="*70)
    
    return base_model, sft_model, grpo_model, tokenizer


# Convenience function for run_circuit_analysis.py
def load_models_for_circuit_analysis(base_model_name, sft_checkpoint, rl_checkpoint, device="cuda"):
    """
    Wrapper function that matches the signature expected by run_circuit_analysis.py
    """
    # If paths are provided, use them directly
    if sft_checkpoint and rl_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        sft_model = load_your_checkpoint(sft_checkpoint, base_model_name)
        rl_model = load_your_checkpoint(rl_checkpoint, base_model_name)
        
        return base_model, sft_model, rl_model, tokenizer
    else:
        # Auto-find checkpoints
        return setup_circuit_analysis_models(base_model_name)


if __name__ == "__main__":
    """Test the checkpoint loading"""
    print("Testing checkpoint loading...")
    
    try:
        # Try to find and load checkpoints
        base, sft, rl, tok = setup_circuit_analysis_models()
        print("\n✅ Successfully loaded all models!")
        print(f"Base model: {base.config._name_or_path}")
        print(f"Number of parameters: {base.num_parameters() / 1e6:.1f}M")
        
    except Exception as e:
        print(f"\n❌ Error loading checkpoints: {e}")
        print("\nMake sure you have run training first:")
        print("  python main.py --task math --method sft")
        print("  python main.py --task math --method rl")