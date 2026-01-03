"""
Test script for circuit discovery implementation.
Tests basic functionality with Qwen model.

Usage:
    python test_circuits.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circuit.discovery import CircuitDiscovery


def test_basic_circuit_discovery():
    """Test basic circuit discovery functionality"""
    print("\n" + "="*70)
    print("TESTING CIRCUIT DISCOVERY IMPLEMENTATION")
    print("="*70)
    
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    print(f"\nLoading model: {model_name}")
    print("(This will download the model if not cached)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("✓ Model loaded successfully!")
    
    # Test 1: Initialization
    print("\n" + "-"*70)
    print("TEST 1: Circuit Discovery Initialization")
    print("-"*70)
    
    discovery = CircuitDiscovery(model, tokenizer)
    print(f"✓ CircuitDiscovery initialized")
    print(f"  Architecture: {discovery.arch_style}")
    print(f"  Layers: {discovery.n_layers}")
    print(f"  Heads per layer: {discovery.n_heads}")
    
    # Test 2: Activation extraction
    print("\n" + "-"*70)
    print("TEST 2: Activation Extraction")
    print("-"*70)
    
    test_text = "What is the capital of France?"
    input_ids = tokenizer(test_text, return_tensors="pt").input_ids.to(model.device)
    
    activations = discovery.extract_activations(input_ids)
    print(f"✓ Extracted activations for {len(activations)} heads")
    
    # Test 3: Path patching
    print("\n" + "-"*70)
    print("TEST 3: Path Patching")
    print("-"*70)
    
    original_text = "The capital of France is Paris."
    counterfactual_text = "The capital of Germany is Berlin."
    
    original_ids = tokenizer(original_text, return_tensors="pt").input_ids.to(model.device)
    counterfactual_ids = tokenizer(counterfactual_text, return_tensors="pt").input_ids.to(model.device)
    
    original_acts = discovery.extract_activations(original_ids)
    counterfactual_acts = discovery.extract_activations(counterfactual_ids)

    patched_logits = discovery.path_patch_head(
        original_ids, counterfactual_ids,
        5, 3,
        original_acts, counterfactual_acts
    )
    
    print(f"✓ Successfully patched Layer 5, Head 3")
    
    # Test 4: Circuit identification (minimal)
    print("\n" + "-"*70)
    print("TEST 4: Circuit Identification (Minimal)")
    print("-"*70)

    examples = [
        {
            'question': "What is 2 + 2?",
            'answer': "4",
            'counterfactual_question': "What is 3 + 3?"
        },
        {
            'question': "What is the capital of France?",
            'answer': "Paris",
            'counterfactual_question': "What is the capital of Germany?"
        },
        {
            'question': "What color is the sky?",
            'answer': "Blue",
            'counterfactual_question': "What color is grass?"
        },
    ]
    
    print(f"Using {len(examples)} example pairs (quick test)")
    
    circuit = discovery.identify_circuit(
        examples,
        top_k=5,
        max_examples=3
    )
    
    print(f"✓ Successfully identified circuit with {len(circuit)} heads")
    print("\nTop 3 heads:")
    for i, score in enumerate(circuit[:3], 1):
        print(f"  {i}. Layer {score.layer}, Head {score.head}: score = {score.score:.4f}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nCircuit discovery is working correctly with Qwen.")
    print("You can now run the full analysis with your trained models.")
    
    # Clean up
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        test_basic_circuit_discovery()
    except Exception as e:
        print("\n" + "="*70)
        print("❌ ERROR DURING TESTING")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)