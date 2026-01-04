"""
Verification script to ensure activation extraction is working correctly.
Run this before doing any circuit analysis.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from circuits.discovery import CircuitDiscovery

def verify_activation_extraction():
    """Verify that per-head activations are actually different."""

    print("Loading model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Use small model for testing
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    discovery = CircuitDiscovery(model, tokenizer)

    # Test input
    test_input = "What is 2 + 2?"
    input_ids = tokenizer(test_input, return_tensors="pt").input_ids.to(model.device)

    print("\nExtracting activations...")
    activations = discovery.extract_activations(input_ids)

    print(f"\nExtracted {len(activations)} head activations")

    # Verify heads are different
    print("\nVerifying heads have different activations...")
    layer_0_heads = [activations[(0, h)] for h in range(min(4, discovery.n_heads))]

    all_same = True
    for i in range(len(layer_0_heads)):
        for j in range(i + 1, len(layer_0_heads)):
            diff = (layer_0_heads[i] - layer_0_heads[j]).abs().mean().item()
            if diff > 1e-6:
                all_same = False
            print(f"  Head 0 vs Head {j}: mean diff = {diff:.6f}")

    if all_same:
        print("\n❌ PROBLEM: All heads have identical activations!")
        print("   This means activation extraction is still broken.")
        return False
    else:
        print("\n✅ SUCCESS: Heads have different activations!")
        return True

def verify_patching_changes_output():
    """Verify that patching actually changes model output."""

    print("\n" + "="*50)
    print("Verifying patching changes output...")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    discovery = CircuitDiscovery(model, tokenizer)

    original = "What is 2 + 2?"
    counterfactual = "What is 5 + 5?"

    orig_ids = tokenizer(original, return_tensors="pt").input_ids.to(model.device)
    cf_ids = tokenizer(counterfactual, return_tensors="pt").input_ids.to(model.device)

    # Get original output
    with torch.no_grad():
        orig_logits = model(orig_ids).logits
    orig_pred = orig_logits[0, -1, :].argmax().item()

    # Extract activations
    orig_acts = discovery.extract_activations(orig_ids)
    cf_acts = discovery.extract_activations(cf_ids)

    # Patch a head and check if output changes
    patched_logits = discovery.path_patch_head(
        orig_ids, cf_ids,
        layer_idx=0, head_idx=0,
        original_activations=orig_acts,
        counterfactual_activations=cf_acts
    )
    patched_pred = patched_logits[0, -1, :].argmax().item()

    print(f"Original prediction token: {orig_pred} ({tokenizer.decode([orig_pred])})")
    print(f"Patched prediction token: {patched_pred} ({tokenizer.decode([patched_pred])})")

    logit_diff = (orig_logits - patched_logits).abs().mean().item()
    print(f"Mean logit difference: {logit_diff:.6f}")

    if logit_diff < 1e-6:
        print("\n❌ PROBLEM: Patching doesn't change output!")
        return False
    else:
        print("\n✅ SUCCESS: Patching changes model output!")
        return True

if __name__ == "__main__":
    print("="*50)
    print("ACTIVATION EXTRACTION VERIFICATION")
    print("="*50)

    test1 = verify_activation_extraction()
    test2 = verify_patching_changes_output()

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Activation extraction: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Patching verification: {'✅ PASS' if test2 else '❌ FAIL'}")

    if test1 and test2:
        print("\n✅ All tests passed! Ready for circuit analysis.")
    else:
        print("\n❌ Some tests failed. Please check the fixes.")