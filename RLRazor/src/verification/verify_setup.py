"""
Comprehensive verification script for Qwen/Qwen2.5-3B-Instruct.
Tests EVERYTHING before running full circuit analysis.

Usage:
    python verify_setup.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circuits.discovery import DCMAnalysis
from config.CONFIG import MODEL_NAME


def test_1_model_loading():
    """Test 1: Can we load Qwen?"""
    print("\n" + "="*70)
    print("TEST 1: MODEL LOADING")
    print("="*70)
    
    model_name = MODEL_NAME
    
    try:
        print(f"\n Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✓ Model loaded")
        
        print(f"\n Model Info:")
        print(f"  Layers: {model.config.num_hidden_layers}")
        print(f"  Attention heads: {model.config.num_attention_heads}")
        print(f"  Hidden size: {model.config.hidden_size}")
        print(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")
        
        return True, model, tokenizer
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False, None, None


def test_2_architecture_detection(model, tokenizer):
    """Test 2: Does circuit discovery detect Qwen correctly?"""
    print("\n" + "="*70)
    print("TEST 2: ARCHITECTURE DETECTION")
    print("="*70)
    
    try:
        from circuits.discovery import DCMAnalysis
        
        print("\n Initializing CircuitDiscovery...")
        discovery = DCMAnalysis(model, tokenizer)
        
        print(f"\n✓ Architecture detected: {discovery.arch_style}")
        
        if discovery.arch_style != 'llama':
            print(f"\n⚠️  WARNING: Expected 'llama' style for Qwen, got '{discovery.arch_style}'")
            return False
        
        print(f"✓ Correct architecture style for Qwen")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_circuit_identification(model, tokenizer):
    """Test 3: Can we identify circuits?"""
    print("\n" + "="*70)
    print("TEST 3: CIRCUIT IDENTIFICATION (Small Test)")
    print("="*70)
    
    try:
        from circuits.discovery import DCMAnalysis
        
        discovery = DCMAnalysis(model, tokenizer)
        
        examples = [
            ("What is 2+2?", "What is 3+3?"),
            ("Paris is in France.", "London is in England."),
        ]
        
        print(f"\n Running circuit identification with {len(examples)} examples...")
        
        circuit = # identify_circuit removed — use dcm.train_circuit_mask(examples)
        
        print(f"\n✓ Circuit identified: {len(circuit)} heads")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_config_integration():
    """Test 4: Is config.py set to Qwen?"""
    print("\n" + "="*70)
    print("TEST 4: CONFIG INTEGRATION")
    print("="*70)
    
    try:
        from config.CONFIG import MODEL_NAME
        
        print(f"\n config.py MODEL_NAME: {MODEL_NAME}")
        
        if "Qwen" in MODEL_NAME:
            print("✓ Config correctly set to Qwen")
            return True
        else:
            print(f"\n⚠️  WARNING: Config is set to {MODEL_NAME}")
            return False
            
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🔍 COMPREHENSIVE QWEN VERIFICATION")
    print("="*70)
    
    results = {}
    
    # Test 1
    success, model, tokenizer = test_1_model_loading()
    results['Model Loading'] = success
    
    if not success:
        print("\n❌ CRITICAL: Cannot load model")
        return False
    
    # Test 2
    results['Architecture Detection'] = test_2_architecture_detection(model, tokenizer)
    
    # Test 3
    results['Circuit Identification'] = test_3_circuit_identification(model, tokenizer)
    
    # Test 4
    results['Config Integration'] = test_4_config_integration()
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\nYou're ready to run circuit analysis!")
        return True
    else:
        print("⚠️  SOME TESTS FAILED")
        print("="*70)
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)