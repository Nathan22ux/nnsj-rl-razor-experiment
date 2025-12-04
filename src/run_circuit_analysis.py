"""
Main script for running circuit discovery experiments.
Compares which circuits are reinforced by SFT vs RL.

Usage:
    python run_circuit_analysis.py --task math --sft_checkpoint <path> --rl_checkpoint <path>
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circuit.discovery import (
    CircuitDiscovery,
    CrossModelCircuitAnalysis,
    create_counterfactual_examples,
    save_circuit_results
)
from circuit.checkpoint_loader import setup_circuit_analysis_models
from config import MODEL_NAME
from load_data import load_dataset_for_task


def run_circuit_analysis(
    base_model,
    sft_model,
    rl_model,
    tokenizer,
    dataset,
    args
):
    """
    Run the full circuit analysis pipeline
    """
    
    print("\n" + "="*70)
    print("STARTING CIRCUIT ANALYSIS")
    print("="*70)
    
    # Create counterfactual examples
    print("\nCreating counterfactual examples...")
    counterfactual_examples = create_counterfactual_examples(
        dataset,
        n_examples=args.max_examples
    )
    print(f"Created {len(counterfactual_examples)} counterfactual pairs")
    
    # Phase 1: Base model circuits
    print("\n" + "="*70)
    print("PHASE 1: IDENTIFYING CIRCUITS IN BASE MODEL")
    print("="*70)
    
    base_discovery = CircuitDiscovery(base_model, tokenizer)
    base_circuit = base_discovery.identify_circuit(
        counterfactual_examples,
        top_k=args.top_k_heads,
        max_examples=args.max_examples
    )
    
    # Phase 2: Fine-tuned model circuits
    print("\n" + "="*70)
    print("PHASE 2: IDENTIFYING CIRCUITS IN FINE-TUNED MODELS")
    print("="*70)
    
    print("\nAnalyzing SFT model circuits...")
    sft_discovery = CircuitDiscovery(sft_model, tokenizer)
    sft_circuit = sft_discovery.identify_circuit(
        counterfactual_examples,
        top_k=args.top_k_heads,
        max_examples=args.max_examples
    )
    
    print("\nAnalyzing RL model circuits...")
    rl_discovery = CircuitDiscovery(rl_model, tokenizer)
    rl_circuit = rl_discovery.identify_circuit(
        counterfactual_examples,
        top_k=args.top_k_heads,
        max_examples=args.max_examples
    )
    
    # Phase 3: Cross-model comparison
    print("\n" + "="*70)
    print("PHASE 3: CROSS-MODEL CIRCUIT COMPARISON")
    print("="*70)
    
    cross_analysis = CrossModelCircuitAnalysis(
        base_model, sft_model, rl_model, tokenizer
    )
    
    # Extract test examples
    test_texts = []
    for i in range(min(args.max_examples, len(dataset))):
        item = dataset[i]
        if isinstance(item, dict):
            if '0' in item and isinstance(item['0'], dict):
                text = item['0'].get('value', '')
            elif 'question' in item:
                text = item['question']
            elif 'prompt' in item:
                text = item['prompt']
            else:
                text = str(item)
        else:
            text = str(item)
        test_texts.append(text)
    
    # Run CMAP
    print("\nRunning Cross-Model Activation Patching (CMAP)...")
    cmap_results = cross_analysis.cross_model_activation_patching(
        base_circuit,
        test_texts,
        max_examples=args.max_examples
    )
    
    # Phase 4: Vulnerable circuits
    print("\n" + "="*70)
    print("PHASE 4: IDENTIFYING VULNERABLE CIRCUITS")
    print("="*70)
    
    vulnerable_circuits = cross_analysis.identify_vulnerable_circuits(
        cmap_results,
        threshold=args.vulnerability_threshold
    )
    
    # Compile results
    results = {
        'config': {
            'task': args.task,
            'max_examples': args.max_examples,
            'top_k_heads': args.top_k_heads,
            'vulnerability_threshold': args.vulnerability_threshold,
            'model': args.base_model
        },
        'base_circuit': [
            {'layer': s.layer, 'head': s.head, 'importance_score': float(s.score)}
            for s in base_circuit
        ],
        'sft_circuit': [
            {'layer': s.layer, 'head': s.head, 'importance_score': float(s.score)}
            for s in sft_circuit
        ],
        'rl_circuit': [
            {'layer': s.layer, 'head': s.head, 'importance_score': float(s.score)}
            for s in rl_circuit
        ],
        'cmap_analysis': cmap_results,
        'vulnerable_circuits': vulnerable_circuits
    }
    
    # Save results
    os.makedirs("results/circuits", exist_ok=True)
    output_path = f"results/circuits/circuit_analysis_{args.task}.json"
    save_circuit_results(results, output_path)
    
    # Print summary
    print("\n" + "="*70)
    print("CIRCUIT ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nKey Findings:")
    print(f"  Base model circuit: {len(base_circuit)} heads")
    print(f"  SFT model circuit: {len(sft_circuit)} heads")
    print(f"  RL model circuit: {len(rl_circuit)} heads")
    print(f"  Vulnerable circuits: {len(vulnerable_circuits)} heads")
    
    # Overlap analysis
    base_heads = set((s.layer, s.head) for s in base_circuit)
    sft_heads = set((s.layer, s.head) for s in sft_circuit)
    rl_heads = set((s.layer, s.head) for s in rl_circuit)
    
    sft_overlap = len(base_heads & sft_heads)
    rl_overlap = len(base_heads & rl_heads)
    
    print(f"\nCircuit Overlap with Base Model:")
    print(f"  SFT: {sft_overlap}/{len(base_circuit)} heads ({100*sft_overlap/len(base_circuit):.1f}%)")
    print(f"  RL: {rl_overlap}/{len(base_circuit)} heads ({100*rl_overlap/len(base_circuit):.1f}%)")
    
    print(f"\nResults saved to: {output_path}")
    print("="*70 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run circuit discovery analysis")
    parser.add_argument("--task", type=str, default="math",
                       choices=["math", "science", "tool"],
                       help="Task to analyze")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                       help="Base model name")
    parser.add_argument("--sft_checkpoint", type=str, required=True,
                       help="Path to SFT checkpoint")
    parser.add_argument("--rl_checkpoint", type=str, required=True,
                       help="Path to RL checkpoint")
    parser.add_argument("--max_examples", type=int, default=50,
                       help="Maximum examples to use")
    parser.add_argument("--top_k_heads", type=int, default=20,
                       help="Number of top heads to identify")
    parser.add_argument("--vulnerability_threshold", type=float, default=0.1,
                       help="Threshold for vulnerable circuits")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Load models
    print(f"\nLoading models...")
    base_model, sft_model, rl_model, tokenizer = setup_circuit_analysis_models(
        base_model_name=args.base_model,
        results_dir="./results",
        sft_checkpoint=args.sft_checkpoint,
        grpo_checkpoint=args.rl_checkpoint
    )
    
    # Load dataset
    print(f"\nLoading dataset for task: {args.task}")
    dataset = load_dataset_for_task(args.task)
    print(f"Dataset loaded: {len(dataset)} examples")
    
    # Run analysis
    results = run_circuit_analysis(
        base_model,
        sft_model,
        rl_model,
        tokenizer,
        dataset,
        args
    )
    
    print("\n✅ Circuit analysis completed successfully!")
    print("\nTo visualize results, run:")
    print(f"  python visualize_circuits.py results/circuits/circuit_analysis_{args.task}.json")


if __name__ == "__main__":
    main()