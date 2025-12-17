"""
Main script for running circuit discovery experiments.
Compares which circuits are reinforced by SFT vs RL.

Usage:
    python run_circuit_analysis.py --task math --sft_checkpoint <path> --rl_checkpoint <path>

CORRECTED VERSION:
- Fixed CMAP to use counterfactual_examples (with answers) instead of test_texts (questions only)
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circuit.discovery import (
    CircuitDiscovery,
    CrossModelCircuitAnalysis,
    create_counterfactual_examples_math,
    save_circuit_results
)
from circuit.checkpoint_loader import setup_circuit_analysis_models
from config import MODEL_NAME
from load_data import load_datasets


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
    counterfactual_examples = create_counterfactual_examples_math(
        dataset,
        n_examples=args.max_examples
    )

    if len(counterfactual_examples) == 0:
        print("❌ ERROR: No counterfactuals created!")
        sys.exit(1)

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

    # FIXED: Use counterfactual_examples which already have question/answer structure
    # The old code used test_texts (strings only) which caused CMAP to skip all examples
    # because the answer field was empty for string inputs.
    print("\nRunning Cross-Model Activation Patching (CMAP)...")
    cmap_results = cross_analysis.cross_model_activation_patching(
        base_circuit,
        counterfactual_examples,  # FIXED: was test_texts (strings without answers)
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

    # Binary Circuit Analysis (for clearer comparison)
    print("\n" + "="*70)
    print("BINARY CIRCUIT ANALYSIS")
    print("="*70)

    # Convert to binary (top-k heads are "in" the circuit)
    base_heads_binary = set((s.layer, s.head) for s in base_circuit[:args.top_k_heads])
    sft_heads_binary = set((s.layer, s.head) for s in sft_circuit[:args.top_k_heads])
    rl_heads_binary = set((s.layer, s.head) for s in rl_circuit[:args.top_k_heads])

    # Calculate overlap with base circuit
    sft_overlap_count = len(base_heads_binary & sft_heads_binary)
    rl_overlap_count = len(base_heads_binary & rl_heads_binary)

    sft_overlap_pct = (sft_overlap_count / len(base_heads_binary)) * 100 if base_heads_binary else 0
    rl_overlap_pct = (rl_overlap_count / len(base_heads_binary)) * 100 if base_heads_binary else 0

    print(f"\nBinary Circuit Preservation (top-{args.top_k_heads} heads):")
    print(f"  Base circuit: {len(base_heads_binary)} heads")
    print(f"  SFT preserves: {sft_overlap_count}/{len(base_heads_binary)} heads ({sft_overlap_pct:.1f}%)")
    print(f"  RL preserves:  {rl_overlap_count}/{len(base_heads_binary)} heads ({rl_overlap_pct:.1f}%)")
    print(f"  RL advantage: +{rl_overlap_pct - sft_overlap_pct:.1f} percentage points")

    # Identify heads unique to each method
    sft_unique = sft_heads_binary - base_heads_binary
    rl_unique = rl_heads_binary - base_heads_binary
    base_lost_in_sft = base_heads_binary - sft_heads_binary
    base_lost_in_rl = base_heads_binary - rl_heads_binary

    print(f"\nCircuit Changes:")
    print(f"  SFT lost {len(base_lost_in_sft)} base heads, gained {len(sft_unique)} new heads")
    print(f"  RL lost {len(base_lost_in_rl)} base heads, gained {len(rl_unique)} new heads")

    # Add binary analysis to results
    results['binary_analysis'] = {
        'base_circuit_size': len(base_heads_binary),
        'sft_overlap_count': int(sft_overlap_count),
        'rl_overlap_count': int(rl_overlap_count),
        'sft_overlap_pct': float(sft_overlap_pct),
        'rl_overlap_pct': float(rl_overlap_pct),
        'rl_advantage': float(rl_overlap_pct - sft_overlap_pct),
        'sft_lost_heads': len(base_lost_in_sft),
        'rl_lost_heads': len(base_lost_in_rl),
        'sft_new_heads': len(sft_unique),
        'rl_new_heads': len(rl_unique)
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
    datasets = load_datasets()
    dataset = datasets[args.task]
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