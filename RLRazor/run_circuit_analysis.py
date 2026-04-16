"""
Main script for running circuit discovery experiments.
Compares which circuits are reinforced by two fine-tuned models vs base.

UPDATED VERSION:
- DCM analysis runs by default (use --skip_dcm to disable)
- All errors handled gracefully
- Faithfulness metrics always computed
- Cross-model faithfulness comparison
- Configurable model labels via --model_a_name / --model_b_name

Usage:
    python run_circuit_analysis.py --task math --sft_checkpoint <path> --rl_checkpoint <path>
    python run_circuit_analysis.py --task science \\
        --sft_checkpoint <path_sft_v1> --rl_checkpoint <path_sft_v2> \\
        --model_a_name sft_v1 --model_b_name sft_v2
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from circuits.discovery import (
    CircuitDiscovery,
    CrossModelCircuitAnalysis,
    DCMAnalysis,
    create_counterfactual_examples_math,
    create_counterfactual_examples_science,
    save_circuit_results
)
from circuits.checkpoint_loader import setup_circuit_analysis_models
from config.CONFIG import MODEL_NAME
from data.load_data import load_dataset_byname


def run_circuit_analysis(base_model, model_a, model_b, tokenizer, dataset, args):
    """Run the full circuit analysis pipeline with error handling."""

    label_a = args.model_a_name
    label_b = args.model_b_name

    print("\n" + "="*70)
    print("STARTING CIRCUIT ANALYSIS")
    print(f"  Model A: {label_a}  |  Model B: {label_b}")
    print("="*70)

    results = {
        'config': {
            'task': args.task,
            'max_examples': args.max_examples,
            'top_k_heads': args.top_k_heads,
            'vulnerability_threshold': args.vulnerability_threshold,
            'model': args.base_model,
            'model_a_name': label_a,
            'model_b_name': label_b,
        },
        'base_circuit': [],
        f'{label_a}_circuit': [],
        f'{label_b}_circuit': [],
        'faithfulness': {},
        'dcm_analysis': {},
        'cmap_analysis': {},
        'vulnerable_circuits': [],
        'binary_analysis': {},
        'errors': []
    }

    # Create counterfactual examples
    print("\nCreating counterfactual examples...")
    try:
        if args.task == 'science':
            counterfactual_examples = create_counterfactual_examples_science(
                dataset, n_examples=args.max_examples
            )
        else:
            counterfactual_examples = create_counterfactual_examples_math(
                dataset, n_examples=args.max_examples
            )

        if len(counterfactual_examples) == 0:
            raise ValueError(f"No counterfactuals created for task '{args.task}'")

        print(f"Created {len(counterfactual_examples)} counterfactual examples")
    except Exception as e:
        print(f"❌ Error creating counterfactuals: {e}")
        results['errors'].append(f"Counterfactual creation: {str(e)}")
        return results

    # Phase 1: Base model circuits
    print("\n" + "="*70)
    print("PHASE 1: IDENTIFYING CIRCUITS IN BASE MODEL")
    print("="*70)

    try:
        base_discovery = CircuitDiscovery(base_model, tokenizer)
        base_circuit = base_discovery.identify_circuit(
            counterfactual_examples, top_k=args.top_k_heads, max_examples=args.max_examples
        )
        results['base_circuit'] = [
            {'layer': s.layer, 'head': s.head, 'importance_score': float(s.score)}
            for s in base_circuit
        ]
    except Exception as e:
        print(f"❌ Error in base model circuit discovery: {e}")
        results['errors'].append(f"Base circuit discovery: {str(e)}")
        base_circuit = []
        base_discovery = None

    # Phase 2: Fine-tuned model circuits
    print("\n" + "="*70)
    print("PHASE 2: IDENTIFYING CIRCUITS IN FINE-TUNED MODELS")
    print("="*70)

    try:
        a_discovery = CircuitDiscovery(model_a, tokenizer)
        a_circuit = a_discovery.identify_circuit(
            counterfactual_examples, top_k=args.top_k_heads, max_examples=args.max_examples
        )
        results[f'{label_a}_circuit'] = [
            {'layer': s.layer, 'head': s.head, 'importance_score': float(s.score)}
            for s in a_circuit
        ]
    except Exception as e:
        print(f"❌ Error in {label_a} model circuit discovery: {e}")
        results['errors'].append(f"{label_a} circuit discovery: {str(e)}")
        a_circuit = []
        a_discovery = None

    try:
        b_discovery = CircuitDiscovery(model_b, tokenizer)
        b_circuit = b_discovery.identify_circuit(
            counterfactual_examples, top_k=args.top_k_heads, max_examples=args.max_examples
        )
        results[f'{label_b}_circuit'] = [
            {'layer': s.layer, 'head': s.head, 'importance_score': float(s.score)}
            for s in b_circuit
        ]
    except Exception as e:
        print(f"❌ Error in {label_b} model circuit discovery: {e}")
        results['errors'].append(f"{label_b} circuit discovery: {str(e)}")
        b_circuit = []
        b_discovery = None

    # Phase 3: Faithfulness Analysis (Equation 4)
    print("\n" + "="*70)
    print("PHASE 3: FAITHFULNESS ANALYSIS (Equation 4)")
    print("="*70)

    faithfulness_examples = min(args.max_examples, 30)

    if base_discovery and base_circuit:
        try:
            base_faithfulness = base_discovery.compute_faithfulness(
                base_circuit, counterfactual_examples,
                top_k=args.top_k_heads, max_examples=faithfulness_examples
            )
            results['faithfulness']['base'] = base_faithfulness
        except Exception as e:
            print(f"⚠️ Base faithfulness failed: {e}")
            results['faithfulness']['base'] = {'faithfulness': 0, 'f_m': 0, 'f_c_m': 0, 'error': str(e)}

    if a_discovery and a_circuit:
        try:
            a_faithfulness = a_discovery.compute_faithfulness(
                a_circuit, counterfactual_examples,
                top_k=args.top_k_heads, max_examples=faithfulness_examples
            )
            results['faithfulness'][label_a] = a_faithfulness
        except Exception as e:
            print(f"⚠️ {label_a} faithfulness failed: {e}")
            results['faithfulness'][label_a] = {'faithfulness': 0, 'f_m': 0, 'f_c_m': 0, 'error': str(e)}

    if b_discovery and b_circuit:
        try:
            b_faithfulness = b_discovery.compute_faithfulness(
                b_circuit, counterfactual_examples,
                top_k=args.top_k_heads, max_examples=faithfulness_examples
            )
            results['faithfulness'][label_b] = b_faithfulness
        except Exception as e:
            print(f"⚠️ {label_b} faithfulness failed: {e}")
            results['faithfulness'][label_b] = {'faithfulness': 0, 'f_m': 0, 'f_c_m': 0, 'error': str(e)}

    # Phase 4: DCM Analysis (Equation 3) - RUNS BY DEFAULT
    print("\n" + "="*70)
    print("PHASE 4: DCM FUNCTIONALITY ANALYSIS (Equation 3)")
    print("="*70)

    if args.skip_dcm:
        print("DCM analysis skipped (--skip_dcm flag set)")
    else:
        dcm_examples = min(args.max_examples, 30)
        dataset_type = args.task if args.task in ('science', 'math') else 'math'

        try:
            print("\nRunning DCM for base model...")
            base_dcm = DCMAnalysis(base_model, tokenizer)
            results['dcm_analysis']['base'] = base_dcm.analyze_all_hypotheses(dataset, n_examples=dcm_examples, dataset_type=dataset_type)
        except Exception as e:
            print(f"⚠️ Base DCM failed: {e}")
            results['dcm_analysis']['base'] = {'error': str(e)}

        try:
            print(f"\nRunning DCM for {label_a} model...")
            a_dcm = DCMAnalysis(model_a, tokenizer)
            results['dcm_analysis'][label_a] = a_dcm.analyze_all_hypotheses(dataset, n_examples=dcm_examples, dataset_type=dataset_type)
        except Exception as e:
            print(f"⚠️ {label_a} DCM failed: {e}")
            results['dcm_analysis'][label_a] = {'error': str(e)}

        try:
            print(f"\nRunning DCM for {label_b} model...")
            b_dcm = DCMAnalysis(model_b, tokenizer)
            results['dcm_analysis'][label_b] = b_dcm.analyze_all_hypotheses(dataset, n_examples=dcm_examples, dataset_type=dataset_type)
        except Exception as e:
            print(f"⚠️ {label_b} DCM failed: {e}")
            results['dcm_analysis'][label_b] = {'error': str(e)}

    # Phase 5: Cross-model comparison (CMAP)
    print("\n" + "="*70)
    print("PHASE 5: CROSS-MODEL CIRCUIT COMPARISON (CMAP)")
    print("="*70)

    cross_analysis = None
    if base_circuit:
        try:
            cross_analysis = CrossModelCircuitAnalysis(base_model, model_a, model_b, tokenizer)
            cmap_results = cross_analysis.cross_model_activation_patching(
                base_circuit, counterfactual_examples, max_examples=args.max_examples
            )
            results['cmap_analysis'] = cmap_results
        except Exception as e:
            print(f"⚠️ CMAP failed: {e}")
            results['cmap_analysis'] = {'error': str(e), 'head_info': [], 'sft_deltas': [], 'rl_deltas': []}
    else:
        print("⚠️ Skipping CMAP - no base circuit available")

    # Phase 6: Vulnerable circuits
    print("\n" + "="*70)
    print("PHASE 6: IDENTIFYING VULNERABLE CIRCUITS")
    print("="*70)

    if cross_analysis and 'head_info' in results['cmap_analysis'] and results['cmap_analysis']['head_info']:
        try:
            vulnerable_circuits = cross_analysis.identify_vulnerable_circuits(
                results['cmap_analysis'], threshold=args.vulnerability_threshold
            )
            results['vulnerable_circuits'] = vulnerable_circuits
        except Exception as e:
            print(f"⚠️ Vulnerable circuit ID failed: {e}")
    else:
        print("⚠️ Skipping vulnerable circuit identification")

    # Binary Circuit Analysis
    print("\n" + "="*70)
    print("BINARY CIRCUIT ANALYSIS")
    print("="*70)

    if base_circuit and a_circuit and b_circuit:
        base_heads_binary = set((s.layer, s.head) for s in base_circuit[:args.top_k_heads])
        a_heads_binary = set((s.layer, s.head) for s in a_circuit[:args.top_k_heads])
        b_heads_binary = set((s.layer, s.head) for s in b_circuit[:args.top_k_heads])

        a_overlap = len(base_heads_binary & a_heads_binary)
        b_overlap = len(base_heads_binary & b_heads_binary)
        a_pct = (a_overlap / len(base_heads_binary)) * 100 if base_heads_binary else 0
        b_pct = (b_overlap / len(base_heads_binary)) * 100 if base_heads_binary else 0

        print(f"\nCircuit Preservation (top-{args.top_k_heads} heads):")
        print(f"  {label_a} preserves: {a_overlap}/{len(base_heads_binary)} ({a_pct:.1f}%)")
        print(f"  {label_b} preserves:  {b_overlap}/{len(base_heads_binary)} ({b_pct:.1f}%)")
        print(f"  {label_b} advantage: +{b_pct - a_pct:.1f} percentage points")

        results['binary_analysis'] = {
            'base_circuit_size': len(base_heads_binary),
            f'{label_a}_overlap_count': a_overlap,
            f'{label_b}_overlap_count': b_overlap,
            f'{label_a}_overlap_pct': a_pct,
            f'{label_b}_overlap_pct': b_pct,
            f'{label_b}_advantage': b_pct - a_pct,
        }

    # Save results
    os.makedirs("results/circuits", exist_ok=True)
    output_path = f"results/circuits/circuit_analysis_{args.task}.json"
    save_circuit_results(results, output_path)

    # Print summary
    print("\n" + "="*70)
    print("CIRCUIT ANALYSIS COMPLETE")
    print("="*70)

    if results['errors']:
        print(f"\n⚠️ Errors: {len(results['errors'])}")
        for err in results['errors']:
            print(f"  - {err}")

    print(f"\nVulnerable circuits: {len(results['vulnerable_circuits'])} heads")

    if results['faithfulness']:
        print(f"\nFaithfulness:")
        for m, metrics in results['faithfulness'].items():
            if isinstance(metrics, dict) and 'faithfulness' in metrics:
                print(f"  {m.upper()}: {metrics['faithfulness']:.4f}")

    print(f"\nResults: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run circuit discovery analysis")
    parser.add_argument("--task", type=str, default="math", choices=["math", "science", "tool"])
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sft_checkpoint", type=str, required=True,
                        help="Path to model A checkpoint (default label: sft)")
    parser.add_argument("--rl_checkpoint", type=str, required=True,
                        help="Path to model B checkpoint (default label: rl)")
    parser.add_argument("--model_a_name", type=str, default="sft",
                        help="Label for model A in output (default: sft)")
    parser.add_argument("--model_b_name", type=str, default="rl",
                        help="Label for model B in output (default: rl)")
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--top_k_heads", type=int, default=20)
    parser.add_argument("--vulnerability_threshold", type=float, default=0.1)
    parser.add_argument("--skip_dcm", action="store_true", help="Skip DCM analysis (faster)")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print(f"\nLoading models...")
    print(f"  Model A ({args.model_a_name}): {args.sft_checkpoint}")
    print(f"  Model B ({args.model_b_name}): {args.rl_checkpoint}")
    try:
        base_model, model_a, model_b, tokenizer = setup_circuit_analysis_models(
            base_model_name=args.base_model,
            results_dir="./results",
            sft_checkpoint=args.sft_checkpoint,
            grpo_checkpoint=args.rl_checkpoint
        )
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        sys.exit(1)

    print(f"\nLoading dataset: {args.task}")
    try:
        dataset = load_dataset_byname(args.task)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)

    results = run_circuit_analysis(base_model, model_a, model_b, tokenizer, dataset, args)
    print("\n✅ Done!")
    print(f"\nVisualize: python visualize_circuits.py results/circuits/circuit_analysis_{args.task}.json")


if __name__ == "__main__":
    main()
