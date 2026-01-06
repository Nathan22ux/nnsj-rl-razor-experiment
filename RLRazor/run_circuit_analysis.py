"""
Main script for running circuit discovery experiments.
Compares which circuits are reinforced by SFT vs RL.

UPDATED VERSION:
- DCM analysis runs by default (use --skip_dcm to disable)
- All errors handled gracefully
- Faithfulness metrics always computed
- Cross-model faithfulness comparison

Usage:
    python run_circuit_analysis.py --task math --sft_checkpoint <path> --rl_checkpoint <path>
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
    save_circuit_results
)
from circuits.checkpoint_loader import setup_circuit_analysis_models
from config.CONFIG import MODEL_NAME
from data.load_data import load_dataset_byname


def run_circuit_analysis(base_model, sft_model, rl_model, tokenizer, dataset, args):
    """Run the full circuit analysis pipeline with error handling."""

    print("\n" + "="*70)
    print("STARTING CIRCUIT ANALYSIS")
    print("="*70)

    results = {
        'config': {
            'task': args.task,
            'max_examples': args.max_examples,
            'top_k_heads': args.top_k_heads,
            'vulnerability_threshold': args.vulnerability_threshold,
            'model': args.base_model
        },
        'base_circuit': [],
        'sft_circuit': [],
        'rl_circuit': [],
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
        counterfactual_examples = create_counterfactual_examples_math(
            dataset, n_examples=args.max_examples
        )

        if len(counterfactual_examples) == 0:
            print("⚠️ WARNING: No counterfactuals created! Using fallback...")
            counterfactual_examples = []
            for i in range(min(args.max_examples, len(dataset))):
                item = dataset[i]
                if isinstance(item, dict) and '0' in item:
                    q = item['0'].get('value', '')
                    try:
                        a = item['1']['ground_truth']['value']
                    except:
                        a = str(item.get('1', ''))
                    counterfactual_examples.append({
                        'question': q,
                        'answer': str(a),
                        'counterfactual_question': q + " (modified)"
                    })
            print(f"Created {len(counterfactual_examples)} fallback examples")
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
        sft_discovery = CircuitDiscovery(sft_model, tokenizer)
        sft_circuit = sft_discovery.identify_circuit(
            counterfactual_examples, top_k=args.top_k_heads, max_examples=args.max_examples
        )
        results['sft_circuit'] = [
            {'layer': s.layer, 'head': s.head, 'importance_score': float(s.score)}
            for s in sft_circuit
        ]
    except Exception as e:
        print(f"❌ Error in SFT model circuit discovery: {e}")
        results['errors'].append(f"SFT circuit discovery: {str(e)}")
        sft_circuit = []
        sft_discovery = None

    try:
        rl_discovery = CircuitDiscovery(rl_model, tokenizer)
        rl_circuit = rl_discovery.identify_circuit(
            counterfactual_examples, top_k=args.top_k_heads, max_examples=args.max_examples
        )
        results['rl_circuit'] = [
            {'layer': s.layer, 'head': s.head, 'importance_score': float(s.score)}
            for s in rl_circuit
        ]
    except Exception as e:
        print(f"❌ Error in RL model circuit discovery: {e}")
        results['errors'].append(f"RL circuit discovery: {str(e)}")
        rl_circuit = []
        rl_discovery = None

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

    if sft_discovery and sft_circuit:
        try:
            sft_faithfulness = sft_discovery.compute_faithfulness(
                sft_circuit, counterfactual_examples,
                top_k=args.top_k_heads, max_examples=faithfulness_examples
            )
            results['faithfulness']['sft'] = sft_faithfulness
        except Exception as e:
            print(f"⚠️ SFT faithfulness failed: {e}")
            results['faithfulness']['sft'] = {'faithfulness': 0, 'f_m': 0, 'f_c_m': 0, 'error': str(e)}

    if rl_discovery and rl_circuit:
        try:
            rl_faithfulness = rl_discovery.compute_faithfulness(
                rl_circuit, counterfactual_examples,
                top_k=args.top_k_heads, max_examples=faithfulness_examples
            )
            results['faithfulness']['rl'] = rl_faithfulness
        except Exception as e:
            print(f"⚠️ RL faithfulness failed: {e}")
            results['faithfulness']['rl'] = {'faithfulness': 0, 'f_m': 0, 'f_c_m': 0, 'error': str(e)}

    # Phase 4: DCM Analysis (Equation 3) - RUNS BY DEFAULT
    print("\n" + "="*70)
    print("PHASE 4: DCM FUNCTIONALITY ANALYSIS (Equation 3)")
    print("="*70)

    if args.skip_dcm:
        print("DCM analysis skipped (--skip_dcm flag set)")
    else:
        dcm_examples = min(args.max_examples, 30)

        try:
            print("\nRunning DCM for base model...")
            base_dcm = DCMAnalysis(base_model, tokenizer)
            results['dcm_analysis']['base'] = base_dcm.analyze_all_hypotheses(dataset, n_examples=dcm_examples)
        except Exception as e:
            print(f"⚠️ Base DCM failed: {e}")
            results['dcm_analysis']['base'] = {'error': str(e)}

        try:
            print("\nRunning DCM for SFT model...")
            sft_dcm = DCMAnalysis(sft_model, tokenizer)
            results['dcm_analysis']['sft'] = sft_dcm.analyze_all_hypotheses(dataset, n_examples=dcm_examples)
        except Exception as e:
            print(f"⚠️ SFT DCM failed: {e}")
            results['dcm_analysis']['sft'] = {'error': str(e)}

        try:
            print("\nRunning DCM for RL model...")
            rl_dcm = DCMAnalysis(rl_model, tokenizer)
            results['dcm_analysis']['rl'] = rl_dcm.analyze_all_hypotheses(dataset, n_examples=dcm_examples)
        except Exception as e:
            print(f"⚠️ RL DCM failed: {e}")
            results['dcm_analysis']['rl'] = {'error': str(e)}

    # Phase 5: Cross-model comparison (CMAP)
    print("\n" + "="*70)
    print("PHASE 5: CROSS-MODEL CIRCUIT COMPARISON (CMAP)")
    print("="*70)

    cross_analysis = None
    if base_circuit:
        try:
            cross_analysis = CrossModelCircuitAnalysis(base_model, sft_model, rl_model, tokenizer)
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

    if base_circuit and sft_circuit and rl_circuit:
        base_heads_binary = set((s.layer, s.head) for s in base_circuit[:args.top_k_heads])
        sft_heads_binary = set((s.layer, s.head) for s in sft_circuit[:args.top_k_heads])
        rl_heads_binary = set((s.layer, s.head) for s in rl_circuit[:args.top_k_heads])

        sft_overlap = len(base_heads_binary & sft_heads_binary)
        rl_overlap = len(base_heads_binary & rl_heads_binary)
        sft_pct = (sft_overlap / len(base_heads_binary)) * 100 if base_heads_binary else 0
        rl_pct = (rl_overlap / len(base_heads_binary)) * 100 if base_heads_binary else 0

        print(f"\nCircuit Preservation (top-{args.top_k_heads} heads):")
        print(f"  SFT preserves: {sft_overlap}/{len(base_heads_binary)} ({sft_pct:.1f}%)")
        print(f"  RL preserves:  {rl_overlap}/{len(base_heads_binary)} ({rl_pct:.1f}%)")
        print(f"  RL advantage: +{rl_pct - sft_pct:.1f} percentage points")

        results['binary_analysis'] = {
            'base_circuit_size': len(base_heads_binary),
            'sft_overlap_count': sft_overlap,
            'rl_overlap_count': rl_overlap,
            'sft_overlap_pct': sft_pct,
            'rl_overlap_pct': rl_pct,
            'rl_advantage': rl_pct - sft_pct
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
    parser.add_argument("--sft_checkpoint", type=str, required=True)
    parser.add_argument("--rl_checkpoint", type=str, required=True)
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--top_k_heads", type=int, default=20)
    parser.add_argument("--vulnerability_threshold", type=float, default=0.1)
    parser.add_argument("--skip_dcm", action="store_true", help="Skip DCM analysis (faster)")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print(f"\nLoading models...")
    try:
        base_model, sft_model, rl_model, tokenizer = setup_circuit_analysis_models(
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

    results = run_circuit_analysis(base_model, sft_model, rl_model, tokenizer, dataset, args)
    print("\n✅ Done!")
    print(f"\nVisualize: python visualize_circuits.py results/circuits/circuit_analysis_{args.task}.json")


if __name__ == "__main__":
    main()