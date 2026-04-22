"""
Main script for running circuit discovery experiments.
Compares which circuits are reinforced by two fine-tuned models vs base.

All circuit analysis uses Differential Binary Masking (DBM) exclusively —
no path patching or ablation-based counterfactual intervention.

DBM reference: Chaudhary & Geiger (2024), arxiv 2409.04478
    interpolated = (1 - sigma(m/T)) * f_base + sigma(m/T) * f_source
    L = CE(model(interpolated), y)
    T annealed 10 → 0.1 over 20 epochs (pushes masks to binary)

Pipeline:
  Phase 1  — DBM circuit discovery: base model
  Phase 2  — DBM circuit discovery: SFT and RL models
  Phase 3  — Mask-based faithfulness (sufficiency of circuit)
  Phase 4  — Mask-based necessity & sufficiency per head
  Phase 5  — Cross-model mask comparison
  Phase 6  — DCM hypothesis analysis (answer_key, molecule, task_type)
  Phase 7  — Binary circuit overlap summary

Usage:
    python run_circuit_analysis.py --task science \\
        --sft_checkpoint <path> --rl_checkpoint <path>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from circuits.discovery import (
    DCMAnalysis,
    create_counterfactual_examples_math,
    create_counterfactual_examples_science,
    save_circuit_results,
)
from circuits.checkpoint_loader import setup_circuit_analysis_models
from config.CONFIG import MODEL_NAME
from data.load_data import load_dataset_byname


def run_circuit_analysis(base_model, model_a, model_b, tokenizer, dataset, args):
    """Run the full mask-based circuit analysis pipeline."""

    label_a = args.model_a_name
    label_b = args.model_b_name

    print("\n" + "="*70)
    print("STARTING CIRCUIT ANALYSIS  (DBM — no path patching)")
    print(f"  Model A: {label_a}  |  Model B: {label_b}")
    print("="*70)

    results = {
        'config': {
            'task': args.task,
            'max_examples': args.max_examples,
            'circuit_method': 'dbm',
            'lambda_sparsity': args.lambda_sparsity,
            'vulnerability_threshold': args.vulnerability_threshold,
            'model': args.base_model,
            'model_a_name': label_a,
            'model_b_name': label_b,
        },
        'base_circuit': [],
        f'{label_a}_circuit': [],
        f'{label_b}_circuit': [],
        'faithfulness': {},
        'necessity_sufficiency': {},
        'cross_model_comparison': {},
        'dcm_analysis': {},
        'binary_analysis': {},
        'errors': [],
    }

    # ------------------------------------------------------------------ #
    # Counterfactual examples
    # ------------------------------------------------------------------ #
    print("\nCreating counterfactual examples...")
    try:
        if args.task == 'science':
            examples = create_counterfactual_examples_science(
                dataset, n_examples=args.max_examples
            )
        else:
            examples = create_counterfactual_examples_math(
                dataset, n_examples=args.max_examples
            )
        if not examples:
            raise ValueError(f"No counterfactuals created for task '{args.task}'")
        print(f"Created {len(examples)} counterfactual examples")
    except Exception as e:
        print(f"❌ Error creating counterfactuals: {e}")
        results['errors'].append(f"Counterfactual creation: {str(e)}")
        return results

    # ------------------------------------------------------------------ #
    # Phase 1: Base model — DBM circuit discovery
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("PHASE 1: BASE MODEL CIRCUIT (DBM mask training)")
    print("="*70)

    base_dcm = DCMAnalysis(base_model, tokenizer)
    base_circuit = []
    try:
        base_circuit = base_dcm.train_circuit_mask(
            examples, lambda_sparsity=args.lambda_sparsity
        )
        results['base_circuit'] = [
            {'layer': s.layer, 'head': s.head, 'mask_value': float(s.score)}
            for s in base_circuit
        ]
        print(f"Base circuit: {len(base_circuit)} heads (mask > 0.5)")
    except Exception as e:
        print(f"❌ Base circuit discovery failed: {e}")
        results['errors'].append(f"Base circuit: {str(e)}")

    # ------------------------------------------------------------------ #
    # Phase 2: Fine-tuned models — DBM circuit discovery
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("PHASE 2: FINE-TUNED MODEL CIRCUITS (DBM mask training)")
    print("="*70)

    a_dcm = DCMAnalysis(model_a, tokenizer)
    a_circuit = []
    try:
        a_circuit = a_dcm.train_circuit_mask(
            examples, lambda_sparsity=args.lambda_sparsity
        )
        results[f'{label_a}_circuit'] = [
            {'layer': s.layer, 'head': s.head, 'mask_value': float(s.score)}
            for s in a_circuit
        ]
        print(f"{label_a} circuit: {len(a_circuit)} heads")
    except Exception as e:
        print(f"❌ {label_a} circuit discovery failed: {e}")
        results['errors'].append(f"{label_a} circuit: {str(e)}")

    b_dcm = DCMAnalysis(model_b, tokenizer)
    b_circuit = []
    try:
        b_circuit = b_dcm.train_circuit_mask(
            examples, lambda_sparsity=args.lambda_sparsity
        )
        results[f'{label_b}_circuit'] = [
            {'layer': s.layer, 'head': s.head, 'mask_value': float(s.score)}
            for s in b_circuit
        ]
        print(f"{label_b} circuit: {len(b_circuit)} heads")
    except Exception as e:
        print(f"❌ {label_b} circuit discovery failed: {e}")
        results['errors'].append(f"{label_b} circuit: {str(e)}")

    eval_examples = examples[:min(args.max_examples, 30)]

    # ------------------------------------------------------------------ #
    # Phase 3: Mask-based faithfulness (sufficiency)
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("PHASE 3: FAITHFULNESS (mask-based — no ablation)")
    print("="*70)

    for label, dcm, circuit in [
        ('base',  base_dcm, base_circuit),
        (label_a, a_dcm,    a_circuit),
        (label_b, b_dcm,    b_circuit),
    ]:
        if not circuit:
            print(f"⚠️ Skipping {label} faithfulness — no circuit")
            continue
        try:
            faith = dcm.compute_faithfulness_dbm(circuit, eval_examples)
            results['faithfulness'][label] = faith
        except Exception as e:
            print(f"⚠️ {label} faithfulness failed: {e}")
            results['faithfulness'][label] = {'error': str(e)}

    # ------------------------------------------------------------------ #
    # Phase 4: Mask-based necessity & sufficiency per head
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("PHASE 4: NECESSITY & SUFFICIENCY (mask-based — no ablation)")
    print("="*70)

    for label, dcm, circuit in [
        ('base',  base_dcm, base_circuit),
        (label_a, a_dcm,    a_circuit),
        (label_b, b_dcm,    b_circuit),
    ]:
        if not circuit:
            print(f"⚠️ Skipping {label} N&S — no circuit")
            continue
        try:
            ns = dcm.compute_necessity_sufficiency_dbm(circuit, eval_examples)
            results['necessity_sufficiency'][label] = ns
        except Exception as e:
            print(f"⚠️ {label} N&S failed: {e}")
            results['necessity_sufficiency'][label] = {'error': str(e)}

    # ------------------------------------------------------------------ #
    # Phase 5: Cross-model mask comparison (replaces CMAP / path patching)
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("PHASE 5: CROSS-MODEL MASK COMPARISON")
    print("="*70)

    if base_circuit:
        try:
            circuits_dict = {
                'base':  base_circuit,
                label_a: a_circuit,
                label_b: b_circuit,
            }
            cmp = base_dcm.compare_circuits_dbm(circuits_dict)
            results['cross_model_comparison'] = cmp

            print(f"\nHead-by-head mask values (base circuit heads):")
            print(f"  {'Head':<12} {'base':>8} {label_a:>8} {label_b:>8} {label_a+' Δ':>8} {label_b+' Δ':>8}")
            for row in cmp.get('head_comparison', []):
                print(
                    f"  L{row['layer']}H{row['head']:<8}"
                    f"  {row['base_mask']:>7.3f}"
                    f"  {row.get(f'{label_a}_mask', 0):>7.3f}"
                    f"  {row.get(f'{label_b}_mask', 0):>7.3f}"
                    f"  {row.get(f'{label_a}_delta', 0):>+7.3f}"
                    f"  {row.get(f'{label_b}_delta', 0):>+7.3f}"
                )
        except Exception as e:
            print(f"⚠️ Cross-model comparison failed: {e}")
            results['cross_model_comparison'] = {'error': str(e)}
    else:
        print("⚠️ Skipping — no base circuit")

    # ------------------------------------------------------------------ #
    # Phase 6: DCM hypothesis analysis
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("PHASE 6: DCM HYPOTHESIS ANALYSIS")
    print("="*70)

    if args.skip_dcm:
        print("Skipped (--skip_dcm)")
    else:
        dcm_n = min(args.max_examples, 30)
        dataset_type = args.task if args.task in ('science', 'math') else 'math'
        for label, dcm in [('base', base_dcm), (label_a, a_dcm), (label_b, b_dcm)]:
            try:
                print(f"\nDCM hypotheses for {label}...")
                results['dcm_analysis'][label] = dcm.analyze_all_hypotheses(
                    dataset, n_examples=dcm_n, dataset_type=dataset_type
                )
            except Exception as e:
                print(f"⚠️ {label} DCM hypothesis analysis failed: {e}")
                results['dcm_analysis'][label] = {'error': str(e)}

    # ------------------------------------------------------------------ #
    # Phase 7: Binary circuit overlap
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("PHASE 7: BINARY CIRCUIT OVERLAP")
    print("="*70)

    if base_circuit and a_circuit and b_circuit:
        base_heads = {(s.layer, s.head) for s in base_circuit}
        a_heads    = {(s.layer, s.head) for s in a_circuit}
        b_heads    = {(s.layer, s.head) for s in b_circuit}

        a_overlap = len(base_heads & a_heads)
        b_overlap = len(base_heads & b_heads)
        a_pct = a_overlap / len(base_heads) * 100 if base_heads else 0
        b_pct = b_overlap / len(base_heads) * 100 if base_heads else 0

        print(f"\n  Base: {len(base_heads)} heads  |  {label_a}: {len(a_heads)}  |  {label_b}: {len(b_heads)}")
        print(f"  {label_a} preserves {a_overlap}/{len(base_heads)} base heads ({a_pct:.1f}%)")
        print(f"  {label_b} preserves {b_overlap}/{len(base_heads)} base heads ({b_pct:.1f}%)")
        print(f"  {label_b} advantage: +{b_pct - a_pct:.1f} pp")

        results['binary_analysis'] = {
            'base_circuit_size': len(base_heads),
            f'{label_a}_circuit_size': len(a_heads),
            f'{label_b}_circuit_size': len(b_heads),
            f'{label_a}_overlap_count': a_overlap,
            f'{label_b}_overlap_count': b_overlap,
            f'{label_a}_overlap_pct': a_pct,
            f'{label_b}_overlap_pct': b_pct,
            f'{label_b}_advantage': b_pct - a_pct,
        }

    # Save
    os.makedirs("results/circuits", exist_ok=True)
    output_path = f"results/circuits/circuit_analysis_{args.task}.json"
    save_circuit_results(results, output_path)

    # Summary
    print("\n" + "="*70)
    print("CIRCUIT ANALYSIS COMPLETE")
    print("="*70)

    if results['errors']:
        print(f"\n⚠️ Errors ({len(results['errors'])}):")
        for e in results['errors']:
            print(f"  - {e}")

    if results['faithfulness']:
        print(f"\nFaithfulness:")
        for m, v in results['faithfulness'].items():
            if isinstance(v, dict) and 'faithfulness' in v:
                print(f"  {m}: {v['faithfulness']:.4f}  (circuit size: {v.get('circuit_size','?')})")

    if results['necessity_sufficiency']:
        print(f"\nNecessity & Sufficiency (circuit-level):")
        for m, ns in results['necessity_sufficiency'].items():
            if isinstance(ns, dict) and 'circuit_necessity' in ns:
                print(f"  {m}: necessity={ns['circuit_necessity']:.4f}  sufficiency={ns['circuit_sufficiency']:.4f}")
                top3 = sorted(ns['per_head'].values(), key=lambda x: x['necessity'], reverse=True)[:3]
                for h in top3:
                    print(f"    L{h['layer']}H{h['head']}: necessity={h['necessity']:.4f}  sufficiency_lp={h['sufficiency_logprob']:.4f}")

    print(f"\nResults saved: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Mask-based circuit analysis (DBM)")
    parser.add_argument("--task", type=str, default="math", choices=["math", "science", "tool"])
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sft_checkpoint", type=str, required=True)
    parser.add_argument("--rl_checkpoint",  type=str, required=True)
    parser.add_argument("--model_a_name", type=str, default="sft")
    parser.add_argument("--model_b_name", type=str, default="rl")
    parser.add_argument("--max_examples", type=int, default=200)
    parser.add_argument("--lambda_sparsity", type=float, default=0.1,
                        help="Sparsity weight for DBM mask training (default: 0.1)")
    parser.add_argument("--vulnerability_threshold", type=float, default=0.1)
    parser.add_argument("--skip_dcm", action="store_true",
                        help="Skip per-hypothesis DCM analysis")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print(f"\nLoading models...")
    print(f"  Base:          {args.base_model}")
    print(f"  {args.model_a_name}: {args.sft_checkpoint}")
    print(f"  {args.model_b_name}:  {args.rl_checkpoint}")
    try:
        base_model, model_a, model_b, tokenizer = setup_circuit_analysis_models(
            base_model_name=args.base_model,
            results_dir="./results",
            sft_checkpoint=args.sft_checkpoint,
            grpo_checkpoint=args.rl_checkpoint,
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

    run_circuit_analysis(base_model, model_a, model_b, tokenizer, dataset, args)
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
