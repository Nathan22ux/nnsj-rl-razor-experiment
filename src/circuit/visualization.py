"""
Visualization tools for circuit analysis results.
Creates plots comparing SFT vs RL circuit preservation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_circuit_results(filepath: str):
    """Load circuit analysis results from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_circuit_overlap(results, save_path=None):
    """
    Plot Venn diagram-style overlap between base, SFT, and RL circuits.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract head sets
    base_heads = set((h['layer'], h['head']) for h in results['base_circuit'])
    sft_heads = set((h['layer'], h['head']) for h in results['sft_circuit'])
    rl_heads = set((h['layer'], h['head']) for h in results['rl_circuit'])

    # Compute overlaps
    sft_only = len(sft_heads - base_heads - rl_heads)
    rl_only = len(rl_heads - base_heads - sft_heads)
    base_only = len(base_heads - sft_heads - rl_heads)

    sft_rl = len((sft_heads & rl_heads) - base_heads)
    sft_base = len((sft_heads & base_heads) - rl_heads)
    rl_base = len((rl_heads & base_heads) - sft_heads)

    all_three = len(base_heads & sft_heads & rl_heads)

    # Create bar plot
    categories = ['Base Only', 'SFT Only', 'RL Only', 'SFT∩Base', 'RL∩Base', 'SFT∩RL', 'All Three']
    counts = [base_only, sft_only, rl_only, sft_base, rl_base, sft_rl, all_three]
    colors = ['gray', 'orange', 'blue', 'yellow', 'purple', 'green', 'red']

    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')

    ax.set_ylabel('Number of Attention Heads', fontsize=12)
    ax.set_title('Circuit Overlap: Base vs SFT vs RL', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.2)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved circuit overlap plot to {save_path}")

    plt.show()


def plot_cmap_comparison(results, save_path=None):
    """
    Plot CMAP results: how SFT vs RL activations affect base model.
    Shows ΔF for each head.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    cmap_data = results['cmap_analysis']
    head_labels = [f"L{h['layer']}H{h['head']}" for h in cmap_data['head_info']]
    sft_deltas = cmap_data['sft_deltas']
    rl_deltas = cmap_data['rl_deltas']

    # Plot 1: SFT vs RL deltas
    x = np.arange(len(head_labels))
    width = 0.35

    bars1 = ax1.bar(x - width/2, sft_deltas, width, label='SFT', color='orange', alpha=0.7)
    bars2 = ax1.bar(x + width/2, rl_deltas, width, label='RL', color='blue', alpha=0.7)

    ax1.set_xlabel('Attention Head', fontsize=12)
    ax1.set_ylabel('ΔF (Change in Performance)', fontsize=12)
    ax1.set_title('Cross-Model Activation Patching Results', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(head_labels, rotation=90, ha='right')
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Differential effect (RL - SFT)
    differential = np.array(rl_deltas) - np.array(sft_deltas)
    colors = ['red' if d < 0 else 'green' for d in differential]

    bars3 = ax2.bar(x, differential, color=colors, alpha=0.7)
    ax2.set_xlabel('Attention Head', fontsize=12)
    ax2.set_ylabel('RL ΔF - SFT ΔF', fontsize=12)
    ax2.set_title('Differential Circuit Preservation\n(Positive = RL preserves better)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(head_labels, rotation=90, ha='right')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved CMAP comparison plot to {save_path}")

    plt.show()


def plot_vulnerable_circuits(results, save_path=None):
    """
    Plot the most vulnerable circuits (those degraded more by SFT than RL).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    vulnerable = results['vulnerable_circuits']

    if not vulnerable:
        print("No vulnerable circuits found!")
        return

    # Take top 15 most vulnerable
    top_vulnerable = vulnerable[:15]

    head_labels = [f"L{h['layer']}H{h['head']}" for h in top_vulnerable]
    vulnerabilities = [h['vulnerability'] for h in top_vulnerable]
    sft_deltas = [h['sft_delta'] for h in top_vulnerable]
    rl_deltas = [h['rl_delta'] for h in top_vulnerable]

    x = np.arange(len(head_labels))
    width = 0.25

    bars1 = ax.bar(x - width, sft_deltas, width, label='SFT ΔF', color='orange', alpha=0.7)
    bars2 = ax.bar(x, rl_deltas, width, label='RL ΔF', color='blue', alpha=0.7)
    bars3 = ax.bar(x + width, vulnerabilities, width, label='Vulnerability',
                   color='red', alpha=0.7)

    ax.set_xlabel('Attention Head', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Most Vulnerable Circuits\n(Heads where SFT degrades more than RL)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(head_labels, rotation=90, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved vulnerable circuits plot to {save_path}")

    plt.show()


def plot_circuit_heatmap(results, model_type='base', save_path=None):
    """
    Plot heatmap of circuit importance across layers and heads.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get circuit for specified model
    circuit = results[f'{model_type}_circuit']

    # Find max layer and head
    max_layer = max(h['layer'] for h in circuit)
    max_head = max(h['head'] for h in circuit)

    # Create matrix
    importance_matrix = np.zeros((max_layer + 1, max_head + 1))

    for head in circuit:
        importance_matrix[head['layer'], head['head']] = abs(head['importance_score'])

    # Plot heatmap
    sns.heatmap(importance_matrix, cmap='YlOrRd', annot=False,
                cbar_kws={'label': 'Importance Score (|score|)'}, ax=ax)

    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    ax.set_title(f'{model_type.upper()} Model: Circuit Importance Heatmap',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved circuit heatmap to {save_path}")

    plt.show()


def generate_all_visualizations(results_path: str, output_dir: str = "results/circuits/plots"):
    """
    Generate all visualizations for circuit analysis results.
    """
    # Load results
    print(f"Loading results from {results_path}...")
    results = load_circuit_results(results_path)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    task = results['config']['task']

    print("\nGenerating visualizations...")

    # 1. Circuit overlap
    print("  1. Circuit overlap plot...")
    plot_circuit_overlap(
        results,
        save_path=f"{output_dir}/circuit_overlap_{task}.png"
    )

    # 2. CMAP comparison
    print("  2. CMAP comparison plot...")
    plot_cmap_comparison(
        results,
        save_path=f"{output_dir}/cmap_comparison_{task}.png"
    )

    # 3. Vulnerable circuits
    print("  3. Vulnerable circuits plot...")
    plot_vulnerable_circuits(
        results,
        save_path=f"{output_dir}/vulnerable_circuits_{task}.png"
    )

    # 4. Circuit heatmaps
    print("  4. Circuit heatmaps...")
    for model_type in ['base', 'sft', 'rl']:
        plot_circuit_heatmap(
            results,
            model_type=model_type,
            save_path=f"{output_dir}/circuit_heatmap_{model_type}_{task}.png"
        )

    print(f"\nAll visualizations saved to {output_dir}")


def print_circuit_summary(results_path: str):
    """
    Print a text summary of the circuit analysis results.
    """
    results = load_circuit_results(results_path)

    print("\n" + "="*70)
    print("CIRCUIT ANALYSIS SUMMARY")
    print("="*70)

    config = results['config']
    print(f"\nConfiguration:")
    print(f"  Task: {config['task']}")
    print(f"  Max examples: {config['max_examples']}")
    print(f"  Top-k heads: {config['top_k_heads']}")
    print(f"  Vulnerability threshold: {config['vulnerability_threshold']}")

    # Circuit sizes
    print(f"\nCircuit Sizes:")
    print(f"  Base model: {len(results['base_circuit'])} heads")
    print(f"  SFT model: {len(results['sft_circuit'])} heads")
    print(f"  RL model: {len(results['rl_circuit'])} heads")

    # Overlaps
    base_heads = set((h['layer'], h['head']) for h in results['base_circuit'])
    sft_heads = set((h['layer'], h['head']) for h in results['sft_circuit'])
    rl_heads = set((h['layer'], h['head']) for h in results['rl_circuit'])

    sft_overlap = len(base_heads & sft_heads)
    rl_overlap = len(base_heads & rl_heads)

    print(f"\nCircuit Overlap with Base:")
    print(f"  SFT: {sft_overlap}/{len(base_heads)} ({100*sft_overlap/len(base_heads):.1f}%)")
    print(f"  RL: {rl_overlap}/{len(base_heads)} ({100*rl_overlap/len(base_heads):.1f}%)")

    # Vulnerable circuits
    vulnerable = results['vulnerable_circuits']
    print(f"\nVulnerable Circuits: {len(vulnerable)} heads")

    if vulnerable:
        print("\nTop 5 Most Vulnerable Heads:")
        for i, head in enumerate(vulnerable[:5], 1):
            print(f"  {i}. Layer {head['layer']}, Head {head['head']}")
            print(f"     SFT ΔF: {head['sft_delta']:.4f}")
            print(f"     RL ΔF: {head['rl_delta']:.4f}")
            print(f"     Vulnerability: {head['vulnerability']:.4f}")

    # Key finding
    if rl_overlap > sft_overlap:
        print("\n" + "="*70)
        print("KEY FINDING: RL preserves base model circuits better than SFT")
        print(f"  RL maintains {rl_overlap - sft_overlap} more base circuit heads than SFT")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("KEY FINDING: SFT and RL show similar circuit preservation")
        print("="*70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python circuit_visualization.py <results_json_path>")
        sys.exit(1)

    results_path = sys.argv[1]

    # Print summary
    print_circuit_summary(results_path)

    # Generate all visualizations
    generate_all_visualizations(results_path)