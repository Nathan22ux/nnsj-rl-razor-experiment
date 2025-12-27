
# USAGE EXAMPLE:
# --------------
# In generate_all_visualizations(), replace:
#   plot_cmap_comparison(results, save_path=...)
# 
# With:
#   plot_cmap_comparison_binary(results, threshold=0.01, save_path=...)
#
# Or call both to get both continuous and binary plots:
#   plot_cmap_comparison(results, save_path=f"{output_dir}/cmap_continuous_{task}.png")
#   plot_cmap_comparison_binary(results, threshold=0.01, save_path=f"{output_dir}/cmap_binary_{task}.png")

"""
Visualization tools for circuit analysis results.
Creates plots comparing SFT vs RL circuit preservation.

UPDATED: Now handles faithfulness metrics (Equation 4) and DCM analysis (Equation 3)
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
    ax.set_ylim(0, max(counts) * 1.2 if counts else 1)

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
    cmap_data = results.get('cmap_analysis', {})

    if not cmap_data or not cmap_data.get('head_info'):
        print("No CMAP data available to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

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

# Binary CMAP Visualization Code
def plot_cmap_comparison_binary(results, threshold=0.01, save_path=None):
    """
    Plot CMAP results with BINARY activation states.
    Classifies circuits as active (1) or inactive (0) based on threshold.
    
    Args:
        results: Circuit analysis results dictionary
        threshold: Threshold for classifying as active (default: 0.01)
                  If |ΔF| > threshold, circuit is active (1), else inactive (0)
        save_path: Path to save plot
    
    Usage:
        # Instead of plot_cmap_comparison(results)
        # Use:
        plot_cmap_comparison_binary(results, threshold=0.01)
    """
    cmap_data = results.get('cmap_analysis', {})
    
    if not cmap_data or not cmap_data.get('head_info'):
        print("No CMAP data available to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    head_labels = [f"L{h['layer']}H{h['head']}" for h in cmap_data['head_info']]
    sft_deltas = np.array(cmap_data['sft_deltas'])
    rl_deltas = np.array(cmap_data['rl_deltas'])
    
    # BINARY CLASSIFICATION: active (1) if |ΔF| > threshold, else inactive (0)
    sft_binary = (np.abs(sft_deltas) > threshold).astype(int)
    rl_binary = (np.abs(rl_deltas) > threshold).astype(int)
    
    # Plot 1: Binary activation states
    x = np.arange(len(head_labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, sft_binary, width, label='SFT', 
                    color='orange', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, rl_binary, width, label='RL', 
                    color='blue', alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel('Attention Head', fontsize=12)
    ax1.set_ylabel('Circuit State (0=Inactive, 1=Active)', fontsize=12)
    ax1.set_title(f'Binary Circuit Activation (threshold={threshold})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(head_labels, rotation=90, ha='right')
    ax1.set_ylim(-0.2, 1.5)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Inactive (0)', 'Active (1)'])
    ax1.legend()
    ax1.axhline(y=0.5, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Binary differential (1 if RL active and SFT inactive, 
    #                              -1 if SFT active and RL inactive,
    #                              0 if both same state)
    binary_differential = rl_binary - sft_binary
    colors = ['red' if d < 0 else ('green' if d > 0 else 'gray') 
              for d in binary_differential]
    
    bars3 = ax2.bar(x, binary_differential, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Attention Head', fontsize=12)
    ax2.set_ylabel('RL Active - SFT Active', fontsize=12)
    ax2.set_title('Differential Circuit Preservation (Binary)\n'
                  '(Green=RL preserves, Red=SFT preserves, Gray=Same)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(head_labels, rotation=90, ha='right')
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['SFT only (-1)', 'Same (0)', 'RL only (+1)'])
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    n_sft_active = sft_binary.sum()
    n_rl_active = rl_binary.sum()
    n_both_active = (sft_binary & rl_binary).sum()
    n_rl_only = ((rl_binary == 1) & (sft_binary == 0)).sum()
    n_sft_only = ((sft_binary == 1) & (rl_binary == 0)).sum()
    
    stats_text = (
        f"Active circuits:\n"
        f"  SFT: {n_sft_active}/{len(sft_binary)}\n"
        f"  RL: {n_rl_active}/{len(rl_binary)}\n"
        f"  Both: {n_both_active}\n"
        f"  RL only: {n_rl_only}\n"
        f"  SFT only: {n_sft_only}"
    )
    
    fig.text(0.98, 0.02, stats_text, fontsize=10, 
             ha='right', va='bottom', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved binary CMAP comparison plot to {save_path}")
    
    plt.show()
    
    # Print binary statistics
    print("\n" + "="*70)
    print("BINARY CIRCUIT ACTIVATION ANALYSIS")
    print("="*70)
    print(f"Threshold: |ΔF| > {threshold}")
    print(f"\nActive circuits:")
    print(f"  SFT:  {n_sft_active}/{len(sft_binary)} ({100*n_sft_active/len(sft_binary):.1f}%)")
    print(f"  RL:   {n_rl_active}/{len(rl_binary)} ({100*n_rl_active/len(rl_binary):.1f}%)")
    print(f"\nOverlap:")
    print(f"  Both active:  {n_both_active}")
    print(f"  RL only:      {n_rl_only}")
    print(f"  SFT only:     {n_sft_only}")
    print(f"  Neither:      {((sft_binary == 0) & (rl_binary == 0)).sum()}")
    
    if n_rl_active > n_sft_active:
        print(f"\n✓ RL preserves {n_rl_active - n_sft_active} more circuits than SFT")
    elif n_sft_active > n_rl_active:
        print(f"\n✗ SFT preserves {n_sft_active - n_rl_active} more circuits than RL")
    else:
        print(f"\n= RL and SFT preserve same number of circuits")
    print("="*70)



def plot_vulnerable_circuits(results, save_path=None):
    """
    Plot the most vulnerable circuits (those degraded more by SFT than RL).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    vulnerable = results.get('vulnerable_circuits', [])

    if not vulnerable:
        print("No vulnerable circuits found!")
        ax.text(0.5, 0.5, 'No vulnerable circuits identified',
                ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
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
    circuit = results.get(f'{model_type}_circuit', [])

    if not circuit:
        print(f"No circuit data for {model_type}")
        return

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


def plot_faithfulness_comparison(results, save_path=None):
    """
    Plot faithfulness metrics comparison (Equation 4).

    NEW: Added to visualize faithfulness across base, SFT, and RL models.
    """
    faithfulness_data = results.get('faithfulness', {})

    if not faithfulness_data:
        print("No faithfulness data available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    models = list(faithfulness_data.keys())

    # Plot 1: Faithfulness scores
    faithfulness_scores = [faithfulness_data[m].get('faithfulness', 0) for m in models]
    f_m_scores = [faithfulness_data[m].get('f_m', 0) for m in models]
    f_c_m_scores = [faithfulness_data[m].get('f_c_m', 0) for m in models]

    x = np.arange(len(models))
    width = 0.25

    bars1 = ax1.bar(x - width, f_m_scores, width, label='F(M) - Full Model', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x, f_c_m_scores, width, label='F(C|M) - Circuit Only', color='coral', alpha=0.8)
    bars3 = ax1.bar(x + width, faithfulness_scores, width, label='Faithfulness', color='green', alpha=0.8)

    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Faithfulness Metrics (Equation 4)\nF(C|M) / F(M)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in models])
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1.1)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Circuit efficiency (how much of model is needed)
    circuit_fractions = [faithfulness_data[m].get('circuit_fraction', 0) * 100 for m in models]

    colors = ['gray', 'orange', 'blue']
    bars = ax2.bar(models, circuit_fractions, color=colors[:len(models)], alpha=0.7, edgecolor='black')

    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Circuit Size (% of total heads)', fontsize=12)
    ax2.set_title('Circuit Efficiency\n(Smaller = More Concentrated)', fontsize=14, fontweight='bold')
    ax2.set_xticklabels([m.upper() for m in models])
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved faithfulness comparison plot to {save_path}")

    plt.show()


def plot_dcm_analysis(results, save_path=None):
    """
    Plot DCM analysis results (Equation 3).

    NEW: Added to visualize which heads encode different functionalities.
    """
    dcm_data = results.get('dcm_analysis', {})

    if not dcm_data:
        print("No DCM data available (run with --run_dcm flag)")
        return

    # Get all hypotheses tested
    models = list(dcm_data.keys())
    if not models:
        print("No DCM results found")
        return

    # Check what hypotheses were tested
    first_model = models[0]
    hypotheses = list(dcm_data[first_model].keys()) if dcm_data[first_model] else []

    if not hypotheses:
        print("No hypotheses in DCM results")
        return

    n_hypotheses = len(hypotheses)
    n_models = len(models)

    fig, axes = plt.subplots(1, n_hypotheses, figsize=(5*n_hypotheses, 5))
    if n_hypotheses == 1:
        axes = [axes]

    for idx, hypothesis in enumerate(hypotheses):
        ax = axes[idx]

        # Count active heads per model for this hypothesis
        active_counts = []
        for model in models:
            if model in dcm_data and hypothesis in dcm_data[model]:
                hyp_result = dcm_data[model][hypothesis]
                if isinstance(hyp_result, dict):
                    active_heads = hyp_result.get('active_heads', [])
                    active_counts.append(len(active_heads))
                else:
                    active_counts.append(0)
            else:
                active_counts.append(0)

        colors = ['gray', 'orange', 'blue'][:n_models]
        bars = ax.bar(models, active_counts, color=colors, alpha=0.7, edgecolor='black')

        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('Active Heads', fontsize=11)
        ax.set_title(f'DCM: {hypothesis.capitalize()}\nFunctionality', fontsize=12, fontweight='bold')
        ax.set_xticklabels([m.upper() for m in models])
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('DCM Analysis: Heads Encoding Different Functionalities (Equation 3)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved DCM analysis plot to {save_path}")

    plt.show()


def plot_binary_analysis(results, save_path=None):
    """
    Plot binary circuit analysis summary.
    """
    binary_data = results.get('binary_analysis', {})

    if not binary_data:
        print("No binary analysis data available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Overlap percentages
    categories = ['SFT', 'RL']
    overlaps = [binary_data.get('sft_overlap_pct', 0), binary_data.get('rl_overlap_pct', 0)]
    colors = ['orange', 'blue']

    bars = ax1.bar(categories, overlaps, color=colors, alpha=0.7, edgecolor='black')

    ax1.set_ylabel('Overlap with Base Circuit (%)', fontsize=12)
    ax1.set_title('Circuit Preservation\n(% of base circuit heads maintained)',
                  fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add RL advantage annotation
    rl_advantage = binary_data.get('rl_advantage', 0)
    if rl_advantage > 0:
        ax1.annotate(f'RL advantage: +{rl_advantage:.1f}%',
                     xy=(1, overlaps[1]), xytext=(1.3, overlaps[1] - 10),
                     fontsize=11, color='green', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='green'))

    # Plot 2: Overlap counts
    base_size = binary_data.get('base_circuit_size', 0)
    sft_overlap = binary_data.get('sft_overlap_count', 0)
    rl_overlap = binary_data.get('rl_overlap_count', 0)

    x = np.arange(2)
    width = 0.35

    ax2.bar(x - width/2, [base_size, base_size], width, label='Base Circuit Size',
            color='gray', alpha=0.5)
    ax2.bar(x + width/2, [sft_overlap, rl_overlap], width, label='Overlap Count',
            color=['orange', 'blue'], alpha=0.7)

    ax2.set_ylabel('Number of Heads', fontsize=12)
    ax2.set_title('Circuit Head Counts', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['SFT', 'RL'])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved binary analysis plot to {save_path}")

    plt.show()


def generate_all_visualizations(results_path: str, output_dir: str = "results/circuits/plots"):
    """
    Generate all visualizations for circuit analysis results.

    UPDATED: Now includes faithfulness and DCM plots.
    """
    # Load results
    print(f"Loading results from {results_path}...")
    results = load_circuit_results(results_path)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    task = results.get('config', {}).get('task', 'unknown')

    print("\nGenerating visualizations...")

    # 1. Circuit overlap
    print("  1. Circuit overlap plot...")
    plot_circuit_overlap(
        results,
        save_path=f"{output_dir}/circuit_overlap_{task}.png"
    )

    # 2. CMAP comparison
    print("  2a. CMAP comparison plot...")
    plot_cmap_comparison(
        results,
        save_path=f"{output_dir}/cmap_comparison_{task}.png"
    )
    
    # 2b. CMAP comparison (binary) - ADD THIS
    print("  2b. CMAP comparison plot (binary)...")
    plot_cmap_comparison_binary(
        results,
        threshold=0.01,  # Adjust based on your data
        save_path=f"{output_dir}/cmap_binary_{task}.png"
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

    # 5. Faithfulness comparison (NEW)
    print("  5. Faithfulness comparison plot...")
    plot_faithfulness_comparison(
        results,
        save_path=f"{output_dir}/faithfulness_comparison_{task}.png"
    )

    # 6. DCM analysis (NEW)
    print("  6. DCM analysis plot...")
    plot_dcm_analysis(
        results,
        save_path=f"{output_dir}/dcm_analysis_{task}.png"
    )

    # 7. Binary analysis summary (NEW)
    print("  7. Binary analysis plot...")
    plot_binary_analysis(
        results,
        save_path=f"{output_dir}/binary_analysis_{task}.png"
    )

    print(f"\nAll visualizations saved to {output_dir}")


def print_circuit_summary(results_path: str):
    """
    Print a text summary of the circuit analysis results.

    UPDATED: Now includes faithfulness and DCM summaries.
    """
    results = load_circuit_results(results_path)

    print("\n" + "="*70)
    print("CIRCUIT ANALYSIS SUMMARY")
    print("="*70)

    config = results.get('config', {})
    print(f"\nConfiguration:")
    print(f"  Task: {config.get('task', 'unknown')}")
    print(f"  Max examples: {config.get('max_examples', 'N/A')}")
    print(f"  Top-k heads: {config.get('top_k_heads', 'N/A')}")
    print(f"  Vulnerability threshold: {config.get('vulnerability_threshold', 'N/A')}")

    # Circuit sizes
    print(f"\nCircuit Sizes:")
    print(f"  Base model: {len(results.get('base_circuit', []))} heads")
    print(f"  SFT model: {len(results.get('sft_circuit', []))} heads")
    print(f"  RL model: {len(results.get('rl_circuit', []))} heads")

    # Overlaps
    base_heads = set((h['layer'], h['head']) for h in results.get('base_circuit', []))
    sft_heads = set((h['layer'], h['head']) for h in results.get('sft_circuit', []))
    rl_heads = set((h['layer'], h['head']) for h in results.get('rl_circuit', []))

    if base_heads:
        sft_overlap = len(base_heads & sft_heads)
        rl_overlap = len(base_heads & rl_heads)

        print(f"\nCircuit Overlap with Base:")
        print(f"  SFT: {sft_overlap}/{len(base_heads)} ({100*sft_overlap/len(base_heads):.1f}%)")
        print(f"  RL: {rl_overlap}/{len(base_heads)} ({100*rl_overlap/len(base_heads):.1f}%)")

    # Faithfulness (NEW)
    faithfulness = results.get('faithfulness', {})
    if faithfulness:
        print(f"\nFaithfulness Metrics (Equation 4):")
        for model, metrics in faithfulness.items():
            if isinstance(metrics, dict):
                f_score = metrics.get('faithfulness', 0)
                print(f"  {model.upper()}: {f_score:.4f}")

    # DCM Summary (NEW)
    dcm_data = results.get('dcm_analysis', {})
    if dcm_data:
        print(f"\nDCM Analysis (Equation 3):")
        for model, hypotheses in dcm_data.items():
            if hypotheses:
                print(f"  {model.upper()}:")
                for hyp, result in hypotheses.items():
                    if isinstance(result, dict):
                        n_active = len(result.get('active_heads', []))
                        print(f"    {hyp}: {n_active} active heads")

    # Vulnerable circuits
    vulnerable = results.get('vulnerable_circuits', [])
    print(f"\nVulnerable Circuits: {len(vulnerable)} heads")

    if vulnerable:
        print("\nTop 5 Most Vulnerable Heads:")
        for i, head in enumerate(vulnerable[:5], 1):
            print(f"  {i}. Layer {head['layer']}, Head {head['head']}")
            print(f"     SFT ΔF: {head['sft_delta']:.4f}")
            print(f"     RL ΔF: {head['rl_delta']:.4f}")
            print(f"     Vulnerability: {head['vulnerability']:.4f}")

    # Key finding
    if base_heads:
        sft_overlap = len(base_heads & sft_heads)
        rl_overlap = len(base_heads & rl_heads)

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
        print("Usage: python visualization.py <results_json_path>")
        sys.exit(1)

    results_path = sys.argv[1]

    # Print summary
    print_circuit_summary(results_path)

    # Generate all visualizations
    generate_all_visualizations(results_path)