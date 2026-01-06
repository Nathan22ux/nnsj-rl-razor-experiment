import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def compute_pareto_frontier(x_vals, y_vals):
    """
    Compute the Pareto frontier for maximizing both x and y.
    
    Args:
        x_vals: List of x values (e.g., new task performance)
        y_vals: List of y values (e.g., prior task performance)
        
    Returns:
        Tuple of (pareto_x, pareto_y) containing points on the frontier
    """
    # Combine and sort by x
    points = list(zip(x_vals, y_vals))
    points.sort(key=lambda p: p[0])
    
    # Find Pareto frontier
    pareto = []
    max_y = -float('inf')
    
    for x, y in points:
        if y >= max_y:
            pareto.append((x, y))
            max_y = y
    
    if len(pareto) > 0:
        pareto_x, pareto_y = zip(*pareto)
        return list(pareto_x), list(pareto_y)
    else:
        return [], []


def plot_pareto_frontier(results, dataset_name="math"):
    """
    Create the Pareto frontier plot (Figure 2 from the paper).
    Shows New Task Performance vs Prior Task Performance trade-off.
    
    Args:
        results: Dictionary containing 'sft' and 'rl' results
        dataset_name: Name of the dataset (for title)
    """
    print("\n" + "="*70)
    print("CREATING PARETO FRONTIER PLOT (Figure 2)")
    print("="*70)
    
    # Extract data (skip missing PT from older result files)
    print("\n Extracting data from results...")
    sft_points = [(r.get('NT'), r.get('PT')) for r in results.get('sft', [])]
    rl_points = [(r.get('NT'), r.get('PT')) for r in results.get('rl', [])]

    sft_points = [(nt, pt) for nt, pt in sft_points if nt is not None and pt is not None]
    rl_points = [(nt, pt) for nt, pt in rl_points if nt is not None and pt is not None]

    sft_nt = [nt for nt, _ in sft_points]
    sft_pt = [pt for _, pt in sft_points]
    rl_nt = [nt for nt, _ in rl_points]
    rl_pt = [pt for _, pt in rl_points]

    if not sft_points and not rl_points:
        print("⚠️  No PT values found in results; skipping Pareto frontier plot.")
        return
    
    print(f"Extracted {len(sft_nt)} SFT results and {len(rl_nt)} RL results")
    
    # Create figure
    plt.figure(figsize=(10, 7))
    
    # Plot all points
    plt.scatter(sft_nt, sft_pt, label='SFT', alpha=0.5, s=80, color='orange', marker='o')
    plt.scatter(rl_nt, rl_pt, label='RL (GRPO)', alpha=0.5, s=80, color='blue', marker='s')
    
    # Compute and plot Pareto frontiers
    sft_pareto_x, sft_pareto_y = compute_pareto_frontier(sft_nt, sft_pt)
    rl_pareto_x, rl_pareto_y = compute_pareto_frontier(rl_nt, rl_pt)
    
    if len(sft_pareto_x) > 1:
        plt.plot(sft_pareto_x, sft_pareto_y, 'o--', color='orange', 
                linewidth=2.5, markersize=8, label='SFT Frontier', alpha=0.8)
    
    if len(rl_pareto_x) > 1:
        plt.plot(rl_pareto_x, rl_pareto_y, 's--', color='blue', 
                linewidth=2.5, markersize=8, label='RL Frontier', alpha=0.8)
    
    # Labels and title
    plt.xlabel('New Task Accuracy', fontsize=14, fontweight='bold')
    plt.ylabel('Avg. Score on Previous Tasks', fontsize=14, fontweight='bold')
    plt.title(f'Pareto Frontier: SFT vs RL ({dataset_name.capitalize()})', 
             fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    filename = f'pareto_frontier_{dataset_name}.png'
    os.makedirs("results", exist_ok=True)
    plt.savefig(f'results/{filename}', dpi=200, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()
    
    print("\n" + "="*70)
    print("PARETO FRONTIER PLOT COMPLETE")
    print("="*70)

def plot_results(results):
    """
    Create visualizations for the experiment results.
    Generates two plots:
    1. KL vs Prior Task Performance (showing forgetting)
    2. SFT vs RL Comparison
    
    Args:
        results: Dictionary containing 'sft' and 'rl' results
    """
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Extract data (KL is always present; PT may be missing in older result files)
    print("\n Extracting data from results...")
    sft_kl = [r.get('kl_divergence') for r in results.get('sft', []) if r.get('kl_divergence') is not None]
    rl_kl = [r.get('kl_divergence') for r in results.get('rl', []) if r.get('kl_divergence') is not None]

    sft_prior = [r.get('PT') for r in results.get('sft', []) if r.get('PT') is not None]
    rl_prior = [r.get('PT') for r in results.get('rl', []) if r.get('PT') is not None]

    print(f"Extracted {len(sft_kl)} SFT KL results and {len(rl_kl)} RL KL results")
    
    # KL vs Prior Task (showing forgetting) - only if PT exists
    if sft_prior and rl_prior:
        # Need aligned pairs; filter per-run pairs
        sft_pairs = [(r.get('kl_divergence'), r.get('PT')) for r in results.get('sft', [])]
        rl_pairs = [(r.get('kl_divergence'), r.get('PT')) for r in results.get('rl', [])]
        sft_pairs = [(k, p) for k, p in sft_pairs if k is not None and p is not None]
        rl_pairs = [(k, p) for k, p in rl_pairs if k is not None and p is not None]

        if sft_pairs or rl_pairs:
            sft_kl_plot = [k for k, _ in sft_pairs]
            sft_prior_plot = [p for _, p in sft_pairs]
            rl_kl_plot = [k for k, _ in rl_pairs]
            rl_prior_plot = [p for _, p in rl_pairs]

            print("\n Creating Plot 1: KL vs Prior Task Performance...")
            plt.figure(figsize=(10, 6))
            plt.scatter(sft_kl_plot, sft_prior_plot, label='SFT', alpha=0.6, s=50)
            plt.scatter(rl_kl_plot, rl_prior_plot, label='RL', alpha=0.6, s=50)
            plt.xlabel('KL Divergence', fontsize=12)
            plt.ylabel('Prior Task Performance (%)', fontsize=12)
            plt.title('KL Predicts Forgetting (Lower KL = Less Forgetting)', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            os.makedirs("results", exist_ok=True)
            plt.savefig('results/kl_vs_forgetting.png', dpi=150)
            print("Saved: kl_vs_forgetting.png")
            plt.show()
        else:
            print("\n⚠️  Skipping Plot 1 (no aligned KL/PT pairs).")
    else:
        print("\n⚠️  Skipping Plot 1 (PT missing in results).")
    
    # Comparison plot
    print("\n Creating Plot 2: SFT vs RL Comparison...")
    plt.figure(figsize=(10, 6))
    
    # Plot SFT and RL results
    methods = ['SFT', 'RL']
    prior_scores = [
        np.mean(sft_prior) if sft_prior else 0.0,
        np.mean(rl_prior) if rl_prior else 0.0
    ]
    kl_divs = [np.mean(sft_kl), np.mean(rl_kl)]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, prior_scores, width, label='Prior Task Score', alpha=0.8)
    ax.bar(x + width/2, kl_divs, width, label='KL Divergence', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('SFT vs RL: Forgetting Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/sft_vs_rl_comparison.png', dpi=150)
    print(" Saved: sft_vs_rl_comparison.png")
    plt.show()
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)

def plot_NT_PT(results):
    """
    NEW TASK vs PRIOR TASK graphs
    """

    sft_points = [(r.get("NT"), r.get("PT")) for r in results.get("sft", [])]
    rl_points = [(r.get("NT"), r.get("PT")) for r in results.get("rl", [])]

    sft_points = [(nt, pt) for nt, pt in sft_points if nt is not None and pt is not None]
    rl_points = [(nt, pt) for nt, pt in rl_points if nt is not None and pt is not None]

    if not sft_points and not rl_points:
        print("⚠️  No PT values found in results; skipping NT vs PT plot.")
        return

    sft_nt = [nt for nt, _ in sft_points]
    sft_pt = [pt for _, pt in sft_points]
    rl_nt = [nt for nt, _ in rl_points]
    rl_pt = [pt for _, pt in rl_points]

    plt.figure(figsize=(10,8))
    plt.scatter(sft_nt, sft_pt, color="orange", label="SFT", alpha=0.7, s=40)
    plt.scatter(rl_nt, rl_pt, color="blue", label="RL (GRPO)", alpha=0.7, s=40)

    plt.xlabel("New Task Performance (NT)", fontsize=12)
    plt.ylabel("Prior Tasks Performance (PT)", fontsize=12)
    plt.title("NT vs PT Comparison (SFT vs RL)", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/nt_vs_pt.png", dpi=300)
    print("figure saved nt_vs_pt")
    plt.show()