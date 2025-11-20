import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    
    # Extract data
    print("\n Extracting data from results...")
    sft_prior = [r['PT'] for r in results['sft']]
    sft_kl = [r['kl_divergence'] for r in results['sft']]
    
    rl_prior = [r['PT'] for r in results['rl']]
    rl_kl = [r['kl_divergence'] for r in results['rl']]
    print(f"Extracted {len(sft_prior)} SFT results and {len(rl_prior)} RL results")
    
    # KL vs Prior Task (showing forgetting)
    print("\n Creating Plot 1: KL vs Prior Task Performance...")
    plt.figure(figsize=(10, 6))
    
    plt.scatter(sft_kl, sft_prior, label='SFT', alpha=0.6, s=50)
    plt.scatter(rl_kl, rl_prior, label='RL', alpha=0.6, s=50)
    
    plt.xlabel('KL Divergence', fontsize=12)
    plt.ylabel('Prior Task Performance', fontsize=12)
    plt.title('KL Predicts Forgetting (Lower KL = Less Forgetting)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/kl_vs_forgetting.png', dpi=150)
    print("Saved: kl_vs_forgetting.png")
    plt.show()
    
    # Comparison plot
    print("\n Creating Plot 2: SFT vs RL Comparison...")
    plt.figure(figsize=(10, 6))
    
    # Plot SFT and RL results
    methods = ['SFT', 'RL']
    prior_scores = [np.mean(sft_prior), np.mean(rl_prior)]
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

    sft_nt = [r["NT"] for r in results["sft"]]
    sft_pt = [r["PT"] for r in results["sft"]]

    rl_nt = [r["NT"] for r in results["rl"]]
    rl_pt = [r["PT"] for r in results["rl"]]

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