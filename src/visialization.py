import matplotlib.pyplot as plt
import numpy as np

def plot_results(results):
    """Create visualizations for experiment results."""
    
    # Extract data
    sft_prior = [r['prior_task_score'] for r in results['sft']]
    sft_kl = [r['kl_divergence'] for r in results['sft']]
    
    rl_prior = [r['prior_task_score'] for r in results['rl']]
    rl_kl = [r['kl_divergence'] for r in results['rl']]
    
    # KL vs Prior Task (showing forgetting)
    plt.figure(figsize=(10, 6))
    
    plt.scatter(sft_kl, sft_prior, label='SFT', alpha=0.6, s=50)
    plt.scatter(rl_kl, rl_prior, label='RL', alpha=0.6, s=50)
    
    plt.xlabel('KL Divergence', fontsize=12)
    plt.ylabel('Prior Task Performance', fontsize=12)
    plt.title('KL Predicts Forgetting (Lower KL = Less Forgetting)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('kl_vs_forgetting.png', dpi=150)
    plt.show()
    
    # Comparison plot
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
    plt.savefig('sft_vs_rl_comparison.png', dpi=150)
    plt.show()
