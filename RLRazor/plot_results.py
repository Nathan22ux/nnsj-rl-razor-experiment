"""
plot_results.py - Generate NT/PT Pareto frontier and KL divergence plots
Usage: python plot_results.py results/results_science_full_v2\(13\).json
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load(path):
    with open(path) as f:
        return json.load(f)


def pareto_frontier(points, x_key, y_key, maximize_both=True):
    """Return Pareto-optimal points (maximize both x and y by default)."""
    sorted_pts = sorted(points, key=lambda p: p[x_key])
    frontier = []
    best_y = -np.inf if maximize_both else np.inf
    for p in sorted_pts:
        y = p[y_key]
        if maximize_both and y >= best_y:
            frontier.append(p)
            best_y = y
        elif not maximize_both and y <= best_y:
            frontier.append(p)
            best_y = y
    return frontier


def plot_nt_pt(sft, rl, save_path="nt_vs_pt.png"):
    fig, ax = plt.subplots(figsize=(8, 5))

    sft_nt = [r["NT"] for r in sft]
    sft_pt = [r["PT"] for r in sft]
    rl_nt  = [r["NT"] for r in rl]
    rl_pt  = [r["PT"] for r in rl]

    ax.scatter(sft_nt, sft_pt, marker="x", color="steelblue",
               alpha=0.6, s=50, label="SFT runs")
    ax.scatter(rl_nt,  rl_pt,  marker="o", color="tomato",
               alpha=0.6, s=50, label="RL (Dr.GRPO) runs")

    # Pareto frontiers — maximize NT (x) while keeping PT (y) as high as possible
    sft_pareto = pareto_frontier(sft, "NT", "PT", maximize_both=True)
    rl_pareto  = pareto_frontier(rl,  "NT", "PT", maximize_both=True)

    ax.plot([p["NT"] for p in sft_pareto], [p["PT"] for p in sft_pareto],
            color="steelblue", linewidth=2, linestyle="--", label="SFT Pareto")
    ax.plot([p["NT"] for p in rl_pareto],  [p["PT"] for p in rl_pareto],
            color="tomato",    linewidth=2, linestyle="--", label="RL Pareto")

    max_y = max(sft_pt + rl_pt)
    ax.set_ylim(max_y / 2, max_y * 1.02)

    ax.set_xlabel("New-Task Performance NT (%)", fontsize=12)
    ax.set_ylabel("Prior-Task Performance PT (%)", fontsize=12)
    ax.set_title("RL's Razor: NT vs PT (Pareto Frontier)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


def plot_nt_kl(sft, rl, save_path="nt_vs_kl.png"):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter([r["kl_divergence"] for r in sft],
               [r["NT"] for r in sft],
               marker="x", color="steelblue", alpha=0.6, s=50, label="SFT runs")
    ax.scatter([r["kl_divergence"] for r in rl],
               [r["NT"] for r in rl],
               marker="o", color="tomato",    alpha=0.6, s=50, label="RL (Dr.GRPO) runs")

    # Pareto: maximize NT while minimizing KL  → negate KL for pareto util
    sft_neg = [{"NT": r["NT"], "neg_kl": -r["kl_divergence"]} for r in sft]
    rl_neg  = [{"NT": r["NT"], "neg_kl": -r["kl_divergence"]} for r in rl]
    sft_pareto = pareto_frontier(sft_neg, "neg_kl", "NT", maximize_both=True)
    rl_pareto  = pareto_frontier(rl_neg,  "neg_kl", "NT", maximize_both=True)

    # Convert back
    sft_pareto_kl = sorted([-p["neg_kl"] for p in sft_pareto])
    sft_pareto_nt = [p["NT"] for p in sorted(sft_pareto, key=lambda p: p["neg_kl"], reverse=True)]
    rl_pareto_kl  = sorted([-p["neg_kl"] for p in rl_pareto])
    rl_pareto_nt  = [p["NT"] for p in sorted(rl_pareto,  key=lambda p: p["neg_kl"], reverse=True)]

    ax.plot(sft_pareto_kl, sft_pareto_nt, color="steelblue",
            linewidth=2, linestyle="--", label="SFT Pareto")
    ax.plot(rl_pareto_kl,  rl_pareto_nt,  color="tomato",
            linewidth=2, linestyle="--", label="RL Pareto")

    all_nt = [r["NT"] for r in sft] + [r["NT"] for r in rl]
    max_y = max(all_nt)
    ax.set_ylim(max_y / 2, max_y * 1.02)

    ax.set_xlabel("Forward KL Divergence  KL(π₀ ‖ π)", fontsize=12)
    ax.set_ylabel("New-Task Performance NT (%)", fontsize=12)
    ax.set_title("RL's Razor: NT vs KL Divergence (Pareto Frontier)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


def plot_pt_kl(sft, rl, save_path="pt_vs_kl.png"):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter([r["kl_divergence"] for r in sft],
               [r["PT"] for r in sft],
               marker="x", color="steelblue", alpha=0.6, s=50, label="SFT runs")
    ax.scatter([r["kl_divergence"] for r in rl],
               [r["PT"] for r in rl],
               marker="o", color="tomato", alpha=0.6, s=50, label="RL (Dr.GRPO) runs")

    # Pareto: maximize PT while minimizing KL
    sft_neg = [{"PT": r["PT"], "neg_kl": -r["kl_divergence"]} for r in sft]
    rl_neg  = [{"PT": r["PT"], "neg_kl": -r["kl_divergence"]} for r in rl]
    sft_pareto = pareto_frontier(sft_neg, "neg_kl", "PT", maximize_both=True)
    rl_pareto  = pareto_frontier(rl_neg,  "neg_kl", "PT", maximize_both=True)

    sft_pareto_kl = sorted([-p["neg_kl"] for p in sft_pareto])
    sft_pareto_pt = [p["PT"] for p in sorted(sft_pareto, key=lambda p: p["neg_kl"], reverse=True)]
    rl_pareto_kl  = sorted([-p["neg_kl"] for p in rl_pareto])
    rl_pareto_pt  = [p["PT"] for p in sorted(rl_pareto,  key=lambda p: p["neg_kl"], reverse=True)]

    ax.plot(sft_pareto_kl, sft_pareto_pt, color="steelblue",
            linewidth=2, linestyle="--", label="SFT Pareto")
    ax.plot(rl_pareto_kl,  rl_pareto_pt,  color="tomato",
            linewidth=2, linestyle="--", label="RL Pareto")

    all_pt = [r["PT"] for r in sft] + [r["PT"] for r in rl]
    max_y = max(all_pt)
    ax.set_ylim(max_y / 2, max_y * 1.02)

    ax.set_xlabel("Forward KL Divergence  KL(π₀ ‖ π)", fontsize=12)
    ax.set_ylabel("Prior-Task Performance PT (%)", fontsize=12)
    ax.set_title("RL's Razor: PT vs KL Divergence (Pareto Frontier)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/results_science_full_v2.json"
    data = load(path)
    sft  = data.get("sft", [])
    rl   = data.get("rl", [])

    print(f"Loaded {len(sft)} SFT runs, {len(rl)} RL runs")

    plot_nt_pt(sft, rl, save_path="nt_vs_pt.png")
    plot_nt_kl(sft, rl, save_path="nt_vs_kl.png")
    plot_pt_kl(sft, rl, save_path="pt_vs_kl.png")
