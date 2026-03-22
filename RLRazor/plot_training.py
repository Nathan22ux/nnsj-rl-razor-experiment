"""
plot_training.py — Plot RL training metrics from training_metrics.json

Usage:
    python plot_training.py <path/to/training_metrics.json>
    python plot_training.py checkpoints/rl/training_metrics.json

Generates: training_metrics.png (4-panel plot)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def smooth(values, window=10):
    """Simple moving average for smoothing noisy curves."""
    if len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def plot_training_metrics(metrics_path: str):
    metrics = load_metrics(metrics_path)

    # Split by mu iteration
    mu_values = sorted(set(m["mu"] for m in metrics))
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]

    steps        = [m["step"] for m in metrics]
    loss         = [m["loss"] for m in metrics]
    reward       = [m["reward_mean"] for m in metrics]
    frac_zero    = [m["frac_zero_std"] for m in metrics]
    lr           = [m["lr"] for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("RL Training Metrics", fontsize=14, fontweight="bold")

    def add_mu_spans(ax):
        """Shade background by mu iteration."""
        if len(mu_values) <= 1:
            return
        for i, mu in enumerate(mu_values):
            mu_steps = [m["step"] for m in metrics if m["mu"] == mu]
            if mu_steps:
                ax.axvspan(min(mu_steps), max(mu_steps),
                           alpha=0.06, color=colors[i % len(colors)],
                           label=f"μ={mu}")

    # --- Panel 1: Loss ---
    ax = axes[0, 0]
    ax.plot(steps, loss, alpha=0.3, color="steelblue", linewidth=0.8)
    ax.plot(steps, smooth(loss), color="steelblue", linewidth=1.8, label="loss (smoothed)")
    add_mu_spans(ax)
    ax.set_title("Training Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Reward mean ---
    ax = axes[0, 1]
    ax.plot(steps, reward, alpha=0.3, color="darkorange", linewidth=0.8)
    ax.plot(steps, smooth(reward), color="darkorange", linewidth=1.8, label="reward mean (smoothed)")
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="random baseline (4-choice)")
    add_mu_spans(ax)
    ax.set_title("Reward Mean")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: frac_zero_std (key health metric) ---
    ax = axes[1, 0]
    ax.plot(steps, frac_zero_std, alpha=0.3, color="crimson", linewidth=0.8)
    ax.plot(steps, smooth(frac_zero_std), color="crimson", linewidth=1.8,
            label="frac_zero_std (smoothed)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="50% threshold")
    add_mu_spans(ax)
    ax.set_title("Fraction of Dead Groups (frac_zero_std)\nLower = more groups learning")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Learning rate schedule ---
    ax = axes[1, 1]
    ax.plot(steps, lr, color="seagreen", linewidth=1.8, label="learning rate")
    add_mu_spans(ax)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Step")
    ax.set_ylabel("LR")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(metrics_path).parent / "training_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_training.py <path/to/training_metrics.json>")
        sys.exit(1)
    plot_training_metrics(sys.argv[1])
