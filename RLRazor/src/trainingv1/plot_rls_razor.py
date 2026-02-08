import json
import matplotlib.pyplot as plt
from trainingv1.eval_pareto import pareto_kl_nt, pareto_forgetting_nt
# if runing test.py
# from src.trainingv1.eval_pareto import pareto_kl_nt, pareto_forgetting_nt


# ------------------------------------------------------------
# Utility: load experiment results
# ------------------------------------------------------------
def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


# ------------------------------------------------------------
# Plot 1: NT vs Forward KL (Pareto)
# ------------------------------------------------------------
def plot_nt_vs_kl(results, save_path=None):
    sft = [r for r in results if r["type"] == "SFT"]
    rl = [r for r in results if r["type"] == "RL"]

    pareto = pareto_kl_nt(results)

    plt.figure(figsize=(6, 4))

    # scatter points
    plt.scatter(
        [r["KL"] for r in sft],
        [r["NT"] for r in sft],
        marker="x",
        label="SFT",
        alpha=0.7,
    )
    plt.scatter(
        [r["KL"] for r in rl],
        [r["NT"] for r in rl],
        marker="o",
        label="RL (Dr.GRPO)",
        alpha=0.7,
    )

    # pareto frontier
    plt.plot(
        [r["KL"] for r in pareto],
        [r["NT"] for r in pareto],
        linewidth=2,
        label="Pareto Frontier",
    )

    plt.xlabel("Forward KL  (π₀ || π)")
    plt.ylabel("New-Task Performance (NT)")
    plt.title("RL’s Razor: NT vs Forward KL")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ------------------------------------------------------------
# Plot 2: NT vs Forgetting (Pareto)
# ------------------------------------------------------------
def plot_nt_vs_forgetting(results, save_path=None):
    sft = [r for r in results if r["type"] == "SFT"]
    rl = [r for r in results if r["type"] == "RL"]

    pareto = pareto_forgetting_nt(results)

    plt.figure(figsize=(6, 4))

    plt.scatter(
        [r["forgetting"] for r in sft],
        [r["NT"] for r in sft],
        marker="x",
        label="SFT",
        alpha=0.7,
    )
    plt.scatter(
        [r["forgetting"] for r in rl],
        [r["NT"] for r in rl],
        marker="o",
        label="RL (Dr.GRPO)",
        alpha=0.7,
    )

    plt.plot(
        [r["forgetting"] for r in pareto],
        [r["NT"] for r in pareto],
        linewidth=2,
        label="Pareto Frontier",
    )

    plt.xlabel("Forgetting (Old-task drop)")
    plt.ylabel("New-Task Performance (NT)")
    plt.title("RL’s Razor: NT vs Forgetting")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # example usage
    results = load_results("results.json")

    plot_nt_vs_kl(
        results,
        save_path="nt_vs_kl.png"
    )

    plot_nt_vs_forgetting(
        results,
        save_path="nt_vs_forgetting.png"
    )
