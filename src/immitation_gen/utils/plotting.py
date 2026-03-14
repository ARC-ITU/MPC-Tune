from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def draw(
    baseline_weights: Dict[str, float],
    baseline_traj: np.ndarray,
    learned_weights: Dict[str, float],
    learned_traj: np.ndarray,
    details: Dict[str, float],
    ref_track,
    data_path: Path,
    exp_idx: int,
):
    """Render and persist a side-by-side trajectory plot with run details."""
    plt.figure(figsize=(12, 6))
    plt.plot(
        baseline_traj[:, 0], baseline_traj[:, 1], "bo-", label=f"Baseline trajectory", linewidth=1, markersize=1
    )
    track_coords = np.array(ref_track.coords)
    plt.plot(track_coords[:, 0], track_coords[:, 1], "k--", label="Reference Track", linewidth=2)
    plt.plot(
        learned_traj[:, 0], learned_traj[:, 1], "go-", label=f"Learned trajectory", linewidth=1, markersize=1
    )
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Vehicle Trajectory Using MPC")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # Add details text on the right side
    details_text = ""
    details_text += "Baseline Weights:\n"
    for key, value in baseline_weights.items():
        details_text += f"  {key}: {value}\n"
    details_text += "\nFinal Learned Weights:\n"
    for key, value in learned_weights.items():
        details_text += f"  {key}: {value}\n"
    details_text += "\nTraining Details:\n"
    for key, value in details.items():
        details_text += f"  {key}: {value}\n"
    plt.text(
        1.02,
        0.5,
        details_text,
        transform=plt.gca().transAxes,
        verticalalignment="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black"),
    )

    name = f"experiment_{exp_idx}_trajectory.png"
    plt.savefig(data_path / name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_policy_evolution(
    initial_weights: Dict[str, float],
    history: Dict,
    data_path: Path,
    exp_idx: int,
    weight_names: List[str] = ["goal_speed", "tracking", "orientation", "acceleration"],
):
    """Plot the evolution of policy mean and std deviation over iterations."""
    means = np.array(history["means"])
    stds = np.array(history["stds"])
    iterations = np.arange(len(means))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Policy Distribution Evolution - Experiment {exp_idx}", fontsize=14)

    # Prepare initial weight values in the same order as weight_names
    initial_vals = [initial_weights.get(name, np.nan) for name in weight_names]
    x_max = iterations[-1] if len(iterations) > 0 else 0

    for idx, (ax, name) in enumerate(zip(axes.flat, weight_names)):
        ax.plot(iterations, means[:, idx], "b-", linewidth=2, label="Mean")
        ax.fill_between(
            iterations,
            means[:, idx] - stds[:, idx],
            means[:, idx] + stds[:, idx],
            alpha=0.3,
            label="±1 std",
        )

        init_val = initial_vals[idx]
        if not np.isnan(init_val):
            ax.hlines(init_val, 0, x_max, colors="r", linestyles="--", linewidth=1.5, label="Initial")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Weight Value")
        ax.set_title(f"{name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(data_path / f"experiment_{exp_idx}_policy_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_rewards_history(history: Dict, data_path: Path, exp_idx: int):
    """Plot reward statistics over training iterations."""
    rewards_mean = np.array(history["rewards_mean"])
    rewards_max = np.array(history["rewards_max"])
    rewards_min = np.array(history["rewards_min"])
    iterations = np.arange(len(rewards_mean))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, rewards_mean, "b-", linewidth=2, label="Mean Reward")
    plt.plot(iterations, rewards_max, "g--", linewidth=1, label="Max Reward")
    plt.plot(iterations, rewards_min, "r--", linewidth=1, label="Min Reward")
    plt.fill_between(iterations, rewards_min, rewards_max, alpha=0.2)

    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title(f"Reward Evolution - Experiment {exp_idx}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(data_path / f"experiment_{exp_idx}_rewards.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_experiments_summary(all_histories: List[Dict], data_path: Path):
    """Plot average rewards across all experiments."""
    num_experiments = len(all_histories)

    # Extract reward means from all experiments
    all_rewards_mean = [np.array(h["rewards_mean"]) for h in all_histories]

    # Find the minimum length (in case experiments have different iteration counts)
    min_len = min(len(r) for r in all_rewards_mean)
    all_rewards_mean = [r[:min_len] for r in all_rewards_mean]

    # Convert to array for easy computation
    rewards_array = np.array(all_rewards_mean)  # shape: (num_exp, num_iters)

    mean_across_exp = np.mean(rewards_array, axis=0)
    std_across_exp = np.std(rewards_array, axis=0)
    iterations = np.arange(min_len)

    plt.figure(figsize=(12, 6))

    for i, rewards in enumerate(all_rewards_mean):
        plt.plot(iterations, rewards, alpha=0.3, linewidth=1, color="gray")

    plt.plot(iterations, mean_across_exp, "b-", linewidth=3, label=f"Average (n={num_experiments})")
    plt.fill_between(
        iterations,
        mean_across_exp - std_across_exp,
        mean_across_exp + std_across_exp,
        alpha=0.3,
        color="blue",
        label="±1 std across experiments",
    )

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Mean Reward", fontsize=12)
    plt.title(f"Training Progress Across All Experiments (n={num_experiments})", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.savefig(data_path / "all_experiments_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Summary plot saved: all_experiments_summary.png")


def plot_sampled_trajectories(
    trajectories: List[np.ndarray],
    target_trajectory: np.ndarray,
    iter_idx: int,
    exp_idx: int,
    data_path: Path,
):
    """Plot all sampled trajectories at a given snapshot iteration.

    Args:
        trajectories      (list of np.ndarray, each shape (T, state_dim)):
                              One trajectory per sampled weight vector.
        target_trajectory (np.ndarray, shape (T, state_dim)):
                              Baseline/target trajectory for reference.
        iter_idx          (int):   Iteration index this snapshot was taken at.
        exp_idx           (int):   Experiment index (for file naming).
        data_path         (Path):  Directory to save the image.
    """
    _SAMPLE_COLORS = [
        "#8B0000",  # dark red
        "#A0522D",  # sienna (dark brownish)
        "#6B3A2A",  # dark brown
        "#B8860B",  # dark goldenrod
        "#556B2F",  # dark olive green
        "#2F4F4F",  # dark slate gray
        "#4B0082",  # indigo
        "#800080",  # purple
        "#8B4513",  # saddle brown
        "#C0392B",  # dark crimson
    ]

    plt.figure(figsize=(12, 6))

    for traj in trajectories:
        color = _SAMPLE_COLORS[np.random.randint(len(_SAMPLE_COLORS))]
        plt.plot(traj[:, 0], traj[:, 1], "-", color=color, alpha=0.4, linewidth=0.8)

    plt.plot(
        target_trajectory[:, 0], target_trajectory[:, 1],
        "k--", linewidth=2, label="Target trajectory",
    )

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Vehicle Trajectory Using MPC")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    details_text = f"Experiment: {exp_idx}\nIteration: {iter_idx}\nSamples: {len(trajectories)}"
    plt.text(
        1.02, 0.5,
        details_text,
        transform=plt.gca().transAxes,
        verticalalignment="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black"),
    )

    plt.savefig(data_path / f"iter_{iter_idx}_sampled_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_initial_distribution(data_path: Path, mean: float, var: float):
    """Plot the initial weight distribution used for HighMPC.

    Parameters:
        data_path: directory to save the plot
        mean: mean of initial distribution
        var: variance of initial distribution
    """
    std = np.sqrt(var)

    x = np.linspace(mean - 4 * std, mean + 4 * std, 1000)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Initial Weight Distribution", color="blue")

    for i in range(-3, 4):
        plt.axvline(mean + i * std, color="green", linestyle="--", alpha=0.8)
        plt.text(mean + i * std, max(y) * 0.1, f"{mean + i * std:.2f}", rotation=270, verticalalignment="bottom", color="black")

    plt.title("Initial Weight Distribution for HighMPC")
    plt.xlabel("Weight Value")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)

    plt.savefig(data_path / "initial_weight_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
