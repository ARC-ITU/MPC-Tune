from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Tuple

import numpy as np

from immitation_gen.policy.high_mpc import HighMPC
from immitation_gen.mpc.traffic.simulate import (
    run_mpc_simulation as traffic_run_mpc_simulation,
)

from immitation_gen.utils.plotting import (
    draw,
    plot_policy_evolution,
    plot_rewards_history,
    plot_all_experiments_summary,
    plot_initial_distribution,
    plot_sampled_trajectories,
)

# Per-variable clip bounds: [goal_speed, tracking, orientation, acceleration]
CLIP_LOW  = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
CLIP_HIGH = np.array([ np.inf,  np.inf,  np.inf,  np.inf])

# Iterations at which to snapshot sampled weight vectors for trajectory visualisation
SNAPSHOT_ITERS = [0,1,2,4,10,20]


def build_training_config() -> Dict[str, float]:
    """Centralize training hyperparameters and plotting metadata."""
    return {
        "max_iter": 15,
        "samples_per_iter": 30,  # HighMPC.Policy.N
        "beta": 3.0,              # Hyperparameter for scaling rewards in the expectation step
        "clip_low": CLIP_LOW,     # shape (4,): [goal_speed, tracking, orientation, acceleration]
        "clip_high": CLIP_HIGH,   # shape (4,): [goal_speed, tracking, orientation, acceleration]
        # Initial Gaussian for weights [gs, tracking, orientation, acceleration]
        "mean_init": 13.0,
        "var_init":  25.0,
    }


def weights_from_vector(vec: np.ndarray) -> Dict[str, float]:
    """Map a 4-dim vector to the MPC weight dictionary with rounding."""
    return {
        "goal_speed": round(float(vec[0]), 2),
        "tracking": round(float(vec[1]), 2),
        "orientation": round(float(vec[2]), 2),
        "acceleration": round(float(vec[3]), 2),
    }

def run_experiment(
    exp_idx: int,
    initial_state: np.ndarray,
    baseline_weights: Dict[str, float],
    data_path: Path,
) -> Tuple[Dict[str, float], Dict]:
    """Run a single experiment: baseline rollout, train HighMPC, compare rollout, save plot."""
    cfg = build_training_config()

    # 1) Baseline rollout to create target trajectory
    baseline_states, _, _, ref_track = traffic_run_mpc_simulation(
        initial_state, baseline_weights
    )

    # 2) HighMPC setup and training
    D = 4  # number of weights
    mean = [cfg["mean_init"]] * D
    covariance = np.diag([cfg["var_init"]] * D)
    N = cfg["samples_per_iter"]

    trainer = HighMPC(
        mean=mean,
        covariance=covariance,
        N=N,
        cliplow=cfg["clip_low"],
        cliphigh=cfg["clip_high"],
        target_trajectory=baseline_states,
        MPC=traffic_run_mpc_simulation,
    )

    t0 = time.time()
    mean_vec, history = trainer.policy_search(
        initial_state=initial_state,
        max_iter=cfg["max_iter"],
        beta=cfg["beta"],
        track_history=True,
        snapshot_iters=SNAPSHOT_ITERS,
    )
    elapsed = time.time() - t0

    learned_weights = weights_from_vector(mean_vec)

    # 3) Rollout with learned weights
    learned_states, _, _, _ = traffic_run_mpc_simulation(initial_state, learned_weights)

    # 4) Persist plot and print summary
    details = {
        "iterations": cfg["max_iter"],
        "samples_per_iter": cfg["samples_per_iter"],
        "beta": cfg["beta"],
        "train_time_sec": round(elapsed, 2),
    }

    draw(
        baseline_weights=baseline_weights,
        baseline_traj=baseline_states,
        learned_weights=learned_weights,
        learned_traj=learned_states,
        details=details,
        ref_track=ref_track,
        data_path=data_path,
        exp_idx=exp_idx,
    )

    # Plot policy evolution
    plot_policy_evolution(baseline_weights, history, data_path, exp_idx)

    # Plot reward history
    plot_rewards_history(history, data_path, exp_idx)

    # 5) Snapshot trajectory plots — one image per snapshot iteration
    progress_dir = data_path / "progress_over_iterations" / f"experiment_{exp_idx}"
    progress_dir.mkdir(parents=True, exist_ok=True)

    for iter_idx, z in history["sampled_weights"].items():
        trajectories = []
        for i in range(len(z)):
            w = weights_from_vector(z[i])
            traj, _, _, _ = traffic_run_mpc_simulation(initial_state, w)
            trajectories.append(traj)
        plot_sampled_trajectories(
            trajectories=trajectories,
            target_trajectory=baseline_states,
            iter_idx=iter_idx,
            exp_idx=exp_idx,
            data_path=progress_dir,
        )

    print(
        f"Experiment {exp_idx}: learned {learned_weights} from baseline {baseline_weights} in {elapsed:.2f}s"
    )

    return learned_weights, history

def generate_random_experiments(n: int) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
    """Generate n initial states and baseline weight dictionaries."""
    states: List[np.ndarray] = []
    weights: List[Dict[str, float]] = []

    for _ in range(n):
        x = round(np.random.uniform(1, 5), 2)
        y = round(np.random.uniform(1, 15), 2)
        weights.append(
            {
                "goal_speed": round(np.random.uniform(2, 8.0), 2),
                "tracking": round(np.random.uniform(2, 8.0), 2),
                "orientation": round(np.random.uniform(2, 8.0), 2),
                "acceleration": round(np.random.uniform(2, 8.0), 2),
            }
        )
        
        # [x, y, heading, vx, vy, omega]
        states.append(np.array([x, y, 0.0, 1.0, 0.0, 0.0]))

    return states, weights


def main():
    # np.random.seed(42)  # For reproducible experiments
    exp_root = Path(f"data/{datetime.now().isoformat(timespec='seconds')}")
    exp_root.mkdir(parents=True, exist_ok=True)

    NUM_EXPERIMENTS = 2
    states, weights = generate_random_experiments(NUM_EXPERIMENTS)
    # Plot initial distribution using configured mean/variance
    cfg = build_training_config()
    plot_initial_distribution(exp_root, cfg["mean_init"], cfg["var_init"]) 

    overall_start = time.time()
    all_histories = []
    finals = []
    for idx, (state, w) in enumerate(zip(states, weights), start=1):
        print(f"Running experiment {idx} with state={state} and baseline weights={w}")
        learned_weights, history = run_experiment(
            exp_idx=idx, 
            initial_state=state, 
            baseline_weights=w, 
            data_path=exp_root
        )
        all_histories.append(history)
        finals.append({"experiment": idx, "state": state, "baseline_weights": w, "learned_weights": learned_weights})

    overall_elapsed = time.time() - overall_start

    # Save final results
    for res in finals:
        print(f"\nExperiment {res['experiment']}:")
        print(f"  Initial State: {res['state']}")
        print(f"  Baseline Weights: {res['baseline_weights']}")
        print(f"  Learned Weights: {res['learned_weights']}")
    
    # Create summary plot across all experiments
    print(f"\nCreating summary plot across all {NUM_EXPERIMENTS} experiments...")
    plot_all_experiments_summary(all_histories, exp_root)

    print(f"\n✅ All experiments completed in {overall_elapsed:.2f}s (~{overall_elapsed/60:.2f} min)")
    print(f"📁 Results saved to: {exp_root}")



if __name__ == "__main__":
    main()

    