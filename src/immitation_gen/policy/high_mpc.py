import numpy as np
from abc import ABC, abstractmethod
from typing import Callable
from multiprocessing import Pool

class Policy:

    def __init__(self, mean, covariance, N, cliplow, cliphigh):
        """
        Initialize the Policy (Gaussian distribution over MPC weight vectors).

        Args:
            mean      (list | np.ndarray, shape (num_weights,)):
                          Initial mean of the Gaussian — one value per MPC weight
            covariance (list | np.ndarray, shape (num_weights, num_weights)):
                          Initial covariance matrix
            N         (int):
                          Number of samples to draw per policy-search iteration.
            For cliplow and cliphigh, a per-variable bound can be specified by passing a length-num_weights array, or a scalar can be passed to use the same bound for all weights.
            cliplow   (float | list | np.ndarray, shape (num_weights,)):
                          Per-variable lower clip bound applied after sampling.
                          Pass a scalar to use the same bound for all weights,
                          or a length-4 array for per-weight bounds.
            cliphigh  (float | list | np.ndarray, shape (num_weights,)):
                          Per-variable upper clip bound applied after sampling.
                          Same convention as cliplow.
        """
        self.mean = np.array(mean, dtype=float)
        self.covariance = np.array(covariance, dtype=float)
        self.N = N
        self.cliplow  = np.asarray(cliplow,  dtype=float) 
        self.cliphigh = np.asarray(cliphigh, dtype=float)  

    def sample(self):
        """
        Draw N weight-vector samples from the current policy distribution.
softmax
        Returns
        -------
        samples : np.ndarray, shape (N, num_weights)
            Sampled weight vectors, clipped to [cliplow, cliphigh] per variable.
        """
        samples = np.random.multivariate_normal(
            self.mean, self.covariance, self.N)                               
        samples = np.clip(samples, self.cliplow, self.cliphigh)
        return samples 

    def update(self, weights, samples):
        """
        Update policy mean and covariance via importance-weighted EM.

        Args:
            weights (np.ndarray, shape (N,)):
                Non-negative importance weights from the expectation step.
            samples (np.ndarray, shape (N, num_weights)):
                The same sample matrix returned by sample().

        Effects
        ------------
        self.mean       updated in-place, shape (num_weights,)
        self.covariance updated in-place, shape (num_weights, num_weights)
        """
        self.mean = weights.dot(samples) / np.sum(weights + 1e-8)              # mean update as weighted average of samples, shape (num_weights,) = (N,) dot (N, num_weights) / scalar
        Y = (np.sum(weights)**2 - np.sum(weights**2)) / np.sum(weights)        
        sum = 0.0
        for i in range(len(weights)):                                          
            diff = samples[i] - self.mean
            sum += weights[i] * np.outer(diff, diff)                           # covariance update as weighted average of outer products of sample deviations, shape (num_weights, num_weights) = scalar * (num_weights, num_weights)
        self.covariance = sum / (Y + 1e-8)

    def expectation(self, rewards, beta):
        """
        Normalize rewards
        Compute importance weights from a reward vector[scale using beta , exponential]

        Args:
            rewards (np.ndarray, shape (N,)):
                Scalar reward for each of the N sampled trajectories.
            beta (float):
                hyperparameter for scaling

        Returns
        -------
        weights : np.ndarray, shape (N,)
            Non-negative importance weights
        """
        mean = np.mean(rewards)
        std = np.std(rewards)

        std = 1e-8 if std == 0 else std                                 # prevent division by zero
        normalized_rewards = (rewards - mean) / std
        weights = np.exp(beta * normalized_rewards)
        return weights


class BasePolicySearch(ABC):                                                # Abstract base class for policy search algorithms, requires implementation of reward and policy_search methods by subclasses

    def __init__(self, mean, covariance, N, cliplow=-np.inf, cliphigh=np.inf):
        self.policy = Policy(mean, covariance, N, cliplow, cliphigh)        # composiing the policy search class with a policy instance

    @abstractmethod
    def reward(self, sampled_trajectory: np.ndarray) -> float: ...

    @abstractmethod
    def policy_search(self, initial_state, max_iter, beta, **kwargs): ...
 

class HighMPC(BasePolicySearch):
    """
        Probabilistic Policy Search for MPC
    """

    def __init__(self, mean, covariance, N, cliplow=-np.inf, cliphigh=np.inf, target_trajectory=None, MPC: Callable = None):
        super().__init__(mean, covariance, N, cliplow, cliphigh)
        self.target_trajectory = target_trajectory
        self.MPC = MPC

    def reward(self, sampled_trajectory: np.ndarray):                      # reward is negative distance between sampled trajectory and target trajectory, averaged over the length of the shorter trajectory
        # choose len of smaller trajectory and calculate reward using equilidian distance input is array x y points
        dimensions = sampled_trajectory.shape[1]
        min_len = min(len(sampled_trajectory), len(self.target_trajectory))

        total_reward = 0
        for i in range(min_len):
            reward = 0
            for d in range(dimensions):
                reward += (((sampled_trajectory[i][d] -
                           self.target_trajectory[i][d])**2))
            # negative of distance as reward
            total_reward += (-np.sqrt(reward))

        return total_reward / min_len

    def policy_search(self, initial_state, max_iter, beta, track_history=True, snapshot_iters=None):
        """
        Run policy search and optionally track training history.

        Args:
            initial_state  (np.ndarray, shape (state_dim,)):  Starting state for MPC rollouts.
            max_iter       (int):                             Number of EM iterations.
            beta           (float):                           Inverse temperature for expectation step.
            track_history  (bool):                            Record per-iteration statistics.
            snapshot_iters (list[int] | None):                Iteration indices at which to save
                                                              the full sample matrix z for later
                                                              trajectory visualisation.

        Returns:
            mean    (np.ndarray, shape (num_weights,)):  Learned policy mean.
            history (dict | None):                       Per-iteration stats when track_history=True.
                                                         Includes 'sampled_weights': dict mapping
                                                         snapshot iteration index -> z (N, num_weights).
        """
        snapshot_iters = set(snapshot_iters) if snapshot_iters is not None else set()

        history = {
            'means': [],
            'stds': [],
            'rewards_mean': [],
            'rewards_max': [],
            'rewards_min': [],
            'sampled_weights': {},   # iter_idx -> np.ndarray (N, num_weights)
        } if track_history else None

        for iter_idx in range(max_iter):
            rewards = np.zeros(shape=(self.policy.N))
            z = self.policy.sample()

            # Save sample matrix for snapshot iterations
            if track_history and iter_idx in snapshot_iters:
                history['sampled_weights'][iter_idx] = z.copy()

            with Pool() as pool:
                results = []
                for i in range(self.policy.N):
                    __weights = {
                        "goal_speed": round(float(z[i][0]), 2),
                        "tracking": round(float(z[i][1]), 2),
                        "orientation": round(float(z[i][2]), 2),
                        "acceleration": round(float(z[i][3]), 2),
                    }
                    results.append(
                        pool.apply_async(
                            self.MPC, args=(initial_state, __weights)
                        )
                    )

                for i in range(self.policy.N):
                    sampled_trajectory, _, _, _ = results[i].get()
                    rewards[i] = self.reward(sampled_trajectory)

            weights = self.policy.expectation(rewards, beta)

            self.policy.update(weights, z)

            if track_history:
                history['means'].append(self.policy.mean.copy())
                history['stds'].append(np.sqrt(np.diag(self.policy.covariance)).copy())
                history['rewards_mean'].append(np.mean(rewards))
                history['rewards_max'].append(np.max(rewards))
                history['rewards_min'].append(np.min(rewards))

        if track_history:
            return self.policy.mean, history
        return self.policy.mean
        

