# MPC-Tune
---

## Introduction
Weight tuning for MPC cost functions using Monte-Carlo Expectation Maximization (MCEM)
This project fine-tunes the cost-function weights of a **Model Predictive Controller (MPC)** using **Expectation Maximisation (EM)**-based probabilistic policy search.

The core idea is straightforward:

1. A baseline MPC rollout is performed with hand-tuned weights, producing a **reference trajectory**.
2. A Gaussian policy over the weight space is initialised and iteratively updated.
3. At each iteration, `N` weight vectors are sampled from the policy, the MPC is rolled out for each, and a reward is computed as the negative trajectory distance from the reference.
4. The EM update step re-fits the Gaussian to the best-performing samples, converging the policy mean towards weights that best **imitate** the reference behaviour.


---

## Project Structure

```
MPC-Tune/
├── src/
│   ├── main.py                              # Entry point — runs all experiments
│   └── immitation_gen/
│       ├── policy/
│       │   └── high_mpc.py                  # Policy, BasePolicySearch (ABC), HighMPC
│       ├── mpc/
│       │   └── traffic/
│       │       ├── dynamics.py              # Vehicle dynamics model
│       │       ├── mpc.py                   # MPC solver
│       │       ├── mpc_config.py            # MPC parameters and track definition
│       │       └── simulate.py              # MPC rollout entry point
│       └── utils/
│           └── plotting.py                  # All plotting functions
├── requirement.txt                          # Third-party dependencies
└── pyproject.toml                           # Package build configuration and editable install
```

### Key module: `policy/high_mpc.py`

The policy search logic is built around an abstract base class that makes it easy to extend to new domains:

```
BasePolicySearch  (ABC)
│
│  @abstractmethod  reward(sampled_trajectory) -> float
│  @abstractmethod  policy_search(initial_state, max_iter, beta, **kwargs)
│
└── HighMPC
        Implements reward as negative mean Euclidean distance to a target trajectory.
        Runs MPC rollouts in parallel using multiprocessing.Pool.
```

**To implement a new domain**, subclass `BasePolicySearch` and provide your own `reward` and `policy_search`:

```python
from immitation_gen.policy.high_mpc import BasePolicySearch
import numpy as np

class MyPolicySearch(BasePolicySearch):

    def reward(self, sampled_trajectory: np.ndarray) -> float:
        # define your own reward signal
        ...

    def policy_search(self, initial_state, max_iter, beta, **kwargs):
        # implement your own search loop using self.policy
        ...
```

The `self.policy` instance (a `Policy` object) provides `sample()`, `update()`, and `expectation()` — so you only need to define the reward and loop logic.

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/ARC-ITU/MPC-Tune
cd MPC-Tune
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate
```

> On Windows use: `venv\Scripts\activate`

**3. Install dependencies**

```bash
pip install -r requirement.txt
```

**4. Install the package in editable mode**

```bash
pip install -e .
```

This registers `immitation_gen` as a local package so all imports resolve correctly without modifying `PYTHONPATH`.

**5. Run experiments**

```bash
python src/main.py
```

---

## Output

Results are saved under `data/<timestamp>/`:

```
data/<timestamp>/
├── initial_weight_distribution.png
├── experiment_1_trajectory.png
├── experiment_1_policy_evolution.png
├── experiment_1_rewards.png
├── all_experiments_summary.png
└── progress_over_iterations/
    ├── experiment_1/
    │   ├── iter_0_sampled_trajectories.png
    │   ├── iter_2_sampled_trajectories.png
    │   └── iter_4_sampled_trajectories.png
    └── experiment_2/
        └── ...
```

The `progress_over_iterations/` folder shows all sampled weight trajectories at key iterations, giving a visual view of how the policy distribution evolves during training.



**Authors:** AsimMasood99 · Mubashir22009 · Tayyab-ur-Rehman

