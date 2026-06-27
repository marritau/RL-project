# Temporal Vendi Score for RL Trajectory Diversity

This project studies how to evaluate reinforcement learning agents not only by reward, win rate, or episode length, but also by the **diversity of trajectories** they generate.

The main research question is:

> Do different RL methods solve the same task through genuinely different behavioral strategies, or do they mostly repeat the same trajectory pattern?

To answer this, the project implements and evaluates the **Temporal Vendi Score (TVS)**, a trajectory-diversity metric proposed in:

**Beyond Reward Maximization: Evaluating the Diversity of Trajectories in Reinforcement Learning with Temporal Vendi Score**
OpenReview: https://openreview.net/forum?id=7qGCADaXjr
PDF: https://openreview.net/pdf?id=7qGCADaXjr

Pacman is used as a controlled grid-world benchmark environment. It provides a compact but non-trivial setting with walls, food, ghosts, stochastic dynamics, and multiple possible routes. The goal of the project is not only to make Pacman agents achieve high reward, but to compare how different RL and planning methods behave under both standard performance metrics and trajectory-diversity metrics.

---

## Project overview

The project compares several policy families:

* heuristic baseline policy;
* random policy;
* tabular Q-learning;
* SARSA;
* empirical Value Iteration;
* empirical Policy Iteration.

For each policy, we evaluate:

1. standard RL performance:

   * win rate;
   * mean return;
   * mean score;
   * mean episode length;

2. trajectory diversity:

   * Temporal Vendi Score;
   * trajectory similarity matrix;
   * TVS convergence;
   * occupancy heatmap;
   * trajectory overlay;
   * coverage and occupancy entropy comparison.

The key point is that **high reward and high diversity are not the same thing**. A random policy can have very diverse trajectories while failing the task completely. A strong planner can win more often but follow fewer behavioral modes. TVS helps make this trade-off visible.

---

## Methodological basis

The project follows the TVS evaluation idea from the paper:

1. sample trajectories from a trained policy;
2. compute pairwise temporal distances between states;
3. compute trajectory similarity with a banded Global Alignment Kernel;
4. normalize the trajectory similarity matrix;
5. compute q=2 Vendi Score from the eigenvalues of the normalized similarity matrix.

In this project, a trajectory is represented as an ordered sequence of Pacman grid positions:

```text
tau = [s_0, s_1, s_2, ..., s_T]
```

where each state projection is Pacman's position on the layout grid.

This intentionally focuses TVS on **route diversity**: how differently agents move through the maze structure. Full Pacman state diversity, including food map, ghost timers, score, and capsules, would require a much larger state-distance model and is outside the scope of this implementation.

---

## Temporal Vendi Score implementation

The implementation uses the following components.

### 1. Trajectory sampling

For a policy `pi`, the evaluator runs multiple episodes and stores the ordered sequence of Pacman positions:

```text
trajectory = [(x_0, y_0), (x_1, y_1), ..., (x_T, y_T)]
```

The evaluator also records:

* total return;
* game score;
* episode length;
* win/loss flag;
* whether the trajectory was selected for scoring.

By default, for meaningful policies the project scores successful trajectories when enough wins are available. This keeps the interpretation close to:

> diversity of useful or successful behavior.

For random policy, all trajectories are scored because the policy usually does not win.

---

### 2. Time-to-reach distance

The local state distance is the shortest path distance through the Pacman maze.

For two grid cells `s` and `s'`, the time-to-reach distance is:

```text
d(s, s') = minimum number of legal moves required to reach s' from s
```

The implementation computes this exactly with BFS over walkable cells:

* walls are excluded;
* each move has unit cost;
* legal moves are North, South, East, West;
* unreachable pairs receive a large finite fallback distance.

This gives a distance that respects the environment geometry. Two cells may be close in Euclidean coordinates but far in practice if a wall separates them.

---

### 3. Local similarity kernel

For two states, the local similarity is:

```text
kappa(s, s') = exp(-d(s, s') / sigma)
```

where `sigma` controls the bandwidth of the similarity kernel.

The project calibrates `sigma` once per environment using the paper-style rule:

```text
sigma = median_trajectory_length * median_time_to_reach / ln(2)
```

For cross-method comparison, the same calibrated `sigma` is reused for all policies. This is important because TVS values should be comparable across agents evaluated in the same environment.

---

### 4. Global Alignment Kernel

Trajectories are sequences, so the project compares them using a Global Alignment Kernel.

For two trajectories:

```text
tau  = [s_1, ..., s_T]
tau' = [s'_1, ..., s'_{T'}]
```

the kernel marginalizes over monotonic temporal alignments.

The dynamic-programming recursion is:

```text
GA(i, j) = kappa(s_i, s'_j) * (GA(i-1, j-1) + GA(i-1, j) + GA(i, j-1))
```

To reduce computation and follow the paper setup, the implementation uses a 20% Sakoe-Chiba band:

```text
|i - j| <= 0.2 * max(T, T')
```

This keeps trajectory comparisons focused on approximately aligned temporal regions.

---

### 5. Kernel normalization

The raw Global Alignment Kernel does not automatically satisfy:

```text
K(tau, tau) = 1
```

which is required for Vendi Score.

Therefore, the pairwise trajectory kernel is normalized as:

```text
K_ij = GA(tau_i, tau_j) / sqrt(GA(tau_i, tau_i) * GA(tau_j, tau_j))
```

After normalization:

```text
K_ii = 1
```

for all trajectories.

---

### 6. q=2 Vendi Score

The final Temporal Vendi Score is computed from the eigenvalues of the normalized similarity matrix.

The project uses q=2:

```text
TVS = 1 / sum(lambda_i^2)
```

where `lambda_i` are the eigenvalues of `K / N`.

This version of the Vendi Score estimates the effective number of dominant behavioral modes and is less sensitive to small superficial variations.

Small negative eigenvalues caused by numerical or kernel PSD issues are clamped to zero before computing the score.

---

## Environment

The project uses a Pacman grid-world environment with a Gymnasium-style API.

Main task:

> control Pacman to collect all food while avoiding ghosts.

The environment is useful for trajectory-diversity analysis because:

* the map contains walls and corridors;
* several routes can lead to successful outcomes;
* ghost movement makes the dynamics stochastic;
* different policies can produce visibly different routes;
* the grid structure allows exact shortest-path distances with BFS.

---

## Observation formats

Observation representation is configurable.

Supported observation modes include:

| Observation name           | Description                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------- |
| `raw`                      | Full observation dictionary with walls, food, capsules, ghosts, score, and step count |
| `chunked_food`             | Chunk-level food representation plus local maps                                       |
| `food_bitmask`             | Integer bitmask over walkable food cells                                              |
| `bitmask_distance_buckets` | Food bitmask plus bucketized nearest-food and nearest-ghost features                  |

The TVS metric itself uses Pacman's grid position as the state projection for trajectory comparison.

---

## Action space

The environment uses five discrete actions:

| ID | Action |
| -: | ------ |
|  0 | North  |
|  1 | South  |
|  2 | East   |
|  3 | West   |
|  4 | Stop   |

---

## Reward function

Default reward configuration:

| Event             | Reward |
| ----------------- | -----: |
| Time step penalty |   -1.0 |
| Food eaten        |  +10.0 |
| Capsule eaten     |   +0.0 |
| Ghost eaten       | +200.0 |
| Win               | +500.0 |
| Lose              | -500.0 |
| Invalid action    |   -5.0 |

---

## Implemented policies and algorithms

### Random policy

The random policy samples uniformly from legal actions.

It is included as a diversity sanity check. Random behavior often produces high trajectory diversity but poor task performance.

---

### Heuristic baseline

The baseline is a non-learning policy.

It follows simple rules:

1. if a ghost is nearby, move away from the closest dangerous ghost;
2. otherwise, move toward the nearest food using BFS;
3. avoid illegal moves;
4. avoid Stop unless no better action exists.

This policy is useful as a deterministic-ish reference point with moderate task performance and limited route diversity.

---

### Q-learning

Q-learning learns an action-value function using the off-policy Bellman update:

```text
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

The implementation uses tabular Q-values over compact encoded observations.

Important details:

* actions are restricted to legal actions;
* exploration uses epsilon-greedy action selection;
* unseen states during evaluation use a heuristic fallback action;
* model is saved as `q_table.pkl`.

Recommended config:

```text
configs/q_learning.yaml
```

---

### SARSA

SARSA learns an action-value function using the on-policy update:

```text
Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
```

The key difference from Q-learning is that SARSA bootstraps from the **actually selected next action** `a'`, not from the maximum action.

Important details:

* actions are restricted to legal actions;
* exploration uses epsilon-greedy action selection;
* epsilon decays during training;
* unseen states during evaluation use a heuristic fallback action;
* model is saved as `q_table.pkl`.

Recommended config:

```text
configs/sarsa.yaml
```

---

### Value Iteration

The Value Iteration agent builds an empirical MDP from collected transitions and then performs Bellman optimality updates.

Empirical transition model:

```text
P_hat(s' | s, a) = N(s, a, s') / N(s, a)
```

Empirical reward model:

```text
R_hat(s, a, s') = average observed reward for transition (s, a, s')
```

Value update:

```text
Q_k(s, a) = sum_s' P_hat(s' | s, a) * [R_hat(s, a, s') + gamma * V_{k-1}(s')]
V_k(s) = max_a Q_k(s, a)
```

The resulting policy chooses:

```text
pi(s) = argmax_a Q(s, a)
```

Recommended config:

```text
configs/bitmask_value_iteration.yaml
```

---

### Policy Iteration

Policy Iteration also uses an empirical MDP.

It alternates between:

1. policy evaluation;
2. policy improvement.

Policy evaluation:

```text
V(s) <- sum_s' P_hat(s' | s, pi(s)) * [R_hat(s, pi(s), s') + gamma * V(s')]
```

Policy improvement:

```text
pi_new(s) = argmax_a sum_s' P_hat(s' | s, a) * [R_hat(s, a, s') + gamma * V(s')]
```

Recommended config:

```text
configs/policy_iteration_obs.yaml
```

---

## Current experimental results

The following results are from completed runs and are used in the final project analysis.

### Standard RL evaluation

| Method           | Eval episodes | Win rate | Mean return |
| ---------------- | ------------: | -------: | ----------: |
| Value Iteration  |           200 |    0.710 |     208.665 |
| Q-learning       |           200 |    0.630 |      49.660 |
| Policy Iteration |           200 |    0.620 |     139.350 |
| SARSA            |           200 |    0.575 |      -8.895 |

Interpretation:

* Value Iteration achieves the strongest task performance.
* Policy Iteration and Q-learning are competitive but less stable.
* SARSA wins in more than half of evaluation episodes but has lower mean return.
* Standard performance metrics alone do not show how diverse the agents' successful routes are.

---

### Temporal Vendi Score evaluation

TVS was computed for all supported policy families.

| Policy           | Episodes | Scored trajectories | Win rate | Mean return |    TVS |
| ---------------- | -------: | ------------------: | -------: | ----------: | -----: |
| Random           |      256 |                 256 |    0.000 |    -526.540 | 13.910 |
| SARSA            |      256 |                 152 |    0.594 |      12.450 | 11.930 |
| Q-learning       |      256 |                 156 |    0.609 |      26.810 | 11.080 |
| Value Iteration  |      256 |                 185 |    0.723 |     220.270 |  5.270 |
| Policy Iteration |      256 |                 140 |    0.547 |      61.790 |  4.740 |
| Baseline         |      256 |                  69 |    0.270 |    -267.340 |  2.260 |

Interpretation:

* Random policy has the highest TVS but zero win rate. This shows that diversity alone is not enough.
* SARSA and Q-learning produce more diverse successful trajectories than the planning methods.
* Value Iteration has the best reward and win rate, but lower TVS. It tends to follow more stable high-value routes.
* Policy Iteration has moderate performance and moderate diversity.
* Baseline has limited diversity because it follows a fixed heuristic route-selection pattern.

The most important conclusion is:

> TVS should be interpreted together with reward and win rate. A good agent should not only be diverse, but should produce diverse high-quality trajectories.

---

## Main output files

After running the complete TVS evaluation, the most important files are:

```text
results/tvs_all/tvs_summary.csv
results/tvs_all/tvs_summary.json
results/tvs_all/tvs_summary.png
results/tvs_all/quality_diversity_scatter.png
results/tvs_all/return_diversity_scatter.png
results/tvs_all/coverage_entropy_tvs_comparison.png
```

For each policy, detailed outputs are stored in:

```text
results/tvs_all/<policy>/
```

Typical per-policy files:

```text
tvs_metrics.json
similarity_matrix.npy
similarity_matrix.png
occupancy_heatmap.png
trajectory_overlay.png
tvs_convergence.json
tvs_convergence.png
trajectories.json
```

---

## Visualizations

The project generates several visualizations for analysis and presentation.

### 1. TVS summary

```text
results/tvs_all/tvs_summary.png
```

Compares policies by Temporal Vendi Score.

---

### 2. Quality-diversity scatter

```text
results/tvs_all/quality_diversity_scatter.png
```

Shows the relationship between win rate and TVS.

This is one of the most useful plots for the final report because it shows that:

* Random is diverse but ineffective;
* Value Iteration is effective but less diverse;
* Q-learning and SARSA are more diverse among successful policies.

---

### 3. Return-diversity scatter

```text
results/tvs_all/return_diversity_scatter.png
```

Shows the relationship between mean return and TVS.

---

### 4. Coverage / entropy / TVS comparison

```text
results/tvs_all/coverage_entropy_tvs_comparison.png
```

Compares TVS against simpler diversity proxies such as state coverage and occupancy entropy.

This is useful because simple state-level metrics can miss temporal differences between trajectories.

---

### 5. Trajectory overlay

```text
results/tvs_all/<policy>/trajectory_overlay.png
```

Shows actual route shapes used by a policy.

This is especially useful for presentation because it makes trajectory diversity visually interpretable.

---

### 6. Occupancy heatmap

```text
results/tvs_all/<policy>/occupancy_heatmap.png
```

Shows how frequently the policy visits each grid cell.

---

### 7. Similarity matrix

```text
results/tvs_all/<policy>/similarity_matrix.png
```

Shows pairwise trajectory similarity. Block structure in this matrix can indicate clusters of similar behavioral modes.

---

### 8. TVS convergence

```text
results/tvs_all/<policy>/tvs_convergence.png
```

Shows how TVS changes as the number of sampled trajectories increases.

---

### 9. GIF rollouts

GIFs provide qualitative examples of policy behavior.

They are not the main TVS evidence, but they are useful for visual comparison in presentation.

Output directory:

```text
results/important/
```

---

## Installation

### Windows PowerShell

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e .
```

Optional test run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_tmp
```

If pytest fails because of Windows temporary-folder permissions, this does not necessarily indicate a project-code error. The training and evaluation scripts can still be run directly.

---

## Training commands

Run commands from the project root.

### Q-learning

```powershell
.\.venv\Scripts\python.exe scripts\train.py q_learning --config configs\q_learning.yaml --episodes 5000
```

### SARSA

```powershell
.\.venv\Scripts\python.exe scripts\train.py sarsa --config configs\sarsa.yaml --episodes 5000
```

### Value Iteration

```powershell
.\.venv\Scripts\python.exe scripts\train.py vi --config configs\bitmask_value_iteration.yaml --collection-episodes 1000 --max-iterations 300
```

### Policy Iteration

```powershell
.\.venv\Scripts\python.exe scripts\train.py pi --config configs\policy_iteration_obs.yaml --episodes 5000
```

---

## Evaluation commands

### Q-learning

```powershell
.\.venv\Scripts\python.exe scripts\eval.py q_learning --config configs\q_learning.yaml --episodes 200 --no-gif --output-dir results\eval_q_learning
```

### SARSA

```powershell
.\.venv\Scripts\python.exe scripts\eval.py sarsa --config configs\sarsa.yaml --episodes 200 --no-gif --output-dir results\eval_sarsa
```

### Value Iteration

```powershell
.\.venv\Scripts\python.exe scripts\eval.py vi --config configs\bitmask_value_iteration.yaml --episodes 200 --no-gif --output-dir results\eval_vi
```

### Policy Iteration

```powershell
.\.venv\Scripts\python.exe scripts\eval.py pi --config configs\policy_iteration_obs.yaml --episodes 200 --no-gif --output-dir results\eval_pi
```

---

## TVS evaluation commands

### Evaluate TVS for all policies

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_all_tvs.py --episodes 256 --seed 42 --output-dir results\tvs_all --save-trajectories
```

This creates the main comparison table and visualizations.

Main outputs:

```text
results/tvs_all/tvs_summary.csv
results/tvs_all/tvs_summary.json
results/tvs_all/tvs_summary.png
results/tvs_all/quality_diversity_scatter.png
results/tvs_all/return_diversity_scatter.png
results/tvs_all/coverage_entropy_tvs_comparison.png
```

---

### Evaluate TVS for one policy

Example for Value Iteration:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tvs.py --policy vi --episodes 256 --seed 42 --output-dir results\tvs_vi --save-trajectories
```

Example for Q-learning:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tvs.py --policy q_learning --episodes 256 --seed 42 --output-dir results\tvs_q_learning --save-trajectories
```

Example for SARSA:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tvs.py --policy sarsa --episodes 256 --seed 42 --output-dir results\tvs_sarsa --save-trajectories
```

Example for Policy Iteration:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tvs.py --policy pi --episodes 256 --seed 42 --output-dir results\tvs_pi --save-trajectories
```

---

### Baseline with additional randomization

This run is useful to show the reward-diversity trade-off:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tvs.py `
  --policy baseline `
  --epsilon-random 0.10 `
  --episodes 256 `
  --seed 42 `
  --output-dir results\tvs_baseline_eps010 `
  --save-trajectories
```

---

## Optional robustness check

The project includes a small robustness check for the TVS similarity matrix.

It evaluates whether adding diagonal jitter to the similarity matrix changes the score significantly.

Example for Value Iteration:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_tvs_robustness.py --matrix results\tvs_all\vi\similarity_matrix.npy --output-dir results\tvs_all\vi
```

Outputs:

```text
results/tvs_all/vi/tvs_jitter_robustness.json
results/tvs_all/vi/tvs_jitter_robustness.png
```

This is optional but useful as an appendix-style validation.

---

## GIF export

For qualitative visual comparison, use `scripts/play.py`.

GIFs are saved to:

```text
results/important/
```

### Q-learning GIF

```powershell
.\.venv\Scripts\python.exe scripts\play.py q_learning `
  --config configs\q_learning.yaml `
  --model results\train_q_learning\q_table.pkl `
  --episodes 1 `
  --seed 42 `
  --gif-title q_learning_seed42
```

### SARSA GIF

```powershell
.\.venv\Scripts\python.exe scripts\play.py sarsa `
  --config configs\sarsa.yaml `
  --model results\train_sarsa\q_table.pkl `
  --episodes 1 `
  --seed 42 `
  --gif-title sarsa_seed42
```

### Policy Iteration GIF

```powershell
.\.venv\Scripts\python.exe scripts\play.py pi `
  --config configs\policy_iteration_obs.yaml `
  --model results\obs_policy_iteration\policy.pkl `
  --episodes 1 `
  --seed 42 `
  --gif-title pi_seed42
```

### Value Iteration GIF

```powershell
.\.venv\Scripts\python.exe scripts\play.py vi `
  --config configs\bitmask_value_iteration.yaml `
  --model results\train_food_bitmask_vi\model.pkl `
  --episodes 1 `
  --seed 42 `
  --gif-title vi_seed42
```

---

## Recommended final run order

For a complete reproducible experiment:

```powershell
.\.venv\Scripts\python.exe scripts\train.py q_learning --config configs\q_learning.yaml --episodes 5000
.\.venv\Scripts\python.exe scripts\train.py sarsa --config configs\sarsa.yaml --episodes 5000
.\.venv\Scripts\python.exe scripts\train.py vi --config configs\bitmask_value_iteration.yaml --collection-episodes 1000 --max-iterations 300
.\.venv\Scripts\python.exe scripts\train.py pi --config configs\policy_iteration_obs.yaml --episodes 5000
```

Then run standard evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\eval.py q_learning --config configs\q_learning.yaml --episodes 200 --no-gif --output-dir results\eval_q_learning
.\.venv\Scripts\python.exe scripts\eval.py sarsa --config configs\sarsa.yaml --episodes 200 --no-gif --output-dir results\eval_sarsa
.\.venv\Scripts\python.exe scripts\eval.py vi --config configs\bitmask_value_iteration.yaml --episodes 200 --no-gif --output-dir results\eval_vi
.\.venv\Scripts\python.exe scripts\eval.py pi --config configs\policy_iteration_obs.yaml --episodes 200 --no-gif --output-dir results\eval_pi
```

Then run TVS evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_all_tvs.py --episodes 256 --seed 42 --output-dir results\tvs_all --save-trajectories
```

---

## Project structure

```text
configs/
  q_learning.yaml
  sarsa.yaml
  bitmask_value_iteration.yaml
  policy_iteration_obs.yaml
  tvs_eval.yaml

scripts/
  train.py
  eval.py
  play.py
  evaluate_tvs.py
  evaluate_all_tvs.py
  analyze_tvs_robustness.py

src/pacman_rldp/
  env/
  agents/
  algorithms/
  diversity/
  third_party/

results/
  eval_q_learning/
  eval_sarsa/
  eval_vi/
  eval_pi/
  tvs_all/
  important/

tests/
```

---

## Notes on interpretation

TVS is not a replacement for reward. It measures a different property.

A policy can have:

* high reward and low diversity;
* low reward and high diversity;
* moderate reward and high diversity;
* high reward and high diversity.

Therefore, final conclusions should use both standard RL metrics and TVS.

In the current experiments:

* Value Iteration is the best method by win rate and mean return.
* SARSA and Q-learning show higher diversity among scored trajectories.
* Random policy shows that high diversity without task success is not sufficient.
* Baseline demonstrates limited diversity caused by rule-based behavior.

This supports the main conclusion of the project:

> Evaluating RL agents only by reward hides important differences in behavior. Temporal Vendi Score makes it possible to compare not only how well agents solve the task, but also how many distinct trajectory strategies they use.