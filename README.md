# Pacman RLDP

Pacman RLDP is a team project workspace for dynamic programming (DP) and reinforcement learning experiments on a Pacman runtime.

## Quick Start
Setup:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```
### Examples
Run train / eval / visualize:
```bash
# Policy Iteration (empirical MDP on observations)
python scripts/train.py pi --config configs/policy_iteration_obs.yaml
python scripts/eval.py  pi --config configs/policy_iteration_obs.yaml
python scripts/play.py  pi --render-mode human

# Value Iteration (food-bitmask)
python scripts/train.py vi --config configs/bitmask_value_iteration.yaml
python scripts/eval.py  vi --config configs/bitmask_value_iteration.yaml
python scripts/play.py  vi --render-mode human

# Q-learning / SARSA
python scripts/train.py q_learning --config configs/default.yaml --episodes 1000
python scripts/train.py sarsa      --config configs/default.yaml --episodes 1000
python scripts/play.py  q_learning --render-mode human
```

Manual play:
```bash
python scripts/play.py manual --config configs/default.yaml --render-mode human
python scripts/play.py manual --config configs/default.yaml --render-mode ansi
```

## Problem Definition
- Task: control Pacman to collect all food while avoiding ghosts.
- Agent objective: maximize cumulative episodic return.
- Environment dynamics: turn-based sequence where Pacman acts first, then each ghost acts.
- Transition type: **stochastic by default** because ghost actions are sampled uniformly from legal actions.

## Environment Specification
Primary API: `pacman_rldp.env.PacmanEnv` (Gymnasium-style).

### State Representation (`ObsDict`)
`reset()` and `step()` return a structured observation dictionary.
Default (`env.observation.name: raw`) fields:
- `pacman_position`: `(2,)` float32
- `ghost_positions`: `(num_ghosts, 2)` float32 (`-1` padding for absent slots)
- `ghost_timers`: `(num_ghosts,)` int32
- `ghost_present`: `(num_ghosts,)` binary
- `walls`: `(width, height)` int8 binary
- `food`: `(width, height)` int8 binary
- `capsules`: `(width, height)` int8 binary
- `score`: `(1,)` float32
- `step_count`: `(1,)` int32

### Observation Registry
Observation format is configurable through `env.observation.name`:
- `raw` (default): full grids (`walls/food/capsules`) + agent features.
- `chunked_food`: chunk-level binary food map + full local maps for Pacman's current chunk.
- `food_bitmask`: one integer food bitmask over walkable (non-wall) cells.
- `bitmask_distance_buckets`: `food_bitmask` + bucketized nearest-food/nearest-ghost distances and coarse directions.

Config block (in `configs/default.yaml`):
```yaml
env:
  observation:
    name: raw
    chunk_w: 4
    chunk_h: 2
    distance_bucket_size: 2
```

Bitmask encoding rule (deterministic):
- only non-wall cells are encoded,
- canonical order is row-major by map rows top-to-bottom, then left-to-right within each row.


### Action Space
- `action_space = Discrete(5)`
- Mapping:
  - `0 -> North`
  - `1 -> South`
  - `2 -> East`
  - `3 -> West`
  - `4 -> Stop`

### Transition Logic
For one `step(action)`:
1. Pacman action is applied (or rejected according to `invalid_action_mode`).
2. Ghosts respond sequentially.
3. Episode ends on win/loss (`terminated=True`) or max-step limit (`truncated=True`).

### Ghost Strategy (Default)
- Config key: `env.ghost_policy`.
- Current supported default: `random`.
- `random` policy means each ghost samples uniformly from its legal actions at its turn.
- Ghost movement rules are as follows:
  - ghost cannot choose `Stop` (except edge-case when no alternatives),
  - ghost usually avoids immediate reverse direction unless forced by map topology.
- Result: transition dynamics are stochastic and reproducible under fixed seed.

### Ghost Strategy (`markovian`)
- Set `env.ghost_policy: markovian` for uniform one-step Markov transitions.
- At each ghost turn:
  - all legal non-`Stop` neighbor transitions are sampled with equal probability,
  - `Stop` transition has probability `0` (unless no move is available).
- This policy is stochastic and reproducible under fixed seed.

### Ghost Strategy (`loop_path`)
- Set `env.ghost_policy: loop_path` for deterministic ghost patrol.
- Additional config:
  - `ghost_loop_matrix`: `0/1` matrix (row-major, top-to-bottom) with `1` for loop cells.
  - `ghost_loop_direction`: `clockwise` or `anticlockwise`.
- Validation rules:
  - matrix shape must match layout size,
  - values must be only `0` or `1`,
  - all `1` cells must be non-wall cells,
  - `1` cells must form one closed simple cycle (degree 2 at each path node).
- Multi-ghost behavior:
  - all ghosts use the same single loop,
  - initial loop indices are evenly spaced along that loop.
- If a ghost spawn is off-loop, it is snapped to nearest loop node for loop indexing.
- Dynamics are deterministic (independent of RNG) once loop policy is active.

### What The Agent Observes
- The environment is **fully observable** in the provided observation dictionary.
- Pacman directly receives:
  - exact Pacman position,
  - exact ghost positions and scared timers,
  - full wall grid,
  - full food grid,
  - full capsule map,
  - current score and step count.
- No hidden-state belief model is required for baseline DP experiments.
- Auxiliary `info` returned by `step()` includes legal action IDs and episode flags (`is_win`, `is_lose`).

### Episode End Conditions
- **Termination**:
  - Win: all food consumed.
  - Loss: Pacman collides with a non-scared ghost.
- **Truncation**:
  - `step_count >= max_steps`.

### Reward Function (Default)
Configured in `configs/default.yaml` under `env.reward`.

| Event | Reward |
|---|---:|
| Time step penalty | `-1.0` |
| Food eaten | `+10.0` |
| Capsule eaten | `+0.0` |
| Ghost eaten | `+200.0` |
| Win | `+500.0` |
| Lose | `-500.0` |
| Invalid action | `-5.0` |


## Results & Artifacts by Algorithm

### Baseline
**Formula**

$$
d_{\text{ghost}}(s)=\min_g \left\lVert p_{\text{pac}}(s)-p_g(s)\right\rVert_1
$$

$$
a_{\text{escape}}^*(s)=\arg\max_{a\in A_{\text{legal}}}\ \min_g \left\lVert p'_{\text{pac}}(s,a)-p_g(s)\right\rVert_1
$$

$$
a_{\text{food}}^*(s)=\text{first action on BFS shortest path to nearest food}
$$

**GIFs / Visual Rollouts**

Loop-path baseline:

![loop2_baseline](results/important/loop2_baseline.gif)

Markovian baseline:

![markov1_baseline](results/important/markov1_baseline.gif)

Random-policy baseline:

![game1_baseline](results/important/game1_baseline.gif)

**Metrics**

Source: `results/eval/eval_metrics.json` (baseline policy)
- Episodes: `200` (seed schedule: `42 + i`)
- Win rate: `0.575`
- Average reward: `21.535`
- Average episode length: `69.215`
- Data representation: **Non-tabular** (rule-based heuristic)

### Q-learning

Here we have two algorithms -- /scripts/q_obs_learning_agent_copy.py (actor-critic, approximate Q-learning) and q_table_learning_agent.py (classic tabular Q-learning).

As large amount of features mentioned above haven't helped with the learning of both these agents (in case of approximate it was way too noisy, for tabular it exploded gradients, after using clipping it was still noisy and way too inefficient), I've decided to use more informative features for both variants.

**Features**
- hit-wall — Whether Pacman’s next move hits a wall (0 or 1)
- scared — At least one ghost is in scared mode (0 or 1)
- eats-food — Pacman eats food on the next move (0 or 1)
- closest-food — Distance to the nearest food (integer or -1 if unreachable)
- food-nearby — Number of food pellets in a 3x3 area around the next position (integer)
- ghost-distance — Distance to the nearest ghost (integer or -1 if none)
- danger — A ghost is within distance ≤1 (0 or 1)
- capsule-distance — Distance to the nearest capsule (integer or -1 if none)
- stop — Action is Stop (0 or 1)


**Policy**

1) Tabular Q-learning Agent:

Stores Q-values for each unique combination of feature values and action.

Policy: For a given state, computes feature values for all legal actions, looks up Q-values in the table, and selects the action with the highest Q-value (breaks ties randomly).

Exploration: With probability epsilon, chooses a random legal action.

2) Approximate Q-learning Agent:

Computes Q-value as a weighted sum of features: $Q(s, a) = w^\top f(s, a)$.

Policy: For a given state, extracts features for all legal actions, computes Q-values using current weights, and selects the action with the highest Q-value (breaks ties randomly).

Exploration: With probability epsilon, chooses a random legal action.


**Formula**

$$
Q(s,a;\mathbf{w})=\mathbf{w}^{\top}\mathbf{f}(s,a)
$$

$$
y=
\begin{cases}
r, & \text{terminal}\\
r+\gamma\max_{a'}Q(s',a';\mathbf{w}), & \text{otherwise}
\end{cases}
$$

$$
\delta = y - Q(s,a;\mathbf{w}),\qquad
w_i \leftarrow w_i + \alpha\,\delta\,f_i(s,a)
$$

**Quickstart**

Use `--render` flag only if you want to see the agent in action.

1) Tabular:

- Training (it is advised to just use q_table.pkl)

`python scripts/q_table_learning_agent.py --train --episodes 10000 --config configs/default.yaml`

- Evaluation 
`python scripts/q_table_learning_agent.py --eval --model q_table.pkl --episodes 200 --config configs/default.yaml --render`

2) Approximate Q-learning

- Training
`python scripts/q_obs_learning_agent_copy.py --train --episodes 10000 --config configs/default.yaml`

- Evaluation
`python scripts/q_obs_learning_agent_copy.py --eval --model q_obs_weights_copy.pkl --episodes 200 --config configs/default.yaml --render`

## Tabular Q-learning metrics

**Training**
- Collected episodes: `60000`
- Unique feature-action pairs (Q-table size): `17836`

**Evaluation** 
- Episodes: `200` (base seed `42`)
- Win rate: `0.59`
- Mean total reward: `28.3`
- Mean score: `410.2`
- Data representation: **Tabular (feature-based)**
![CS188 Pacman 2026-03-16 19-11-30](https://github.com/user-attachments/assets/3e886495-a470-42b3-aa04-37315f94c77e)
---

**Approximate Q-learning metrics**

**Training**
- Collected episodes: `10000`
- Feature weights: `9` (linear function approximation)
- Collection mean total reward: `-110.7`

**Evaluation** 
- Episodes: `200` (base seed `42`)
- Win rate: `0.72`
- Mean total reward: `45.1`
- Mean score: `480.8`
- Data representation: **Linear function approximation**
![CS188 Pacman 2026-03-16 05-41-06 (online-video-cutter com) (1)](https://github.com/user-attachments/assets/7fa68473-97e4-4adb-8ed2-3da808f4dd68)

### Value Iteration (food-bitmask empirical VI)
**Formula**

$$
\hat{P}(s'|s,a)=\frac{N(s,a,s')}{N(s,a)}
$$

$$
\hat{R}(s,a,s')=\frac{\sum \text{rewards}(s,a,s')}{N(s,a,s')}
$$

$$
Q_k(s,a)=\sum_{s'}\hat{P}(s'|s,a)\left[\hat{R}(s,a,s')+\gamma V_{k-1}(s')\right]
$$

$$
V_k(s)=\max_a Q_k(s,a),\qquad
\pi(s)=\arg\max_a Q_k(s,a)
$$

$$
\text{residual}_k=\max_s\left|V_k(s)-V_{k-1}(s)\right|
$$

**GIFs / Curves**

Fast/high-return VI rollout:

![fast_high_return_VI](results/important/fast_high_return_VI.gif)

Best overall VI rollout:

![best_overall_VI](results/important/best_overall_VI.gif)

VI training reward curve:

![train_bitmask_vi_reward_curve](results/important/train_bitmask_vi_reward_curve.png)

**Metrics**

Training source: `results/important/train_bitmask_vi_metrics.json`
- Collected episodes: `2500`
- Discovered states: `136415`
- Transition samples: `217329`
- VI iterations: `750`
- Final Bellman residual: `0.0706064815`
- Collection mean return: `-183.1356`

Evaluation source: `results/important/eval_bitmask_vi_metrics.json`
- Episodes: `200` (base seed `42`)
- Win rate: `0.6`
- Mean return: `31.715`
- Mean score: `475.915`
- Mean steps: `138.785`
- Best episode return: `719.0` (seed `83`)
- Data representation: **Tabular** over aggregated food-bitmask observation states.

### SARSA 
## Algorithm

The canonical update rule:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot Q(s', a') - Q(s, a) \right]$$

where:
- $s, a$ — current state and action
- $r$ — received reward (including reward shaping)
- $s', a'$ — next state and the **actually selected** next action
- $\alpha$ — learning rate
- $\gamma$ — discount factor

At episode termination (`terminated=True`), $Q(s', a') = 0$ is assumed.

### Agent Parameters

| Parameter | Value | Description |
|---|---|---|
| `alpha` | `0.1` | Learning rate. Controls how strongly new information overwrites old estimates. |
| `gamma` | `0.99` | Discount factor. Close to 1 means the agent values long-term reward highly. |
| `epsilon` | `1.0` | Initial probability of a random action (exploration). Starts at full randomness. |
| `action_size` | `5` | Number of available actions in the environment. |

### Training Parameters

| Parameter | Value | Description |
|---|---|---|
| `episodes` | `3000` | Number of training episodes. |
| `epsilon_decay` | `0.9995` | Exponential decay multiplier applied to epsilon after each episode. |
| `epsilon_end` | `0.05` | Lower bound for epsilon. The agent always retains a minimum level of exploration. |

## Action Selection Strategy

The agent uses an **ε-greedy** strategy with a legal action mask:

1. With probability `epsilon` — a random action is chosen from the set of legal actions (`legal_action_ids`).
2. Otherwise — the action with the highest Q-value among legal actions is chosen; ties are broken randomly.

Epsilon decay is applied after each episode:

```
epsilon = max(epsilon_end, epsilon * epsilon_decay)
```

**GIFs / Curves**

- Qualitative artifact (gif/mp4): `TBD`
- Training curve: `TBD`

**Metrics**

- Metrics summary: `TBD`
- Data representation: **Tabular** (`q_table` dictionary).

### Policy Iteration (Empirical Observation MDP)
We build an empirical MDP over aggregated observations and run classic policy iteration.

**Empirical model**

$$
\hat{P}(s'\mid s,a)=\frac{N(s,a,s')}{N(s,a)},\qquad
\hat{R}(s,a,s')=\frac{1}{N(s,a,s')}\sum_{i=1}^{N(s,a,s')} r_i
$$

**Policy evaluation**

$$
V(s)\leftarrow \sum_{s'} \hat{P}(s'\mid s,\pi(s))\,[\hat{R}(s,\pi(s),s')+\gamma V(s')]
\quad \text{until} \quad
\max_s |V_{\text{new}}(s)-V_{\text{old}}(s)|<\theta
$$

**Policy improvement**

$$
\pi_{\text{new}}(s)=\arg\max_a \sum_{s'} \hat{P}(s'\mid s,a)\,[\hat{R}(s,a,s')+\gamma V(s')]
$$

**Results**
- Win rate: **54%**

**Artifacts**

![best_pi](results/important/best_pi.gif)

![policy_iteration_curve](results/important/policy_iteration_curve.png)


## Algorithm and Visualization Review

A second pass checked the original Q-learning, SARSA, Value Iteration, and Policy Iteration implementations in addition to the new TVS metric.  Notes are in `docs/ALGORITHM_AND_VISUAL_CHECK.md`.  The most important fixes are:

- SARSA now uses `configs/sarsa.yaml` by default and applies `epsilon_decay` / `epsilon_end`.
- SARSA was tested to bootstrap from the actually selected next action `a'`, not `max_a Q(s',a)`.
- Q-learning/SARSA saved model paths now match eval/play defaults.
- GIF capture now records only the Pacman Tk canvas.  For `smallClassic` at `zoom: 1.0`, the expected frame is `630 x 275`, not the full `2520 x 1680` desktop.

## GIF Export
`scripts/play.py` saves GIF from human visualization (`--render-mode human`).
Output directory: `results/important/`.
- Custom filename:
```bash
python scripts/play.py pi --render-mode human --gif-title my_run
```
- Disable GIF export:
```bash
python scripts/play.py pi --render-mode human --no-gif
```
- Default filename (if `--gif-title` is omitted): `experiment_{local_time}.gif`
- `--render-mode ansi` does not export GIF.

## DP Scaffold Interfaces
- `pacman_rldp.algorithms.MDPModel`
- `pacman_rldp.algorithms.TransitionOutcome`
- `pacman_rldp.algorithms.PacmanMDPAdapter`

These provide model-first abstractions (state encoding, legal actions, transition outcomes, reward, terminal checks) for teammate DP solver implementations.

## Project Structure
- `src/pacman_rldp/third_party/bk/`: environment core, rendering, keyboard control, layouts.
- `src/pacman_rldp/env/`: Gymnasium environment wrapper.
- `src/pacman_rldp/algorithms/`: DP model interfaces and adapter.
- `src/pacman_rldp/agents/`: baseline random/manual policies.
- `scripts/`: train/eval/play entrypoints.
- `configs/`: reproducible experiment configuration.


## Temporal Vendi Score Extension

This fork adds a project extension for the paper:

**Beyond Reward Maximization: Evaluating the Diversity of Trajectories in Reinforcement Learning with Temporal Vendi Score**  
OpenReview: https://openreview.net/forum?id=7qGCADaXjr  
PDF: https://openreview.net/pdf/4b56e283f49a8cc5a6271a62bc723a09aefd088d.pdf

The new code evaluates not only reward/win rate, but also whether a policy reaches good outcomes through genuinely different trajectories.

Run baseline-policy TVS:

```bash
python scripts/evaluate_tvs.py \
  --config configs/tvs_eval.yaml \
  --policy baseline \
  --episodes 128 \
  --seed 42 \
  --output-dir results/tvs_baseline \
  --save-trajectories
```

Run random-policy TVS:

```bash
python scripts/evaluate_tvs.py \
  --config configs/tvs_eval.yaml \
  --policy random \
  --episodes 128 \
  --seed 42 \
  --quality-filter all \
  --output-dir results/tvs_random
```

Main outputs:

- `tvs_metrics.json`
- `similarity_matrix.npy`
- `similarity_matrix.png`
- `occupancy_heatmap.png`
- `tvs_convergence.json`
- `tvs_convergence.png`

Implementation notes:

- The Pacman state projection for TVS is Pacman's grid position.
- Time-to-reach distance is exact shortest-path distance through layout walls.
- Trajectory similarity uses the paper-style banded Global Alignment Kernel with the 20% Sakoe-Chiba rule `|i-j| <= 0.2 * max(T,T')`.
- The bandwidth `sigma` is calibrated with the paper's Appendix A.1 rule: `sigma = median_length * median_time_to_reach / ln(2)`.
- Consecutive repeated states are kept by default to preserve temporal sequence information; `--compress-repeats` is optional.
- The final diversity score is q=2 Vendi Score, matching the paper's main experiments.
- See `docs/PROJECT_REVIEW_TVS.md` for the full project review, limitations, and next improvements.

## Temporal Vendi Score Full Integration

The project now includes full TVS evaluation for every implemented policy family: `baseline`, `random`, `q_learning`, `sarsa`, `vi`, and `pi`.

Paper: **Beyond Reward Maximization: Evaluating the Diversity of Trajectories in Reinforcement Learning with Temporal Vendi Score**  
OpenReview: https://openreview.net/forum?id=7qGCADaXjr

Run TVS for one trained policy:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tvs.py --policy vi --episodes 256 --seed 42 --output-dir results\tvs_vi --save-trajectories
```

Run TVS for all supported policies and create one comparison table:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_all_tvs.py --episodes 256 --seed 42 --output-dir results\tvs_all --save-trajectories
```

The summary is saved to:

```text
results/tvs_all/tvs_summary.csv
results/tvs_all/tvs_summary.json
results/tvs_all/tvs_summary.png
```

Detailed notes and commands are in `docs/TVS_FULL_INTEGRATION.md`.

### Article-aligned TVS check (latest)

After a detailed check against the TVS paper, `scripts/evaluate_all_tvs.py` now uses **one shared sigma per environment by default**, calibrated from a strong policy (`vi`) and reused for all methods. This follows the paper's Appendix A.1 idea that sigma is calibrated once per environment, not separately per policy. Use `--per-policy-sigma` only for diagnostics.

Additional report-ready visualizations are generated automatically:

```text
results/tvs_all/quality_diversity_scatter.png
results/tvs_all/return_diversity_scatter.png
results/tvs_all/coverage_entropy_tvs_comparison.png
results/tvs_all/<policy>/trajectory_overlay.png
```

`trajectory_overlay.png` is especially useful for presentations: it shows the actual route shapes used in the scored trajectories, complementing GIFs and the numerical TVS table.

See `docs/ARTICLE_ALIGNMENT_REVIEW.md` for the checklist mapping the paper's method to this Pacman implementation and the explicit limitations of the state projection.

Optional appendix-style robustness check:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_tvs_robustness.py --matrix results\tvs_all\vi\similarity_matrix.npy --output-dir results\tvs_all\vi
```
