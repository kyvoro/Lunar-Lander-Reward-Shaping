# Lunar Lander: Reward Shaping with PPO

A guided learning project investigating how reward design affects the behaviour of a reinforcement learning agent in Gymnasium's `LunarLander-v3` environment.

## Status

Initial setup verified on macOS ARM64 with Python 3.12.13. A one-million-step PPO baseline, 100,000-step reward-shaping pilots, and a one-million-step landing-zone model have been run. On a separate 100-seed comparison (seeds 200–299), the zone model landed fully inside the pad in 79 episodes, versus 22 for the baseline; crashes fell from 17 to 5. Both models used training seed 0, so results across training seeds remain to be checked. On the smaller development set 100–109, the zone model succeeded in 8/10 episodes. The original baseline's 200-point score threshold also overstated pad success: it reached 200 in 73/100 episodes but landed fully in the pad in only 22. Final equal-budget, multi-seed comparisons remain to be run.

## Research question

Can a reward for remaining upright and slow encourage hovering instead of landing, and what happens when that incentive is removed and waiting incurs a small cost?

The project compares the original reward, a deliberately flawed hovering bonus, and corrections that encourage a completed landing inside the pad. PPO is provided by Stable-Baselines3; the simulation is provided by Gymnasium. The project's own work consists of the reward wrappers, experiment pipeline, evaluation, and analysis.

## Setup

Prerequisites: Python 3.12, Git, and [uv](https://docs.astral.sh/uv/).

Run these commands from the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python swig==4.5.1
PATH="$PWD/.venv/bin:$PATH" uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
python --version
```

SWIG is installed first so it is available if a physics dependency needs to be built from source. `requirements.txt` records the exact versions installed in the development environment; platform-specific compatibility still needs to be verified on other operating systems.

## Random baseline

Run ten seeded episodes of random actions:

```bash
source .venv/bin/activate
python baseline.py
```

The script prints each episode return and their mean, then writes the per-episode returns, seeds, episode lengths, and ending flags to `results/random_baseline.csv`. Re-running it replaces this generated CSV. These scores are an untrained random-agent baseline, not a training result.

Training uses the CPU.

## PPO baseline

Train the baseline policy for 1,000,000 environment steps on the CPU:

```bash
source .venv/bin/activate
python train_baseline.py
```

The model is saved under `models/ppo_baseline.zip`. The trained baseline and reward-variant model files are included so you can evaluate them without running training first. Evaluate the baseline on ten separate environment seeds with the original, unmodified reward:

```bash
python evaluate_baseline.py
```

Per-episode results are written to `results/ppo_baseline_eval.csv`. PPO completes its current rollout, so the actual step count can slightly exceed the requested budget. The first 100,000-step workflow run did not land on any of ten evaluation episodes; see [the experiment log](docs/experiments.md). Episode returns alone do not establish safe landing success.

Record the deterministic policy on evaluation seed 100:

```bash
python record_baseline.py
```

The MP4 is written under `videos/ppo_baseline_1m/`; newly generated training recordings are ignored by Git. The earlier 100,000-step recording remains in that local output folder.

To watch the trained agent live in Gymnasium's built-in window without recording a video, run:

```bash
python watch_agent.py
```

The live run uses the landing-zone model and seed 100 by default. A successful landing requires both complete landing-leg fixtures to fit inside the pad; resting safely elsewhere is reported as a missed zone. The built-in view displays Gymnasium's 2D simulation; it is not an aerospace-validated simulator. The window closes two seconds after the episode ends. You can select another model, seed, step limit, or pause duration with `--model-path`, `--seed`, `--max-steps`, and `--pause-seconds`.

The [half-speed demo video](videos/lunar-lander-demo-english-0.5x.mp4) shows one evaluation playback of the trained landing-zone policy on seed 100 using the earlier custom renderer. It is not a visualization of training. The repository also includes a [hover-bonus pilot clip](videos/hover_bonus_100k/ppo_hover_bonus_seed_0_100000-seed-100-episode-0.mp4) that illustrates the agent remaining airborne.

In PyCharm, create a Python run configuration for `watch_agent.py` with the repository root as the working directory.

Additional training seeds and budgets can be named explicitly so runs do not overwrite each other's models:

```bash
python train_baseline.py --seed 1 --timesteps 1000000 --model-name ppo_baseline_seed_1
```

## Reward-shaping pilots

Train each reward variant from scratch. These examples use 100,000 steps for a quick pilot; the planned comparison uses 1,000,000 steps per run:

```bash
python train_reward_variant.py --variant hover_bonus --timesteps 100000 --seed 0
python train_reward_variant.py --variant time_penalty --timesteps 100000 --seed 0
```

Train a policy that targets a landing fully inside the pad:

```bash
python train_reward_variant.py --variant landing_zone --timesteps 1000000 --seed 0
python evaluate_baseline.py \
  --model-path models/ppo_landing_zone_seed_0_1000000 \
  --output results/landing_zone_seed_0_eval.csv
python watch_agent.py \
  --model-path models/ppo_landing_zone_seed_0_1000000 \
  --seed 100
```

The `landing_zone` variant adds a small reward for moving toward the pad center, a terminal bonus for a stable landing with both leg fixtures inside the pad, and a smaller penalty for a stable landing outside it. Evaluation reports zone success separately from the 200-point score threshold.

The training monitor CSV keeps original and added reward totals separate. Evaluate any saved variant in the original environment, with no reward wrapper:

```bash
python evaluate_baseline.py \
  --model-path models/ppo_hover_bonus_seed_0_100000 \
  --output results/hover_bonus_100k_eval.csv
```

The hover rule adds `+1` when the lander is nearly upright and slow, above the surface, with neither leg touching. The correction adds `-0.05` on every step. Gymnasium's original reward already includes engine-use costs, so the project adds no second thrust penalty.

Record a variant for visual inspection by passing its model path and a separate video folder:

```bash
python record_baseline.py \
  --model-path models/ppo_hover_bonus_seed_0_100000 \
  --video-dir videos/hover_bonus_100k \
  --seed 100
```

## Documentation

- [Experiment plan](docs/experiments.md)
- [Progress and learning log](docs/progress.md)

## Sources and attribution

Project inspiration: Hari (@harixganesh), *Training an AI to land a rocket with reward shaping: how it works and how to build it*, supplied as pasted text. The original publication URL has not been provided. This repository does not reproduce the article.

- [Gymnasium: Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Stable-Baselines3: installation](https://stable-baselines3.readthedocs.io/en/stable/guide/install.html)
- [Stable-Baselines3: RL evaluation guidance](https://stable-baselines3.readthedocs.io/en/stable/guide/rl_tips.html)
- [Gymnasium: recording videos](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo)

Development is assisted by Codex. The learner defined the task requirements and reward preferences and worked through guided exercises; Codex wrote or modified several scripts at the learner's explicit request. The progress log records contributions and verification by milestone. Planned work is not presented as a completed result.
