# Lunar Lander: Reward Shaping with PPO

A reproducible reinforcement-learning study of how reward design changes landing behaviour in Gymnasium's `LunarLander-v3` environment.

[![Tests](https://github.com/kyvoro/Lunar-Lander-Reward-Shaping/actions/workflows/tests.yml/badge.svg)](https://github.com/kyvoro/Lunar-Lander-Reward-Shaping/actions/workflows/tests.yml)

## Result

On the same 100 evaluation seeds (200–299), the landing-zone policy placed both landing-leg fixtures fully inside the pad in **79 episodes**, compared with **22** for the original-reward baseline. It crashed in **5** episodes compared with **17** for the baseline. Both policies were trained with seed 0; results across independent training seeds are still needed before claiming general reliability.

Among those matched episodes, landing-zone shaping improved pad success in 58 cases and regressed in one; the policies tied in the other 41. The evaluation uses the original, unmodified environment reward and reports physical pad success separately from Gymnasium's 200-point score threshold.

## Demo

[Watch the half-speed seed-100 evaluation](videos/lunar-lander-demo-english-0.5x.mp4). This is one playback of a trained policy, not a recording of the training process. The hover-bonus pilot clip shows a failure mode in which the lander remains airborne: [watch the pilot](videos/hover_bonus_100k/ppo_hover_bonus_seed_0_100000-seed-100-episode-0.mp4).

## Reward-design question

Can a locally attractive reward produce behaviour that conflicts with the actual task? The project compares the original reward with three deliberately different reward designs:

| Variant | Added reward | Intended lesson |
| --- | --- | --- |
| Hover bonus | `+1` while upright, slow, airborne, and without leg contact | A proxy objective can reward hovering instead of landing. |
| Time penalty | `−0.05` each step | Penalising delay does not guarantee safer landings. |
| Landing-zone shaping | Progress toward the pad center; `+100` for a stable in-zone landing; `−50` for a stable landing outside it | Reward feedback should distinguish task success from merely ending an episode. |

Gymnasium's original reward already penalises engine use, so the project does not add a second thrust penalty. Evaluation disables all shaping and measures return, crashes, stable landings, and whether the complete landing gear fits inside the pad.

## Run it

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python swig==4.5.1
PATH="$PWD/.venv/bin:$PATH" uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

The checked-in models let you view the trained policy without running a training job:

```bash
python watch_agent.py
```

To reproduce the matched 100-seed comparison:

```bash
python evaluate_baseline.py \
  --model-path models/ppo_baseline \
  --episodes 100 --seed-start 200 \
  --output results/ppo_baseline_eval_200_299.csv

python evaluate_baseline.py \
  --model-path models/ppo_landing_zone_seed_0_1000000 \
  --episodes 100 --seed-start 200 \
  --output results/landing_zone_seed_0_eval_200_299.csv
```

To train a fresh policy, specify the variant, seed, and step budget. PPO may finish its current rollout after the requested budget.

```bash
python train_reward_variant.py \
  --variant landing_zone --seed 1 --timesteps 1000000
```

## Verification

Run the unit tests with Python's standard test runner:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs these tests on pushes and pull requests. The tests cover stable-landing and pad-boundary metrics, reward components, terminal outcomes, and wrapper reset behaviour. They do not run model training.

## Evidence and limitations

- The 100 paired evaluation episodes use environment seeds 200–299; each policy was trained once with seed 0. The reported difference is strong for this sample, but repeated training seeds are required to assess training variability.
- The landing-zone reward includes a distance-progress term and therefore changes the objective. Evaluation with the shaping wrapper disabled helps expose whether a policy performs the original task.
- The 200-point threshold is Gymnasium's episode-score criterion, not a measurement that both legs landed inside the marked pad.
- `LunarLander-v3` is a 2D educational simulation, not an aerospace-validated flight model.
- The exact dependency pins were verified on macOS ARM64 with Python 3.12; other platforms still need compatibility checks.

See [the experiment log](docs/experiments.md) for methods and per-run results, and [the progress log](docs/progress.md) for milestones, authorship, and verification notes.

## CV-ready project entry

**English:** Designed and analysed an AI-assisted PPO reward-shaping study with Gymnasium and Stable-Baselines3; on 100 matched evaluation seeds, full-pad landings rose from 22% to 79% and crashes fell from 17 to 5 (one training seed per policy).

**Deutsch:** Konzipierte und analysierte eine KI-unterstützte PPO-Reward-Shaping-Studie mit Gymnasium und Stable-Baselines3; auf 100 gepaarten Test-Seeds stieg die Quote vollständiger Landezone-Landungen von 22 % auf 79 %, während die Crashs von 17 auf 5 sanken (je ein Trainings-Seed pro Variante).

## Project structure

```text
├── baseline.py                 # Random-policy reference
├── train_baseline.py           # Original-reward PPO training
├── train_reward_variant.py     # Reward-variant training
├── reward_wrappers.py          # Reward definitions
├── landing_metrics.py          # Stable and in-zone landing checks
├── evaluate_baseline.py        # Seeded evaluation and CSV output
├── watch_agent.py              # Live Gymnasium visualisation
├── models/                     # Small trained policies for replay
├── results/                    # Episode-level evaluation and training logs
├── tests/                      # Unit tests for reward and landing logic
├── .github/workflows/          # Automated test workflow
└── docs/                       # Experiment and contribution logs
```

## Sources and authorship

The project was inspired by Hari (@harixganesh), *Training an AI to land a rocket with reward shaping: how it works and how to build it*, supplied as pasted text; the original publication URL was not provided. This repository is an independent implementation and experiment.

The learner defined the task and reward preferences and completed guided Python and reinforcement-learning exercises. Codex wrote or modified several scripts at the learner's explicit request; contributions and verification are recorded in the progress log.

Technical references: [Gymnasium Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/), [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/stable/), [Gymnasium video recording](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo).
