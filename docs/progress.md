# Progress and learning log

Record implemented behaviour, verification, and authorship after each milestone. Learning objectives are only marked as demonstrated after the learner explains or applies them.

## Roadmap

- [x] Explain the agent–environment basics and write small Python exercises.
- [x] Verify the Python environment and a short simulation run.
- [x] Implement a random agent and record its baseline.
- [x] Implement and verify PPO training, saving, loading, and evaluation.
- [ ] Complete baseline training and evaluation across three training seeds.
- [x] Implement and run equal-budget reward-variant pilots.
- [ ] Run the multi-seed comparison and analyse results.
- [ ] Prepare final documentation and GitHub publication.

## 2026-10-17 — Initial setup

### Decisions

- The learner develops programming and reinforcement-learning understanding through guided exercises and helps define the task and reward requirements. Codex implementation work is recorded by milestone.
- Sessions are divided into approximately 20–30 minute learning blocks.
- Documentation is written alongside implementation, primarily in English for the eventual GitHub repository.
- Begin locally with Python 3.12 and CPU training.

### Contributions

Codex prepared the documentation and isolated environment, initialized the local Git repository on `main`, and ran a temporary setup smoke check. That check is infrastructure verification, not a learner-authored random-agent implementation. No learner-authored Python code or trained model exists yet. Nothing has been published to GitHub.

### Verification

- Python 3.12.13; Gymnasium 1.4.0; Stable-Baselines3 2.9.0; PyTorch 2.14.1.
- All installed package dependencies passed `uv pip check`.
- A seeded setup smoke check completed 100 random steps in `LunarLander-v3`, resetting after episode endings as needed.
- Verified eight observation values, four discrete actions, and a rendered RGB frame of shape `(400, 600, 3)`.
- Pinned all 35 installed package versions in `requirements.txt`.
- The first process startup was slow; a process sample showed the macOS dynamic loader loading a native image module. The check subsequently exited successfully. The sample alone does not establish the underlying cause of the delay.
- Full model training, model loading, and video encoding have not been tested yet.

### Next learning objective

Explain what observations, actions, and rewards mean in the rocket simulation. Then write the first small Python exercise before implementing the random agent.

## 2026-10-17 — Reward-design discussion

### Learner contributions

- Correctly identified the agent as the AI controlling the rocket and the environment as the physical simulation.
- Proposed penalising failure to land by the episode deadline and rewarding a safe landing.
- Recognised that a slow, safe landing serves the intended task better than a quick crash.

### Concepts discussed with guidance

- Reward follows the specified rules and does not automatically measure progress towards the intended goal.
- A per-step hovering bonus can encourage remaining airborne.
- Penalising thrust can also penalise necessary braking; rewarding more thrust can encourage unnecessary engine use.
- Accumulated bonuses can outweigh a penalty at the end of an episode.
- A time penalty must not make a quick crash preferable to a safe landing.

These concepts have been discussed, not yet verified through code or a numerical exercise. No reward-design experiment has been executed.

### Next exercise

Write the first learner-authored Python code in `first_steps.py`: store the environment name and an initial cumulative reward in variables, then print both values. Codex prepared an empty file; the exercise solution is left to the learner.

## 2026-10-17 — First Python exercise

The learner wrote two variable assignments and printed both variables in a single `print()` call. Codex reviewed and executed the saved file with the project's Python interpreter; it exited successfully and printed `LunarLander-v2 0`.

The learner subsequently corrected the name to `LunarLander-v3` and independently added 10 to `total_reward`, subtracted 3, and printed the result. Codex executed the updated file successfully; its output was `LunarLander-v3 0` followed by `7` on the next line.

Variable assignment, basic output, and updating a running total were demonstrated. No environment is created by this exercise; the environment name is currently only a string.

The learner then used a `for` loop to accumulate `[10, -3, 5, -2]`. An initial attempt retained the previous total of 7 and produced 17. After review, the learner removed the earlier arithmetic so accumulation started at zero. Codex executed the corrected file successfully; it printed `LunarLander-v3 0` followed by `10`. List iteration and accumulation are demonstrated; the exercise also exposed the importance of initial state.

The learner added an `if`/`else` classification of total reward. Tests exposed that the initial `>= 0` condition incorrectly classified zero as positive. The learner changed it to `> 0`. Codex then checked positive, zero, and negative totals using in-memory input substitutions without editing the learner's file; all three classifications passed.

Next: create `random_agent.py`, import Gymnasium, create and reset the environment, inspect its initial observation, and close it. Action selection and the episode loop will follow separately.

## 2026-10-17 — First environment interaction

Following a three-line example from Codex, the learner created `random_agent.py`, imported Gymnasium, created `LunarLander-v3`, and unpacked the two return values of `reset()` into `observation` and `info`. The learner added observation output. An initial `render()` call produced a warning because no render mode was configured; after feedback, the learner replaced it with `close()`.

Codex executed the corrected script successfully: it printed eight observation values and exited without warnings. No action has been taken by this script yet, and it has no episode loop or trained policy. Next: sample one random action, execute one environment step, and inspect the returned reward.

The learner subsequently added the supplied `action_space.sample()` and `step(action)` calls. After feedback distinguishing observation output from action/reward output, the learner printed `action` and `reward`. Codex executed the updated file successfully and confirmed that one random action is taken and its scalar reward is printed. This is a single simulation step, with no learning or complete episode loop yet. Next: accumulate rewards over one complete random episode and stop on termination or truncation.

## 2026-10-17 — Complete random episode

The learner attempted the episode loop. Review identified a `break` outside the loop, invalid action sampling, a premature environment close, and an unrelated import. At the learner's explicit request, Codex wrote the corrected episode-loop implementation in `random_agent.py`.

The script samples one valid random action per step, executes it, accumulates reward, and stops on `terminated` or `truncated`, with a maximum of 1000 steps. It prints total reward and closes the environment. This implementation was supplied by Codex; independent learner understanding of the loop has not yet been checked.

Codex executed the saved script and verified that the episode ended within the step limit: 92 steps, `terminated=True`, `truncated=False`, total original reward approximately -134.8040. The unseeded run is a smoke check, not a reproducible benchmark or training result. No policy is trained.

The learner enabled Gymnasium's `human` render mode in `random_agent.py`. Codex ran the visible script through one episode; it completed and printed total reward approximately -396.10. The run is unseeded and illustrative only. The episode window closed when the script finished. Repeated baseline runs and result logging remain to be added.

An attempted terminal `input()` pause left the human-rendered game window unresponsive while Python waited for input. Codex replaced that approach with Gymnasium's `RecordVideo`, saving a random episode as an MP4. Codex executed that version successfully; the game ran off-screen, so the unresponsiveness was avoided. The clip was made available for playback.

## 2026-10-17 — Seeded random baseline

At the learner's request, Codex expanded `baseline.py` to run ten random-action episodes with seeds 0–9 for both environment resets and action sampling. The script reports the return and episode length per run, writes episode metadata to `results/random_baseline.csv`, and prints the mean return. The CSV is recreated each time `baseline.py` runs. The visual recording remains a separate one-episode script.

Codex ran the baseline script successfully. Mean original return was -173.23, with a range from -279.90 to -81.49. All ten episodes ended with `terminated=True` and `truncated=False`; the recorded flags do not distinguish crashing from a successful resting landing. This establishes a random-policy score reference only; it is neither a landing success rate nor a trained-agent result.

The planned short PPO pipeline run was completed; see the training and evaluation entry below.

## 2026-10-17 — Initial PPO training and evaluation

At the learner's request, Codex wrote `train_baseline.py` and `evaluate_baseline.py`. The training script uses PPO with `MlpPolicy`, CPU execution, seed 0, the original Gymnasium reward, and a requested budget of 100,000 steps. The model is saved to the `models/` directory. The evaluation script loads the saved model and evaluates deterministic actions on seeds 100–109 in an unmodified `LunarLander-v3` environment, writing episode records to `results/ppo_baseline_eval.csv`.

Codex executed the training run: PPO completed 100,352 steps (the full rollout exceeded the requested budget) and saved `models/ppo_baseline.zip`. Evaluation completed successfully. Mean original reward was -59.81, ranging from -84.65 to -43.90. All ten episodes reached the 1,000-step time limit; none terminated before it. This preliminary score is better than the random sample's mean of -173.23, but the evaluation uses only ten seeds and did not demonstrate a landing.

The next step is to record evaluation seed 100 with `record_baseline.py` and inspect the behavior before diagnosing why the policy reaches the time limit. The 100,000-step run is a workflow milestone, not the final baseline comparison; the planned main runs still use 1,000,000 requested steps per seed.

Codex wrote and ran `record_baseline.py` to save deterministic evaluation seed 100 as an MP4 in `videos/ppo_baseline/`. The recording is 20.02 seconds long. Frames sampled from the beginning, middle, and end show the lander still airborne at the end and left of the landing pad. This confirms the observed seed did not visibly land, but the sampled frames do not explain the complete flight path. The run still ends at the 1,000-step time limit.

That next step was completed in the one-million-step baseline entry below.

## 2026-10-17 — One-million-step baseline

Codex increased the baseline budget to 1,000,000 requested steps and trained PPO with seed 0 on the original reward. PPO completed 1,001,472 steps and saved the model to `models/ppo_baseline.zip`. The evaluation script loads this saved model and runs deterministic episodes with environment seeds 100–109; the CSV includes the documented 200-point solution threshold alongside episode returns and ending flags.

The ten-episode mean original return was 214.19, with a range from 119.78 to 259.49. Seven episodes scored at least 200; nine terminated before the time limit and one reached the 1,000-step limit. This is a preliminary result for one training seed, not the planned multi-seed baseline. Gymnasium defines an episode score of 200 or more as a solution; this threshold is recorded separately from `terminated` and `truncated`.

Codex updated `record_baseline.py` to save the million-step model's seed-100 video separately under `videos/ppo_baseline_1m/`, preserving the earlier 100,000-step recording. The video shows the lander on the marked pad at the end of the 364-step episode, which scored 214.03 and ended with `terminated=True`. This is one visually inspected seed, not a multi-seed landing rate. The earlier next-step note to inspect the clip was completed before the pilot comparison below.

## 2026-10-17 — Reward-wrapper implementation and pilot comparison

At the learner's request, Codex added `RewardDesignWrapper`, `train_reward_variant.py`, and argument options for training, evaluation, and video recording. The hover wrapper adds `+1` for a nearly upright, low-speed lander above the surface with neither leg in contact. The time-penalty wrapper adds `-0.05` each step. The wrapper records original and added reward totals in Stable-Baselines3 Monitor output; evaluation loads each model into the unmodified environment.

Codex ran equal 100,000-step pilots with training seed 0 for the original reward, hover bonus, and time penalty, then evaluated all three models deterministically on environment seeds 100–109. Mean original evaluation returns were -59.81, -71.94, and -97.34 respectively. None of the pilot episodes scored 200 or more. All ten hover-bonus evaluation episodes reached the time limit; the time-penalty policy reached the limit in eight of ten.

The hover-bonus training monitor covered 332 episodes and showed mean original return -103.95, added reward +156.23, and combined return +52.28; 57 episodes reached the time limit. A recorded seed-100 episode remained airborne above the pad at its final frame. This supports the intended reward-hacking example at pilot scale, while the time-penalty pilot did not demonstrate improved landings. Neither result replaces the planned 1,000,000-step, three-training-seed comparison.

Pilot evaluations are saved in `results/ppo_baseline_100k_pilot.csv`, `results/hover_bonus_100k_eval.csv`, and `results/time_penalty_100k_eval.csv`. Monitor logs preserve the separate reward components. Generated training clips are local outputs; the hover-bonus seed-100 clip is included as a supporting demo.

## 2026-10-17 — Live policy animation

At the learner's request, Codex added `watch_agent.py` to display a trained policy directly in a Gymnasium `human` render window, without recording a video. It loads the one-million-step baseline by default, runs deterministic actions on seed 100, reports the episode result, and closes the window automatically after a short pause.

Codex ran the script successfully. The live episode completed in 364 steps with original reward 214.03, `terminated=True`, and `truncated=False`. The script exited normally after the two-second close pause. This is a visual run of one fixed evaluation seed, not a new evaluation or training result.

## 2026-10-17 — PyCharm run target and landing result

When the learner reported a crash, PyCharm's selected run configuration was `random_agent`, which executes random actions and can crash by design. That run also waited at an `input()` prompt after the episode, making the Run console appear unresponsive. Codex stopped it and selected the `watch_agent` configuration for the live trained policy.

To make the distinction visible, `watch_agent.py` now reports `sichere Landung`, `Crash`, or a non-landing outcome based on the environment's final state. The `watch_agent` configuration was run from PyCharm and reported `sichere Landung` on seed 100 after 364 steps, with reward 214.03. The underlying lander had both legs in contact, was no longer awake, and `game_over` was false. The updated run target is documented above.

## 2026-10-17 — Detailed live scene

At the learner's request, Codex added a custom Pygame renderer for the live PPO run. `watch_agent.py` now reads the trained policy and Gymnasium state as before, while `rocket_renderer.py` draws a star field, distant planet, lunar terrain and craters, landing zone, lander details, active engine plumes, and flight telemetry. No video is recorded, and no policy, reward, or physics behavior was changed. The README describes the scene as a stylized 2D visualization rather than a more realistic physics model.

Codex compiled both changed Python files and ran the full deterministic seed-100 episode in the visible PyCharm configuration. It finished after 364 steps with reward 214.03 and reported `sichere Landung`; the run exited with code 0. The first visual review exposed duplicated landing legs, so the renderer now draws the Box2D leg geometry once and colors actual ground-contact legs green. This improves the presentation only; the underlying policy, reward, and physics remain unchanged.

## 2026-10-17 — Landing-zone requirement

The learner clarified that the rocket must land directly inside the marked landing zone. A deterministic seed-100 readback showed why the previous success label was too broad: the lander center was at x=11.407 within the pad interval [8, 12], but one leg fixture extended to x=12.410. Both legs had ground contact, so the old check counted this as a safe landing even though the complete landing gear did not fit inside the pad.

Codex changed the shared landing metric to require a stable lander and both complete leg fixtures inside the pad interval. The live run and evaluation now report this zone result separately from a generic resting landing and the 200-point score. A new `landing_zone` reward variant adds horizontal progress toward pad center, a +100 terminal reward for a stable in-zone landing, and a -50 terminal reward for a stable off-zone landing.

The new policy trained for 1,001,472 steps from seed 0 and was evaluated deterministically on seeds 100–109. It achieved 8/10 complete in-zone landings, 9/10 stable landings, one episode without a stable resting landing, and no crashes. Mean original reward was 280.32. All ten returns exceeded 200, so this run confirms that the score threshold can overstate pad-landing performance. The model is now the default for `watch_agent.py`; the default seed 100 passes the strict zone check after 201 steps with reward 247.93.

Codex then compared this model with the original baseline on the same additional seeds 200–299. The landing-zone model reached 79/100 complete in-zone landings, 84/100 stable landings, and 5 crashes; the baseline reached 22/100, 63/100, and 17 crashes, respectively. Mean original returns were 260.11 and 177.30. Both policies used training seed 0. The zone shaping greatly improved placement on this sample, but it does not guarantee success on every seed. Evaluation files are `results/landing_zone_seed_0_eval_200_299.csv` and `results/ppo_baseline_eval_200_299.csv`.

## 2026-10-17 — Photorealistic visual assets

At the learner's request, Codex used the built-in Imagegen tool to generate four original game assets: an empty lunar vista, a regolith texture, a transparent metal-and-gold lander body, and transparent engine exhaust. The PNG files are in `assets/`; `assets/manifest.json` records the exact prompts, tool, date, and transparency settings. These are generated artwork, not photographs or validated aerospace reference material.

Codex integrated the assets into the existing live Pygame renderer at 1200×800. The regolith is masked to the actual Box2D terrain and cached between frames. The pad top and boundaries match the physics coordinates; the lander body rotates with the physical body, and the landing legs retain their actual fixture geometry. Exhaust follows the selected action, with a ground shadow and brief surface-dust streaks near touchdown. No additional dependencies, model training, reward changes, or physics changes were needed. A frame captured from the live renderer is saved in `docs/graphics-preview.png`.

Verification used SDL's dummy display driver: the default `watch_agent.py` command completed seed 100 with a successful in-zone landing after 201 steps and original return 247.93. A separate paired run compared every observation, reward, and ending flag against an identical environment without rendering; all matched. All four actions, a terrain reset, the final outcome display, the short end pause, and a synthetic window-close event were checked. Mean frame rendering time was 4.30 ms, with a 4.84 ms 95th percentile in this headless check; this is not a visible-window performance benchmark. The in-game preview was visually inspected. Opening the new scene in PyCharm remains a user-side step.

## 2026-10-17 — Repository cleanup

To reduce the GitHub project, the custom renderer, generated image assets, and static preview were removed. `watch_agent.py` now uses Gymnasium's built-in live rendering; training, rewards, models, evaluation metrics, and results are unchanged. The standalone beginner exercises and older local recordings were also removed as redundant. The half-speed demo and hover-bonus pilot clip remain; the README identifies the half-speed video as a capture of the earlier renderer.
