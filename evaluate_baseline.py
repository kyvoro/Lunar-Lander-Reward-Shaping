import argparse
import csv
from pathlib import Path
from statistics import fmean

import gymnasium as gym
from stable_baselines3 import PPO

from landing_metrics import is_stable_landing, is_successful_landing_in_zone

MAX_STEPS = 1_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/ppo_baseline"))
    parser.add_argument("--output", type=Path, default=Path("results/ppo_baseline_eval.csv"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=100)
    args = parser.parse_args()

    model_path = args.model_path.with_suffix("") if args.model_path.suffix == ".zip" else args.model_path
    model_file = model_path.with_suffix(".zip")
    if not model_file.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden: {model_file}. Bitte zuerst train_baseline.py ausführen."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make("LunarLander-v3")
    episode_returns = []
    stable_landings = 0
    landing_zone_successes = 0
    crashes = 0

    try:
        model = PPO.load(str(model_path), device="cpu")

        with args.output.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "episode",
                    "seed",
                    "original_episode_return",
                    "score_at_least_200",
                    "steps",
                    "terminated",
                    "truncated",
                    "crash",
                    "stable_landing",
                    "landing_zone_success",
                    "lander_x",
                    "pad_left",
                    "pad_right",
                ]
            )

            for episode, seed in enumerate(
                range(args.seed_start, args.seed_start + args.episodes), start=1
            ):
                observation, info = env.reset(seed=seed)
                total_reward = 0.0
                terminated = False
                truncated = False

                for step in range(MAX_STEPS):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward

                    if terminated or truncated:
                        break

                episode_returns.append(total_reward)
                score_at_least_200 = total_reward >= 200
                base_env = env.unwrapped
                crash = bool(base_env.game_over)
                stable_landing = is_stable_landing(base_env)
                landing_zone_success = is_successful_landing_in_zone(base_env)
                lander_x = float(base_env.lander.position.x)
                crashes += int(crash)
                stable_landings += int(stable_landing)
                landing_zone_successes += int(landing_zone_success)
                writer.writerow(
                    [
                        episode,
                        seed,
                        total_reward,
                        score_at_least_200,
                        step + 1,
                        terminated,
                        truncated,
                        crash,
                        stable_landing,
                        landing_zone_success,
                        lander_x,
                        float(base_env.helipad_x1),
                        float(base_env.helipad_x2),
                    ]
                )
                print(
                    f"Episode {episode}: Reward {total_reward:.2f}, "
                    f"200-Punkte-Ziel={score_at_least_200}, Schritte {step + 1}, "
                    f"beendet={terminated}, Zeitlimit={truncated}, "
                    f"Crash={crash}, "
                    f"stabile Landung={stable_landing}, "
                    f"Landung in Zone={landing_zone_success}, x={lander_x:.3f}"
                )
    finally:
        env.close()

    print(f"Mittlerer Original-Reward: {fmean(episode_returns):.2f}")
    print(f"Stabile Landungen: {stable_landings}/{args.episodes}")
    print(f"Landung vollständig in der Zone: {landing_zone_successes}/{args.episodes}")
    print(f"Crashes: {crashes}/{args.episodes}")
    print(f"Ergebnisse gespeichert in: {args.output}")


if __name__ == "__main__":
    main()
