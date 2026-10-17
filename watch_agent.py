import argparse
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from landing_metrics import is_stable_landing, is_successful_landing_in_zone

DEFAULT_MODEL_PATH = Path("models/ppo_landing_zone_seed_0_1000000")
DEFAULT_SEED = 100
DEFAULT_MAX_STEPS = 1_000
DEFAULT_PAUSE_SECONDS = 2.0


def main():
    parser = argparse.ArgumentParser(
        description="Zeigt eine PPO-Landung live in einem Python-Fenster."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--pause-seconds", type=float, default=DEFAULT_PAUSE_SECONDS)
    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max-steps muss mindestens 1 sein.")
    if args.pause_seconds < 0:
        parser.error("--pause-seconds darf nicht negativ sein.")

    model_path = (
        args.model_path.with_suffix("")
        if args.model_path.suffix == ".zip"
        else args.model_path
    )
    model_file = model_path.with_suffix(".zip")
    if not model_file.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden: {model_file}. Bitte zuerst train_baseline.py ausführen."
        )

    env = gym.make("LunarLander-v3", render_mode="human")
    try:
        model = PPO.load(str(model_path), device="cpu")
        observation, _ = env.reset(seed=args.seed)
        total_reward = 0.0
        terminated = False
        truncated = False
        step_count = 0
        action = 0
        base_env = env.unwrapped

        env.render()

        for step in range(args.max_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step_count = step + 1
            base_env = env.unwrapped

            if terminated or truncated:
                break

        if is_successful_landing_in_zone(base_env):
            outcome = "Landung in der Landezone"
        elif base_env.game_over:
            outcome = "Crash"
        elif is_stable_landing(base_env):
            outcome = "Landung außerhalb der Landezone"
        elif truncated:
            outcome = "Zeitlimit, keine sichere Landung"
        else:
            outcome = "keine sichere Landung bestätigt"

        print(
            f"Ergebnis: {outcome} | Seed {args.seed} | Reward {total_reward:.2f} | "
            f"Schritte {step_count}"
        )
        if args.pause_seconds:
            print(f"Das Live-Fenster bleibt {args.pause_seconds:.1f} Sekunden geöffnet.")
            time.sleep(args.pause_seconds)
    finally:
        env.close()


if __name__ == "__main__":
    main()
