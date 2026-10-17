import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

DEFAULT_MODEL_PATH = Path("models/ppo_baseline")
DEFAULT_VIDEO_DIR = Path("videos/ppo_baseline_1m")
DEFAULT_RECORDING_SEED = 100
MAX_STEPS = 1_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_RECORDING_SEED)
    args = parser.parse_args()

    model_path = args.model_path.with_suffix("") if args.model_path.suffix == ".zip" else args.model_path
    model_file = model_path.with_suffix(".zip")
    if not model_file.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden: {model_file}. Bitte zuerst train_baseline.py ausführen."
        )

    args.video_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(args.video_dir),
        episode_trigger=lambda episode_id: True,
        name_prefix=f"{model_path.name}-seed-{args.seed}",
    )

    try:
        model = PPO.load(str(model_path), device="cpu")
        observation, info = env.reset(seed=args.seed)
        total_reward = 0.0
        terminated = False
        truncated = False

        for step in range(MAX_STEPS):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        print(
            f"Seed {args.seed}: Reward {total_reward:.2f}, "
            f"Schritte {step + 1}, beendet={terminated}, Zeitlimit={truncated}"
        )
        print(f"Video gespeichert in: {args.video_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
