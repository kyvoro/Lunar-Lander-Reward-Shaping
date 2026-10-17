import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

DEFAULT_TRAINING_SEED = 0
DEFAULT_TOTAL_TIMESTEPS = 1_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--model-name", default="ppo_baseline")
    args = parser.parse_args()

    model_path = Path("models") / args.model_name
    model_path.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make("LunarLander-v3")

    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            seed=args.seed,
            verbose=1,
            device="cpu",
        )
        model.learn(total_timesteps=args.timesteps)
        model.save(str(model_path))
        print(f"Trainings-Seed: {args.seed}")
        print(f"Angeforderte Trainingsschritte: {args.timesteps}")
        print(f"Ausgeführte Trainingsschritte: {model.num_timesteps}")
        print(f"Modell gespeichert: {model_path}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
