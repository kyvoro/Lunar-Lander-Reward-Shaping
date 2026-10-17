import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from reward_wrappers import RewardDesignWrapper

DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_TRAINING_SEED = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("hover_bonus", "time_penalty", "landing_zone"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    args = parser.parse_args()

    model_path = Path("models") / f"ppo_{args.variant}_seed_{args.seed}_{args.timesteps}"
    monitor_path = (
        Path("results")
        / f"training_{args.variant}_seed_{args.seed}_{args.timesteps}.monitor.csv"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    monitor_path.parent.mkdir(parents=True, exist_ok=True)

    env = RewardDesignWrapper(gym.make("LunarLander-v3"), variant=args.variant)
    env = Monitor(
        env,
        filename=str(monitor_path),
        info_keywords=("episode_original_reward", "episode_shaping_reward"),
    )

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
        print(f"Reward-Variante: {args.variant}")
        print(f"Trainings-Seed: {args.seed}")
        print(f"Angeforderte Trainingsschritte: {args.timesteps}")
        print(f"Ausgeführte Trainingsschritte: {model.num_timesteps}")
        print(f"Modell gespeichert: {model_path}.zip")
        print(f"Episodenprotokoll: {monitor_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
