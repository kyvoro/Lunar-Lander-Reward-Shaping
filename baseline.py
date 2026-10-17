import csv
from pathlib import Path
from statistics import fmean

import gymnasium as gym

episode_returns = []
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
results_path = results_dir / "random_baseline.csv"
env = gym.make("LunarLander-v3")

with results_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["episode", "seed", "episode_return", "steps", "terminated", "truncated"])

    for episode in range(10):
        seed = episode
        observation, info = env.reset(seed=seed)
        env.action_space.seed(seed)
        total_reward = 0.0

        for step in range(1000):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        episode_returns.append(total_reward)
        writer.writerow([episode + 1, seed, total_reward, step + 1, terminated, truncated])
        print(f"Episode {episode + 1}: Reward {total_reward:.2f}, Schritte {step + 1}")

env.close()

print(f"Mittlerer Reward: {fmean(episode_returns):.2f}")
print(f"Ergebnisse gespeichert in: {results_path}")
