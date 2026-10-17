import gymnasium as gym

from landing_metrics import is_stable_landing, is_successful_landing_in_zone

HOVER_BONUS = 1.0
TIME_PENALTY = -0.05
MAX_UPRIGHT_ANGLE = 0.2
MAX_SLOW_SPEED = 0.2
MIN_HOVER_HEIGHT = 0.1
PAD_CENTER_PROGRESS_SCALE = 20.0
PAD_LANDING_BONUS = 100.0
OFF_PAD_LANDING_PENALTY = -50.0


class RewardDesignWrapper(gym.Wrapper):
    def __init__(self, env, variant):
        super().__init__(env)
        if variant not in {"hover_bonus", "time_penalty", "landing_zone"}:
            raise ValueError(f"Unbekannte Reward-Variante: {variant}")
        self.variant = variant
        self.episode_original_reward = 0.0
        self.episode_shaping_reward = 0.0
        self.previous_pad_distance = 0.0

    def reset(self, **kwargs):
        self.episode_original_reward = 0.0
        self.episode_shaping_reward = 0.0
        observation, info = self.env.reset(**kwargs)
        base_env = self.env.unwrapped
        pad_center = (base_env.helipad_x1 + base_env.helipad_x2) / 2
        self.previous_pad_distance = abs(float(base_env.lander.position.x) - pad_center)
        return observation, info

    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)

        if self.variant == "time_penalty":
            shaping_reward = TIME_PENALTY
        elif self.variant == "landing_zone":
            base_env = self.env.unwrapped
            pad_center = (base_env.helipad_x1 + base_env.helipad_x2) / 2
            pad_distance = abs(float(base_env.lander.position.x) - pad_center)
            shaping_reward = PAD_CENTER_PROGRESS_SCALE * (
                self.previous_pad_distance - pad_distance
            )
            self.previous_pad_distance = pad_distance

            if terminated or truncated:
                if is_successful_landing_in_zone(base_env):
                    shaping_reward += PAD_LANDING_BONUS
                elif is_stable_landing(base_env):
                    shaping_reward += OFF_PAD_LANDING_PENALTY
        else:
            angle = abs(float(observation[4]))
            speed_x = float(observation[2])
            speed_y = float(observation[3])
            speed = (speed_x**2 + speed_y**2) ** 0.5
            height = float(observation[1])
            no_leg_contact = observation[6] < 0.5 and observation[7] < 0.5
            is_hovering = (
                angle < MAX_UPRIGHT_ANGLE
                and speed < MAX_SLOW_SPEED
                and height > MIN_HOVER_HEIGHT
                and no_leg_contact
            )
            shaping_reward = HOVER_BONUS if is_hovering else 0.0

        original_reward = float(original_reward)
        self.episode_original_reward += original_reward
        self.episode_shaping_reward += shaping_reward

        info = dict(info)
        if terminated or truncated:
            info["episode_original_reward"] = self.episode_original_reward
            info["episode_shaping_reward"] = self.episode_shaping_reward

        return (
            observation,
            original_reward + shaping_reward,
            terminated,
            truncated,
            info,
        )
