import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from reward_wrappers import (
    HOVER_BONUS,
    OFF_PAD_LANDING_PENALTY,
    PAD_LANDING_BONUS,
    TIME_PENALTY,
    RewardDesignWrapper,
)


class Point:
    def __init__(self, x, y=0.0):
        self.x = x
        self.y = y


class Transform:
    def __init__(self):
        self.offset_x = 0.0

    def __mul__(self, point):
        return Point(self.offset_x + point.x, point.y)


def make_leg(relative_x):
    shape = SimpleNamespace(vertices=(Point(-0.1), Point(0.1)))
    fixture = SimpleNamespace(shape=shape)
    return SimpleNamespace(
        ground_contact=False,
        fixtures=(fixture,),
        transform=Transform(),
        relative_x=relative_x,
    )


class FakeLanderEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)
        self.helipad_x1 = 8.0
        self.helipad_x2 = 12.0
        self.lander = SimpleNamespace(position=Point(6.0), awake=True)
        self.legs = [make_leg(-0.25), make_leg(0.25)]
        self.game_over = False
        self.start_x = 6.0
        self.step_x = 6.0
        self.next_observation = np.asarray(
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self.episode_end = (True, False)
        self.next_game_over = False
        self.next_awake = True
        self.next_leg_contacts = (False, False)
        self.original_reward = 2.5

    def _update_leg_transforms(self):
        for leg in self.legs:
            leg.transform.offset_x = self.lander.position.x + leg.relative_x

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.lander.position.x = self.start_x
        self.lander.awake = True
        self.game_over = False
        for leg in self.legs:
            leg.ground_contact = False
        self._update_leg_transforms()
        return self.next_observation.copy(), {}

    def step(self, action):
        self.lander.position.x = self.step_x
        self.game_over = self.next_game_over
        self.lander.awake = self.next_awake
        for leg, contact in zip(self.legs, self.next_leg_contacts):
            leg.ground_contact = contact
        self._update_leg_transforms()
        terminated, truncated = self.episode_end
        return (
            self.next_observation.copy(),
            self.original_reward,
            terminated,
            truncated,
            {"source": "fake"},
        )


class RewardDesignWrapperTests(unittest.TestCase):
    def run_step(self, variant, env):
        wrapper = RewardDesignWrapper(env, variant=variant)
        observation, _ = wrapper.reset()
        result = wrapper.step(0)
        return wrapper, observation, result

    def test_hover_bonus_is_added_and_logged_separately(self):
        env = FakeLanderEnv()
        wrapper, _, result = self.run_step("hover_bonus", env)
        observation, reward, terminated, truncated, info = result

        self.assertTrue(np.array_equal(observation, env.next_observation))
        self.assertEqual(reward, env.original_reward + HOVER_BONUS)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["episode_original_reward"], env.original_reward)
        self.assertEqual(info["episode_shaping_reward"], HOVER_BONUS)
        self.assertEqual(wrapper.episode_shaping_reward, HOVER_BONUS)

    def test_time_penalty_is_applied_on_truncated_episode(self):
        env = FakeLanderEnv()
        env.episode_end = (False, True)
        _, _, result = self.run_step("time_penalty", env)
        _, reward, terminated, truncated, info = result

        self.assertAlmostEqual(reward, env.original_reward + TIME_PENALTY)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["episode_original_reward"], env.original_reward)
        self.assertEqual(info["episode_shaping_reward"], TIME_PENALTY)

    def test_landing_zone_rewards_progress_and_successful_touchdown(self):
        env = FakeLanderEnv()
        env.step_x = 10.0
        env.next_awake = False
        env.next_leg_contacts = (True, True)
        _, _, result = self.run_step("landing_zone", env)
        _, reward, _, _, info = result

        expected_shaping = 20.0 * 4.0 + PAD_LANDING_BONUS
        self.assertAlmostEqual(reward, env.original_reward + expected_shaping)
        self.assertAlmostEqual(info["episode_shaping_reward"], expected_shaping)

    def test_off_pad_stable_touchdown_gets_off_pad_penalty(self):
        env = FakeLanderEnv()
        env.step_x = 13.0
        env.next_awake = False
        env.next_leg_contacts = (True, True)
        _, _, result = self.run_step("landing_zone", env)
        _, reward, _, _, info = result

        expected_progress = 20.0 * (4.0 - 3.0)
        expected_shaping = expected_progress + OFF_PAD_LANDING_PENALTY
        self.assertAlmostEqual(reward, env.original_reward + expected_shaping)
        self.assertAlmostEqual(info["episode_shaping_reward"], expected_shaping)

    def test_reset_clears_episode_totals_and_distance_reference(self):
        env = FakeLanderEnv()
        wrapper = RewardDesignWrapper(env, variant="time_penalty")
        wrapper.reset()
        wrapper.step(0)
        wrapper.reset()

        self.assertEqual(wrapper.episode_original_reward, 0.0)
        self.assertEqual(wrapper.episode_shaping_reward, 0.0)
        self.assertEqual(wrapper.previous_pad_distance, 4.0)


if __name__ == "__main__":
    unittest.main()
