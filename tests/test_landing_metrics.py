import unittest
from types import SimpleNamespace

from landing_metrics import (
    is_stable_landing,
    is_successful_landing_in_zone,
    landing_leg_x_extents,
)


class Point:
    def __init__(self, x, y=0.0):
        self.x = x
        self.y = y


class Transform:
    def __init__(self, offset_x):
        self.offset_x = offset_x

    def __mul__(self, point):
        return Point(self.offset_x + point.x, point.y)


def make_leg(center_x, ground_contact=True, half_width=0.2):
    shape = SimpleNamespace(
        vertices=(Point(-half_width), Point(half_width))
    )
    fixture = SimpleNamespace(shape=shape)
    return SimpleNamespace(
        ground_contact=ground_contact,
        fixtures=(fixture,),
        transform=Transform(center_x),
    )


def make_env(legs, awake=False, game_over=False):
    return SimpleNamespace(
        helipad_x1=8.0,
        helipad_x2=12.0,
        legs=legs,
        lander=SimpleNamespace(awake=awake),
        game_over=game_over,
    )


class LandingMetricTests(unittest.TestCase):
    def test_stable_landing_requires_quiet_lander_and_both_legs_in_contact(self):
        self.assertTrue(is_stable_landing(make_env([make_leg(9), make_leg(11)])))
        self.assertFalse(
            is_stable_landing(
                make_env([make_leg(9), make_leg(11)], game_over=True)
            )
        )
        self.assertFalse(
            is_stable_landing(
                make_env([make_leg(9), make_leg(11)], awake=True)
            )
        )
        self.assertFalse(
            is_stable_landing(
                make_env([make_leg(9), make_leg(11, ground_contact=False)])
            )
        )

    def test_complete_leg_fixtures_must_fit_inside_pad(self):
        env = make_env([make_leg(8.2), make_leg(11.8)])
        actual = landing_leg_x_extents(env)
        expected = [(8.0, 8.4), (11.6, 12.0)]
        for actual_extent, expected_extent in zip(actual, expected):
            for actual_x, expected_x in zip(actual_extent, expected_extent):
                self.assertAlmostEqual(actual_x, expected_x)
        self.assertTrue(is_successful_landing_in_zone(env))

        outside = make_env([make_leg(9.0), make_leg(11.81)])
        self.assertFalse(is_successful_landing_in_zone(outside))

    def test_unstable_touchdown_is_not_a_zone_success(self):
        env = make_env(
            [make_leg(9.0), make_leg(11.0, ground_contact=False)]
        )
        self.assertFalse(is_successful_landing_in_zone(env))


if __name__ == "__main__":
    unittest.main()
