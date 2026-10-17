def is_stable_landing(base_env) -> bool:
    return (
        not base_env.game_over
        and not base_env.lander.awake
        and all(leg.ground_contact for leg in base_env.legs)
    )


def landing_leg_x_extents(base_env) -> list[tuple[float, float]]:
    extents = []
    for leg in base_env.legs:
        world_x_positions = [
            float((leg.transform * vertex).x)
            for fixture in leg.fixtures
            for vertex in getattr(fixture.shape, "vertices", ())
        ]
        if not world_x_positions:
            return []
        extents.append((min(world_x_positions), max(world_x_positions)))
    return extents


def is_successful_landing_in_zone(base_env) -> bool:
    if not is_stable_landing(base_env):
        return False

    leg_extents = landing_leg_x_extents(base_env)
    if len(leg_extents) != len(base_env.legs):
        return False

    pad_left = float(base_env.helipad_x1) - 1e-6
    pad_right = float(base_env.helipad_x2) + 1e-6
    return all(
        pad_left <= left <= right <= pad_right
        for left, right in leg_extents
    )
