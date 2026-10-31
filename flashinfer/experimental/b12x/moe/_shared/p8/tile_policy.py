"""The P8 serving tile and resident-grid policy.

These are the values the GLM-5.3 Flash serving image launches with. They are
one candidate of the launch configuration a plan races
(:class:`b12x.moe._shared.p8.launch.P8LaunchConfig`): a plan may pin another
tile or another resident-grid cap, and the race decides. Nothing here is a
fixed choice of the runtime. ``resolve_grid_cap`` is where the cap's owners
meet: an explicit constructor pin wins, then the raced configuration, then
this policy.
"""


def select_tile(expected_m):
    """The serving tile M of a launch of ``expected_m`` rows."""

    if type(expected_m) is not int or expected_m < 1:
        raise ValueError("expected_m must be a positive integer")
    return (16 if expected_m <= 1 else 32 if expected_m <= 128 else 64, 128)


def row_tiles(valid_rows, tile_m):
    if type(valid_rows) is not int or valid_rows < 0 or tile_m not in (16, 32, 64):
        raise ValueError("invalid row tiling")
    return tuple(
        (start, min(tile_m, valid_rows - start))
        for start in range(0, valid_rows, tile_m)
    )


def direct_grid_capacity(m: int, default: int = 64) -> int:
    """The serving resident-grid cap of a direct or grouped launch of ``m`` rows."""

    if m > 16:
        return 128
    return {4: 192, 8: 128, 16: 192}.get(m, default)


def resolve_grid_cap(
    m: int,
    *,
    arm: int,
    config: int = 0,
    policy: bool = True,
    pin: int | None = None,
) -> int:
    """The resident-grid cap one launch uses.

    An explicit constructor pin wins over everything, so a raced cap can never
    change it. Otherwise a nonzero cap from the launch configuration (``config``,
    the race's winner) wins, and ``mac=0`` falls back to the runtime's grid
    policy (``policy``) or, with the policy off, to the arm's own cap (``arm``).
    """

    if pin is not None:
        return int(pin)
    if config:
        return int(config)
    if policy:
        return direct_grid_capacity(int(m), int(arm))
    return int(arm)
