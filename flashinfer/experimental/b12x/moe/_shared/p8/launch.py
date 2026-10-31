"""The launch configuration one P8 call runs with.

The kernels accept their tile, residual-carrier stride, FC1 pipeline depth and
warp count as parameters, and the resident-grid cap as a launch argument. One
``P8LaunchConfig`` fixes all of them, so a plan can race whole configurations
the way it races the other b12x components' knobs, and the state that
materializes the winner carries it into every call and every captured replay.

A zero (or ``None``) axis keeps the runtime's serving policy for that axis,
which is the configuration the GLM-5.3 Flash serving image runs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

__all__ = ["P8_LAUNCH_AXES", "P8LaunchConfig"]

#: Axes whose value specializes a compiled program.
COMPILE_AXES = (
    "direct_tile_m",
    "grouped_tile_m",
    "fc1_pipeline_stages",
    "fc1_warps",
    "rp2_stride",
)
#: Axes read only at launch, so racing them compiles nothing new.
RUNTIME_AXES = ("mac",)
P8_LAUNCH_AXES = ("backend",) + COMPILE_AXES + RUNTIME_AXES

#: Direct RP2 tile M values the kernels accept; 0 keeps the tile policy.
_DIRECT_TILES = (0, 16, 32)
#: Grouped tile M values the kernels accept; 0 keeps the tile policy.
_GROUPED_TILES = (0, 32, 64)
_STRIDES = (128, 192)


@dataclass(frozen=True, kw_only=True)
class P8LaunchConfig:
    """One complete P8 launch configuration.

    ``direct_tile_m`` and ``grouped_tile_m`` are the tile M of the RP2 direct
    arm and of the grouped arm (0: the tile policy's choice for the row count).
    ``fc1_pipeline_stages`` and ``fc1_warps`` specialize the direct arm's FC1
    owner, ``rp2_stride`` the distance between its high and residual carriers,
    and ``mac`` the resident-grid cap (0: the runtime's own cap, which is its
    grid policy, or the cap it pins at construction).
    """

    backend: str = "trellismx_p8"
    direct_tile_m: int = 0
    grouped_tile_m: int = 0
    fc1_pipeline_stages: int = 2
    fc1_warps: int = 4
    rp2_stride: int = 192
    mac: int = 0

    def __post_init__(self) -> None:
        if self.backend != "trellismx_p8":
            raise ValueError(f"unsupported P8 backend {self.backend!r}")
        for name in P8_LAUNCH_AXES[1:]:
            if type(getattr(self, name)) is not int:
                raise TypeError(f"P8 launch axis {name} must be an integer")
        if self.direct_tile_m not in _DIRECT_TILES:
            raise ValueError(
                f"direct tile M must be one of {_DIRECT_TILES}; got {self.direct_tile_m}"
            )
        if self.grouped_tile_m not in _GROUPED_TILES:
            raise ValueError(
                f"grouped tile M must be one of {_GROUPED_TILES}; "
                f"got {self.grouped_tile_m}"
            )
        if self.fc1_pipeline_stages not in (2, 3):
            raise ValueError("FC1 pipeline supports two or three stages")
        if self.fc1_warps not in (4, 8):
            raise ValueError("FC1 supports four or eight warps")
        if self.rp2_stride not in _STRIDES:
            raise ValueError(
                f"RP2 carrier stride must be one of {_STRIDES}; got {self.rp2_stride}"
            )
        if self.mac < 0:
            raise ValueError("the resident-grid cap must be non-negative")

    @classmethod
    def serving(cls) -> "P8LaunchConfig":
        """The GLM-5.3 Flash serving configuration: every axis at its policy."""

        return cls()

    @classmethod
    def from_config(cls, payload) -> "P8LaunchConfig":
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise ValueError(
                "P8 launch config fields must be exactly "
                f"{sorted(expected)}; got {sorted(payload)}"
            )
        return cls(**{name: payload[name] for name in expected})

    def to_dict(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def compile_identity(self) -> tuple:
        """What the config changes in a compiled program (no launch arguments)."""

        return tuple(getattr(self, name) for name in COMPILE_AXES)
