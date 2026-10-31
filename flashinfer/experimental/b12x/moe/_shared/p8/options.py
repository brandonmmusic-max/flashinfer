"""Construction options of the in-tree P8 runtime (:class:`P8NativeTPMoE`).

The runtime resolves every kernel, schedule and storage choice once, when it is
constructed. In the serving image it read part of them from the process
environment (``B12X_P8_*``, plus ``GLM53_P8_*`` fallbacks for three FC1/grid
options); here all of them are fields of one immutable options object, so a
runtime's behavior is fully described by its arguments.

The defaults are the GLM-5.3 Flash serving configuration:
``B12X_P8_DOWN_REMAINDER=rp2``, ``B12X_P8_DOWN_REMAINDER_PHASES=both``,
``B12X_P8_EPI_PAR=1``, ``B12X_P8_RP2_DIRECT_MAX_TOKENS=16``,
``B12X_P8_RP2_SKIP_INPUT_CLEAR=1`` and the constructor options
``small_m_scheduler=True``, ``fc1_tile_n=128``, ``prefill_chunk_tokens=0``,
``grid_policy=True``, ``fc1_warp_quant=False``, ``fc1_broadcast_a=True``.
Every other field keeps the runtime's own constructor default.

The launch geometry (direct and grouped tile, carrier stride, FC1 pipeline
depth and warp count) is not an option: it is the per-call
:class:`b12x.moe._shared.p8.launch.P8LaunchConfig` a prepared plan races, whose
serving value reproduces the GLM-5.3 Flash defaults. A call names one; a
runtime's own geometry is that serving value.

The resident-grid cap has one meaning per owner. ``grid_policy`` is what a
launch configuration's ``mac=0`` selects: the measured per-row-count capacity.
``mac_override`` pins a cap, and a pin wins over any raced cap, so a race never
changes it (``TrellisMxP8Query.mac_pin`` keeps such a runtime's race off the
cap axis).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

__all__ = ["DOWN_REMAINDER_MODES", "P8RuntimeOptions", "GLM53_SERVING_OPTIONS"]

#: ``down_remainder`` values and the ``B12X_P8_DOWN_REMAINDER`` value each
#: replaces: ``"none"`` (unset), ``"dx2"`` (``1``: two-term carrier at the
#: FC2 input), ``"rp"`` (row-packed remainder at the down hop of the direct
#: kernels), ``"rp2"`` (row-packed remainder at both hops).
DOWN_REMAINDER_MODES = ("none", "dx2", "rp", "rp2")
_PHASES = ("both", "fc1", "fc2")


@dataclass(frozen=True, kw_only=True)
class P8RuntimeOptions:
    """Every behavior setting of the P8 runtime; the defaults are the serving configuration."""

    #: Down-hop remainder carrier (was ``B12X_P8_DOWN_REMAINDER``).
    down_remainder: str = "rp2"
    #: Diagnostic split of the ``"dx2"`` carrier (was
    #: ``B12X_P8_DOWN_REMAINDER_PHASES``); read only with ``"dx2"``.
    down_remainder_phases: str = "both"
    #: Largest token count kept on the direct RP2 kernels (was
    #: ``B12X_P8_RP2_DIRECT_MAX_TOKENS``); 16 is the floor.
    rp2_direct_max_tokens: int = 16
    #: Leave the input carrier uninitialized where the direct RP2 prologue
    #: overwrites every byte it reads (was ``B12X_P8_RP2_SKIP_INPUT_CLEAR``).
    rp2_skip_input_clear: bool = True
    #: Parallel small-M FC1 epilogue (was ``B12X_P8_EPI_PAR``).
    epi_par: bool = True
    small_m_scheduler: bool = True
    fc1_tile_n: int = 128
    prefill_chunk_tokens: int = 0
    #: The measured direct/grouped launch capacities a launch configuration's
    #: ``mac=0`` selects (was ``GLM53_P8_GRID_POLICY`` when not passed).
    grid_policy: bool = True
    #: Warp-level FC1 quantization (was ``GLM53_P8_FC1_WARP_QUANT`` when not
    #: passed).
    fc1_warp_quant: bool = False
    #: Broadcast the single A row in the direct FC1 owner (was
    #: ``GLM53_P8_FC1_BROADCAST_A`` when not passed).
    fc1_broadcast_a: bool = True
    force_materialized: bool | None = None
    #: A pinned resident-grid cap; wins over the raced configuration's ``mac``.
    mac_override: int | None = None
    deterministic_output: bool = True
    debug_capture: bool = False
    compact_scale_storage: bool = False
    compact_input_storage: bool = False
    tile_major_tasks: bool = False
    fc1_a_swizzle_rotate: bool = False
    fc1_exact_staging: bool = False

    def __post_init__(self) -> None:
        for spec in fields(self):
            value = getattr(self, spec.name)
            optional = "None" in str(spec.type)
            if value is None and optional:
                continue
            if "bool" in str(spec.type):
                if type(value) is not bool:
                    raise TypeError(
                        f"P8 option {spec.name} must be a bool"
                        f"{' or None' if optional else ''}; got {value!r}"
                    )
            elif "int" in str(spec.type):
                if type(value) is not int:
                    raise TypeError(
                        f"P8 option {spec.name} must be an int"
                        f"{' or None' if optional else ''}; got {value!r}"
                    )
            elif type(value) is not str:
                raise TypeError(f"P8 option {spec.name} must be a str; got {value!r}")
        if self.down_remainder not in DOWN_REMAINDER_MODES:
            raise ValueError(
                f"down_remainder must be one of {DOWN_REMAINDER_MODES}; "
                f"got {self.down_remainder!r}"
            )
        if self.down_remainder_phases not in _PHASES:
            raise ValueError(
                f"down_remainder_phases must be one of {_PHASES}; "
                f"got {self.down_remainder_phases!r}"
            )

    def as_dict(self) -> dict[str, object]:
        return {spec.name: getattr(self, spec.name) for spec in fields(self)}


#: The GLM-5.3 Flash serving configuration.
GLM53_SERVING_OPTIONS = P8RuntimeOptions()
