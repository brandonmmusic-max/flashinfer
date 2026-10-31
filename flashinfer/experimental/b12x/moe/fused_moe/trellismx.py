"""TrellisMX (procedural NPK) expert sidecars as a fused-MoE weight source.

TrellisMX is a *weight source format* under the existing fused-MoE API, not a
new operator: a released sidecar carries the assembled ``int16`` trellis
carriers, the UE8M0 K32 weight scales (plain for GLM-5.3 Flash, lossless CSF
slabs for DeepSeek-V4.1 Flash), the per-expert coupled-boundary rotations and
the per-hidden-channel ``suh``/``svh`` side vectors.  Preparation validates the
release metadata, maps every tensor into the W4A8 trellis prepared layout and
hands the result to :func:`plan_weights` / :func:`prepare_weights` exactly like
any other packed checkpoint.

Two invariants hold throughout:

* no codeword and no scale is numerically converted.  The carriers are the
  output of ``assemble_uniform_slots`` and are consumed as they are, so the
  SHA-256 of the carrier bytes is preserved bit for bit;
* the only permitted scale operation is the lossless *storage* expansion of a
  CSF slab into plain UE8M0 bytes, which is byte-exact and tested.

The compressed-scale container is the same one b12x already reads for
DeepSeek MXFP4 expert scales: ``CsfSource(encoding="csf")`` with
``quantization="mxfp4"``.  Its DS41 spelling is
``mxfp4-csf-row-base-offset1-u24-exceptions-1``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import torch

from .config import (
    RateGranularity,
    ScaleGranularity,
    TrellisCodebook,
    TrellisConfig,
    TrellisExpertTransform,
    TrellisProjectionTransform,
    TrellisRateConfig,
    TrellisScaleConfig,
    TrellisScaleFactorsConfig,
    TrellisTransformConfig,
)
from .planning import (
    TRELLISMX_SWIGLU_LIMIT,
    ActivationMode,
    ActivationSpec,
    MoEGeometry,
    WeightPlanConstraints,
    plan_weights,
    prepare_weights,
)
from .source import TrellisExtent, TrellisSource
from .weights import (
    PreparedExperts,
    ScaleFactors,
    WeightPacking,
)

#: The owner's codec law.  ``procedural-mcg-alpha2`` is the older label GLM
#: sidecars carry for the same law (compander 2.0); it is accepted as an alias
#: and never as a second law.
LAW_PROCEDURAL_NPK = "procedural-npk"
LAW_ALIASES = frozenset({LAW_PROCEDURAL_NPK, "procedural-mcg-alpha2"})
COMPANDER_SCALE = 2.0
ALPHABET = "e4m3"
SCALE_GRID = "ue8m0-k32"
BOUNDARY = "coupled-h512-h128-suh-svh-v1"
ACTIVATION = "silu-cap10"
SUPPORTED_BITS = (2, 3, 4)
SLOT_CHANNELS = 32

#: Projection block of the coupled boundary's outer (hidden) Hadamard stage.
PROJECTION_BLOCK = 512
#: Pre/post blocks of the coupled boundary's per-expert (intermediate) stage.
EXPERT_BLOCKS = (128, 128)

#: Sidecar tensor names, by container family.
CARRIER_NAMES = ("w13_trellis", "w2_trellis")
PLAIN_SCALE_NAMES = ("w13_scale_ue8m0", "w2_scale_ue8m0")
CSF_SCALE_NAMES = ("w13_scale.mxfp4_csf_fixed", "w2_scale.mxfp4_csf_fixed")
CSF_EXCEPTION_NAMES = (
    "w13_exceptions",
    "w13_exception_offsets",
    "w2_exceptions",
    "w2_exception_offsets",
)
BOUNDARY_NAMES = (
    "intermediate_scales_fp16",
    "gate_up_suh_fp16",
    "down_svh_fp16",
)
OPTIONAL_NAMES = ("coupled_sign_draw_u8",)


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


def _field(
    metadata: Mapping[str, object], *names: str, default: object = ...
) -> object:
    for name in names:
        if name in metadata:
            return metadata[name]
    if default is not ...:
        return default
    raise ValueError(
        "TrellisMX metadata is missing required field "
        + " / ".join(repr(name) for name in names)
    )


def _int_field(
    metadata: Mapping[str, object], *names: str, default: object = ...
) -> int:
    value = _field(metadata, *names, default=default)
    if isinstance(value, bool):
        raise ValueError(f"TrellisMX metadata {names[0]!r} must be an integer")
    try:
        return int(str(value)) if isinstance(value, str) else int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"TrellisMX metadata {names[0]!r} must be an integer; got {value!r}"
        ) from exc


def _float_field(
    metadata: Mapping[str, object], *names: str, default: object = ...
) -> float:
    value = _field(metadata, *names, default=default)
    try:
        return float(str(value)) if isinstance(value, str) else float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"TrellisMX metadata {names[0]!r} must be a number; got {value!r}"
        ) from exc


def _text_field(
    metadata: Mapping[str, object], *names: str, default: object = ...
) -> str:
    value = _field(metadata, *names, default=default)
    if not isinstance(value, str):
        raise ValueError(
            f"TrellisMX metadata {names[0]!r} must be a string; got {value!r}"
        )
    return value.strip()


@dataclass(frozen=True)
class TrellisMxMetadata:
    """Validated release metadata for one rank-local TrellisMX layer.

    Constructed only through :meth:`from_sidecar`, which is where every
    release gate lives: a wrong law, compander, alphabet, scale grid,
    boundary, bitrate or geometry raises ``ValueError`` naming the field.
    """

    law: str
    compander_scale: float
    alphabet: str
    scale: str
    boundary: str
    activation: str
    bits: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    global_intermediate_size: int
    first_slot: int
    slot_count: int
    top_k: int
    scale_storage: str | None = None

    @classmethod
    def from_sidecar(
        cls,
        metadata: Mapping[str, object],
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        global_intermediate_size: int | None = None,
    ) -> "TrellisMxMetadata":
        if not isinstance(metadata, Mapping):
            raise TypeError("TrellisMX metadata must be a mapping")
        law = _text_field(metadata, "law")
        if law not in LAW_ALIASES:
            raise ValueError(
                f"TrellisMX law {law!r} is not {LAW_PROCEDURAL_NPK!r} (or the "
                "older label 'procedural-mcg-alpha2'); refusing to interpret "
                "another codec"
            )
        # GLM-5.3 sidecars carry the older ``procedural-mcg-alpha2`` label and
        # no compander field: that label *is* the alpha-2 compander, so 2.0 is
        # implied. The canonical ``procedural-npk`` id must state it, and a
        # present value other than 2.0 is always rejected.
        if "compander_scale" in metadata:
            compander = _float_field(metadata, "compander_scale")
            if compander != COMPANDER_SCALE:
                raise ValueError(
                    "TrellisMX compander_scale must be 2.0; got "
                    f"{compander!r} (a different compander is a different law)"
                )
        elif law == "procedural-mcg-alpha2":
            compander = COMPANDER_SCALE
        else:
            raise ValueError(
                "TrellisMX metadata is missing required field 'compander_scale'; "
                f"law {law!r} does not imply it"
            )
        alphabet = _text_field(metadata, "alphabet")
        if alphabet != ALPHABET:
            raise ValueError(
                f"TrellisMX alphabet must be {ALPHABET!r}; got {alphabet!r}"
            )
        scale = _text_field(metadata, "scale")
        if scale != SCALE_GRID:
            raise ValueError(f"TrellisMX scale must be {SCALE_GRID!r}; got {scale!r}")
        boundary = _text_field(metadata, "boundary")
        if boundary != BOUNDARY:
            raise ValueError(
                f"TrellisMX boundary must be {BOUNDARY!r}; got {boundary!r}"
            )
        activation = _text_field(metadata, "activation", default=ACTIVATION)
        if activation != ACTIVATION:
            raise ValueError(
                f"TrellisMX activation must be {ACTIVATION!r}; got {activation!r}"
            )
        bits = _int_field(metadata, "bits")
        if bits not in SUPPORTED_BITS:
            raise ValueError(
                f"TrellisMX bits must be one of {list(SUPPORTED_BITS)}; got {bits}"
            )

        # ---- geometry -----------------------------------------------------
        # The caller declares the rank-local shape it is loading; a container
        # that also records it (DS41 manifests/sidecars) is cross-checked, and
        # a container that does not (GLM-5.3 sidecars) is trusted from the
        # declaration. Either way a disagreement raises.
        def _opt_int(*names: str) -> int | None:
            value = _field(metadata, *names, default=None)
            if value is None:
                return None
            if isinstance(value, bool):
                raise ValueError(f"TrellisMX metadata {names[0]!r} must be an integer")
            try:
                return int(str(value)) if isinstance(value, str) else int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"TrellisMX metadata {names[0]!r} must be an integer; got {value!r}"
                ) from exc

        def _agrees(declared: int, *names: str) -> int:
            value = _opt_int(*names)
            if value is not None and value != int(declared):
                raise ValueError(
                    f"TrellisMX metadata {names[0]}={value} does not match the "
                    f"declared {names[0] if len(names) == 1 else names[1]}="
                    f"{int(declared)}"
                )
            return int(declared)

        hidden = _agrees(hidden_size, "hidden", "hidden_size")
        experts = _agrees(num_experts, "experts_end", "num_experts")
        experts_start = _opt_int("experts_start") or 0
        if experts_start != 0:
            raise ValueError(
                "TrellisMX expert-parallel sharding is not a weight-format "
                f"feature; experts_start must be 0, got {experts_start}"
            )
        local = _agrees(intermediate_size, "intermediate_per_rank", "intermediate_size")
        topk = _agrees(top_k, "topk", "top_k")
        if topk <= 0 or topk > experts:
            raise ValueError(f"TrellisMX topk={topk} must be in [1, {experts}]")
        world_size = _opt_int("world_size") or 1
        if world_size <= 0:
            raise ValueError(f"TrellisMX world_size must be positive, got {world_size}")
        slice_begin = _opt_int("slice_begin")
        if slice_begin is None:
            # GLM sidecars record the rank's first intermediate *atom* (a
            # 32-channel slot index), not a channel index: layer 3 rank 2
            # carries 32, which is channel 1024, not channel 32.
            atom_begin = _opt_int("local_atom_begin")
            slice_begin = 0 if atom_begin is None else SLOT_CHANNELS * atom_begin
        slice_end = _opt_int("slice_end")
        if slice_end is not None and slice_end - slice_begin != local:
            raise ValueError(
                "TrellisMX slice_begin/slice_end span "
                f"{slice_end - slice_begin} channels, not intermediate_per_rank={local}"
            )
        if slice_begin % SLOT_CHANNELS or local % SLOT_CHANNELS:
            raise ValueError(
                "TrellisMX rank slices must contain whole "
                f"{SLOT_CHANNELS}-channel slots; got begin={slice_begin}, "
                f"width={local}"
            )
        if global_intermediate_size is not None:
            global_intermediate = int(global_intermediate_size)
        else:
            recorded = _opt_int("global_intermediate")
            global_intermediate = world_size * local if recorded is None else recorded
        if global_intermediate % SLOT_CHANNELS:
            raise ValueError(
                "TrellisMX global_intermediate_size must be a multiple of "
                f"{SLOT_CHANNELS}; got {global_intermediate}"
            )
        first_slot = slice_begin // SLOT_CHANNELS
        slot_count = local // SLOT_CHANNELS
        if SLOT_CHANNELS * (first_slot + slot_count) > global_intermediate:
            raise ValueError(
                "TrellisMX rank extent "
                f"[{first_slot}, {first_slot + slot_count}) slots exceeds the "
                f"global intermediate axis ({global_intermediate} channels)"
            )
        storage = metadata.get("scale_storage")
        return cls(
            law=law,
            compander_scale=compander,
            alphabet=alphabet,
            scale=scale,
            boundary=boundary,
            activation=activation,
            bits=bits,
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size=local,
            global_intermediate_size=global_intermediate,
            first_slot=first_slot,
            slot_count=slot_count,
            top_k=topk,
            scale_storage=None if storage is None else str(storage),
        )

    def config(self) -> TrellisConfig:
        """The v2 ``TrellisConfig`` this release declares."""

        return TrellisConfig(
            version=2,
            codebook=TrellisCodebook.PROCEDURAL_NPK,
            rate=TrellisRateConfig(granularity=RateGranularity.UNIFORM),
            scale=TrellisScaleConfig(
                input_scales=TrellisScaleFactorsConfig(
                    vectors=ScaleGranularity.PER_LAYER,
                    gains=ScaleGranularity.NONE,
                ),
                intermediate_scales=TrellisScaleFactorsConfig(
                    vectors=ScaleGranularity.PER_EXPERT,
                    gains=ScaleGranularity.NONE,
                ),
                output_scales=TrellisScaleFactorsConfig(
                    vectors=ScaleGranularity.PER_LAYER,
                    gains=ScaleGranularity.NONE,
                ),
            ),
            transform=TrellisTransformConfig(
                projection=TrellisProjectionTransform(
                    kind="scaled_hadamard", block_size=PROJECTION_BLOCK
                ),
                expert=TrellisExpertTransform(
                    kind="intermediate_hadamard",
                    pre_block_size=EXPERT_BLOCKS[0],
                    post_block_size=EXPERT_BLOCKS[1],
                    sign_pattern_granularity="per_expert",
                ),
            ),
        )

    def source(self) -> TrellisSource:
        return TrellisSource(
            config=self.config(),
            uniform_bits=self.bits,
            extent=TrellisExtent(
                global_intermediate_size=self.global_intermediate_size,
                first_slot=self.first_slot,
                slot_count=self.slot_count,
            ),
            source_format="trellismx",
        )


# --------------------------------------------------------------------------
# CSF: lossless storage expansion
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _CsfSlab:
    """One ``mxfp4-csf-row-base-offset1-u24-exceptions-1`` scale slab."""

    plane: torch.Tensor  # uint8 [E, rows/16, 16 + 16*ceil(cols/8)]
    records: torch.Tensor  # uint32 [N]
    starts: torch.Tensor  # int64 [E + 1]
    rows: int
    columns: int


def _csf_slab(source: object, *, name: str) -> _CsfSlab:
    """Validate a DS41 CSF slab and return its three components."""

    from b12x.loader._csf_source import CsfSource

    if not isinstance(source, CsfSource):
        raise TypeError(f"{name} must be a CsfSource, got {type(source).__name__}")
    if source.encoding != "csf":
        raise NotImplementedError(
            f"{name} uses CSF encoding {source.encoding!r}; the TrellisMX "
            "sidecars use the lossless 'csf' slab encoding"
        )
    if source.quantization != "mxfp4":
        raise ValueError(
            f"{name} CSF quantization must be 'mxfp4'; got {source.quantization!r}"
        )
    if source.join != 1:
        raise NotImplementedError(f"{name} CSF join={source.join} is unsupported")
    for field in ("row_range", "column_range", "padded_rows", "padded_columns"):
        if getattr(source, field) is not None:
            raise NotImplementedError(
                f"{name} CSF {field} selection is unsupported for TrellisMX "
                "sidecars (their slabs are already rank-local)"
            )
    if source.padding_base != 0:
        raise ValueError(
            f"{name} CSF padding_base must be 0; got {source.padding_base}"
        )
    rows, columns = int(source.source_rows), int(source.source_columns)
    plane, records = source.components
    if plane.dtype != torch.uint8 or plane.ndim != 3:
        raise ValueError(f"{name} CSF plane must be uint8 [E, rows/16, stride]")
    if records.dtype != torch.uint32:
        raise ValueError(
            f"{name} CSF exception records must be uint32; got {records.dtype}"
        )
    if rows % 16 or columns <= 0:
        raise ValueError(f"{name} CSF geometry {rows}x{columns} is not 16-row aligned")
    stride = 16 * (1 + (columns + 7) // 8)
    if tuple(plane.shape) != (int(source.segments), rows // 16, stride):
        raise ValueError(
            f"{name} CSF slab must be uint8 "
            f"{(int(source.segments), rows // 16, stride)}; got "
            f"{tuple(plane.shape)}"
        )
    return _CsfSlab(
        plane=plane,
        records=records,
        starts=source.starts,
        rows=rows,
        columns=columns,
    )


def decode_csf_scale_plane(
    source: object, *, name: str = "scale", chunk: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand one CSF slab to ``uint8 [E, rows, columns]`` UE8M0 bytes.

    Returns ``(grid, bases)``.  ``bases`` is the slab's stored one-byte row
    base plane ``[E, rows]``; :func:`encode_csf_scale_plane` needs it to
    reproduce the container exactly.

    The law is ``_lib/quant/_csf_prepare.py:246-287`` (the ``LEGACY`` branch of
    ``_read``): the first 16 bytes of each 16-row group are the row bases, the
    remaining bytes hold one LSB-first bit field per row, and an exception
    record ``(position | value << 24)`` overrides the position
    ``row * columns + column``.  Every invariant the Triton validator checks
    is checked here, on the CPU, before any byte is trusted.
    """

    slab = _csf_slab(source, name=name)
    rows, columns = slab.rows, slab.columns
    plane, records, starts = slab.plane, slab.records, slab.starts
    segments = int(plane.shape[0])
    if starts.numel() != segments + 1:
        raise ValueError(
            f"{name} CSF exception starts must hold {segments + 1} entries; got "
            f"{starts.numel()}"
        )
    if int(starts[0]) != 0 or int(starts[-1]) != records.numel():
        raise ValueError(
            f"{name} CSF exception starts must span every record "
            f"(0..{records.numel()}); got {int(starts[0])}..{int(starts[-1])}"
        )
    sizes = (starts[1:] - starts[:-1]).to(torch.int64)
    if bool((sizes < 0).any()):
        raise ValueError(f"{name} CSF exception segment sizes must be nonnegative")

    codes_per_byte = 8
    bytes_per_row = (columns + codes_per_byte - 1) // codes_per_byte
    bases = plane[:, :, :16].reshape(segments, rows).contiguous()
    if int(bases.max()) > 255:
        raise ValueError(f"{name} CSF row bases must fit one byte")
    code_bytes = plane[:, :, 16:].reshape(segments, rows, bytes_per_row)
    # Every helper tensor follows the slab's device: a sidecar staged on CUDA
    # must decode without host round trips (and without a mixed-device error).
    shifts = torch.arange(codes_per_byte, dtype=torch.int16, device=code_bytes.device)
    code_bits = (code_bytes.to(torch.int16).unsqueeze(-1) >> shifts) & 1
    codes = code_bits.reshape(segments, rows, bytes_per_row * codes_per_byte)
    if bytes_per_row * codes_per_byte != columns:
        codes = codes[:, :, :columns]
    grid = torch.empty(segments, rows, columns, dtype=torch.uint8, device=plane.device)

    positions = records.to(torch.int64)
    record_pos = positions & 0xFFFFFF
    record_val = ((positions >> 24) & 0xFF).to(torch.uint8)
    for e0 in range(0, segments, max(1, int(chunk))):
        e1 = min(segments, e0 + max(1, int(chunk)))
        block = bases[e0:e1].unsqueeze(-1) + codes[e0:e1]
        if int(block.max()) > 255:
            raise ValueError(
                f"{name} CSF decoded scales exceed one UE8M0 byte; the row "
                "bases or bit fields are not a mxfp4 slab"
            )
        grid[e0:e1] = block.to(torch.uint8)
        for e in range(e0, e1):
            lo, hi = int(starts[e]), int(starts[e + 1])
            if hi == lo:
                continue
            pos = record_pos[lo:hi]
            if int(pos[0]) < 0 or int(pos[-1]) >= rows * columns:
                raise ValueError(
                    f"{name} CSF exception position {int(pos[-1])} is outside "
                    f"the {rows}x{columns} expert grid"
                )
            if bool((pos[1:] <= pos[:-1]).any()):
                raise ValueError(
                    f"{name} CSF exception positions must be sorted and unique "
                    f"(expert {e})"
                )
            grid[e].reshape(-1)[pos] = record_val[lo:hi]
    return grid, bases


def encode_csf_scale_plane(
    grid: torch.Tensor, bases: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-encode a decoded grid into the CSF slab given its stored bases.

    Inverse of :func:`decode_csf_scale_plane` for every grid that came from a
    valid slab: a position is representable iff ``value - base`` is 0 or 1,
    the code field stores that difference (0 otherwise, which the decoder
    ignores at exception positions), and every non-representable position
    emits one sorted record ``position | value << 24``.
    """

    if grid.dtype != torch.uint8 or grid.ndim != 3:
        raise ValueError("CSF encode requires a uint8 [E, rows, columns] grid")
    segments, rows, columns = grid.shape
    if tuple(bases.shape) != (segments, rows) or bases.dtype != torch.uint8:
        raise ValueError(
            f"CSF encode requires uint8 bases {(segments, rows)}; got "
            f"{bases.dtype} {tuple(bases.shape)}"
        )
    delta = grid.to(torch.int16) - bases.to(torch.int16).unsqueeze(-1)
    representable = (delta == 0) | (delta == 1)
    code = torch.where(representable, delta, torch.zeros_like(delta)).to(torch.uint8)
    bytes_per_row = (columns + 7) // 8
    # A row of code bits need not fill its last byte (DS41 K2's w2 plane has
    # 36 columns = 4.5 bytes); the unused high bits of that byte are written as
    # zero, which is what the released slabs carry.
    padded = torch.zeros(
        segments, rows, bytes_per_row * 8, dtype=torch.uint8, device=grid.device
    )
    padded[:, :, :columns] = code
    packed = (
        padded.reshape(segments, rows, bytes_per_row, 8).to(torch.int64)
        << torch.arange(8, device=grid.device)
    ).sum(-1)
    plane = torch.zeros(
        segments,
        rows // 16,
        16 + 16 * bytes_per_row,
        dtype=torch.uint8,
        device=grid.device,
    )
    plane[:, :, :16] = bases.reshape(segments, rows // 16, 16)
    plane[:, :, 16:] = packed.to(torch.uint8).reshape(
        segments, rows // 16, 16 * bytes_per_row
    )
    records = []
    for e in range(segments):
        positions = (~representable[e]).reshape(-1).nonzero().squeeze(-1)
        if positions.numel():
            values = grid[e].reshape(-1)[positions].to(torch.int64) << 24
            records.append(values | positions)
    arena = (
        torch.cat(records).to(torch.uint32)
        if records
        else torch.zeros(0, dtype=torch.uint32, device=grid.device)
    )
    return plane, arena


def csf_scale_source(
    plane: torch.Tensor,
    records: torch.Tensor,
    starts: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> object:
    """Build the ``CsfSource`` for one DS41 sidecar slab."""

    from b12x.loader._csf_source import CsfSource

    segments = int(plane.shape[0])
    source = CsfSource(
        (plane, records),
        starts.to(torch.int64),
        int(rows),
        int(columns),
        "mxfp4",
        segments,
        "csf",
    )
    source.validate()
    return source


# --------------------------------------------------------------------------
# Tensor container
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrellisMxBoundaryOps:
    """The coupled boundary as prepared ops of one TrellisMX layer.

    Every op is pure torch, static-shape, host-sync free and never writes in
    place, so a prepared instance is CUDA-graph capturable and replays
    bitwise.  The arithmetic lives in
    :mod:`b12x.moe._shared.kernels.trellismx_boundary`; this object only
    binds it to the released per-layer vectors, which is what makes the
    boundary part of the *prepared* contract rather than a caller's choice.

    The declared order is the release's:

    * input — ``bf16 -> fp16 -> H512 -> gate_up_suh -> H128 -> E4M3/UE8M0 K32``;
    * output — per route ``FP16 FC2 -> H128 -> down_svh``, then the
      router-order FP32 top-k sum and the outer ``H512``.

    ``input_boundary_quantized`` returns the kernel's A operands
    (``packed_a_storage``/``scale_storage`` layout).  A kernel that consumes
    them must skip its own activation producer, otherwise the H128 stage runs
    twice.
    """

    hidden_size: int
    intermediate_size: int
    bits: int
    gate_up_suh: torch.Tensor
    down_svh: torch.Tensor

    def __post_init__(self) -> None:
        from b12x.moe._shared.kernels.trellismx_boundary import (
            TrellisMxBoundarySpec,
        )

        TrellisMxBoundarySpec(
            hidden_size=int(self.hidden_size),
            intermediate_size=int(self.intermediate_size),
            bits=int(self.bits),
        )
        for name in ("gate_up_suh", "down_svh"):
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float16:
                raise TypeError(f"TrellisMX boundary {name} must be fp16")
            if tuple(tensor.shape) != (int(self.hidden_size),):
                raise ValueError(
                    f"TrellisMX boundary {name} must have shape "
                    f"({int(self.hidden_size)},); got {tuple(tensor.shape)}"
                )

    # -- input stage -------------------------------------------------------

    def input_boundary(self, x_bf16: torch.Tensor) -> torch.Tensor:
        """``H512 -> suh -> H128`` in FP32, before quantization."""

        from b12x.moe._shared.kernels.trellismx_boundary import input_boundary

        return input_boundary(
            x_bf16, self.gate_up_suh, hidden_size=int(self.hidden_size)
        )

    def input_boundary_pre_h128(self, x_bf16: torch.Tensor) -> torch.Tensor:
        """``H512 -> suh`` in FP32: the stage a kernel that owns H128 expects.

        A normalized Sylvester Hadamard is its own inverse, so feeding this to
        a kernel whose activation producer applies H128 yields the declared
        operand without a kernel-side switch.  The extra BF16 store on the way
        in is a deviation from the declared order and must be stated by the
        caller.
        """

        from b12x.moe._shared.kernels.trellismx_boundary import (
            input_boundary_pre_h128,
        )

        return input_boundary_pre_h128(
            x_bf16, self.gate_up_suh, hidden_size=int(self.hidden_size)
        )

    def outer_h512(self, value: torch.Tensor) -> torch.Tensor:
        """The coupled reducer's outer stage, for a kernel that already summed."""

        from b12x.moe._shared.kernels.trellismx_boundary import h512_normalized

        return h512_normalized(value)

    def input_boundary_quantized(
        self, x_bf16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The kernel's A operands: ``(payload uint8 [tokens,H], scales uint8)``.

        On CUDA this is ONE fused kernel launch
        (:mod:`b12x.moe._shared.kernels.trellismx_boundary_fused`), 9.7x faster
        than the ~40-launch torch chain and byte-exact to that chain evaluated
        on the CPU, which divides exactly as the kernel does (the CUDA chain
        multiplies by reciprocals, so it can differ by an fp32 ulp). On CPU the
        torch op is used, and it stays the test oracle.
        """

        from b12x.moe._shared.kernels.trellismx_boundary_fused import (
            fused_input_boundary_quantized,
        )

        return fused_input_boundary_quantized(
            x_bf16, self.gate_up_suh, hidden_size=int(self.hidden_size)
        )

    def input_boundary_fp16(self, x_bf16: torch.Tensor) -> torch.Tensor:
        """The unquantized input stage as FP16 (fused kernel on CUDA)."""

        from b12x.moe._shared.kernels.trellismx_boundary_fused import (
            fused_input_boundary_fp16,
        )

        return fused_input_boundary_fp16(
            x_bf16, self.gate_up_suh, hidden_size=int(self.hidden_size)
        )

    # -- output stage ------------------------------------------------------

    def output_boundary(self, route_fp16: torch.Tensor) -> torch.Tensor:
        """Per-route ``H128 -> *down_svh`` in FP32, before the reduction."""

        from b12x.moe._shared.kernels.trellismx_boundary import output_boundary

        return output_boundary(
            route_fp16, self.down_svh, hidden_size=int(self.hidden_size)
        )

    def coupled_topk_sum(
        self, routes: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Router-order FP32 weighted sum, then the outer ``H512``, to BF16."""

        from b12x.moe._shared.kernels.trellismx_boundary import coupled_topk_sum

        return coupled_topk_sum(routes, weights, hidden_size=int(self.hidden_size))

    def apply_output(
        self,
        route_fp16: torch.Tensor,
        weights: torch.Tensor,
        *,
        top_k: int,
    ) -> torch.Tensor:
        """The whole output stage: per-route FP16 result in, BF16 tokens out."""

        hidden = int(self.hidden_size)
        if route_fp16.ndim != 2 or route_fp16.shape[-1] != hidden:
            raise ValueError(
                f"route_fp16 must be [tokens*top_k, {hidden}]; got "
                f"{tuple(route_fp16.shape)}"
            )
        if route_fp16.dtype != torch.float16:
            raise TypeError(
                "the FC2 result must reach the output boundary as fp16 (the "
                f"declared store); got {route_fp16.dtype}"
            )
        routes = self.output_boundary(route_fp16)
        return self.coupled_topk_sum(routes.view(-1, int(top_k), hidden), weights)


@dataclass(frozen=True)
class TrellisMxWeights:
    """Rank-local TrellisMX tensors, exactly as the release stores them.

    ``w13``/``w2`` are the assembled ``int16`` carriers (the output of
    ``assemble_uniform_slots``), never the slot-major uint8 codes, so no
    permutation happens between the checkpoint and the kernel.
    ``w13_scale``/``w2_scale`` are either the plain UE8M0 plane
    ``[E, rows, K/32]`` or a ``CsfSource`` whose slab is expanded losslessly
    by preparation.
    """

    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: object
    w2_scale: object
    input_scales: ScaleFactors
    intermediate_scales: ScaleFactors
    output_scales: ScaleFactors
    expert_sign_patterns: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("w13", "w2"):
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"TrellisMxWeights.{name} must be a torch.Tensor")
            if tensor.dtype != torch.int16:
                raise TypeError(
                    f"TrellisMxWeights.{name} must be the released int16 "
                    f"carrier; got {tensor.dtype}"
                )
        if not isinstance(self.expert_sign_patterns, torch.Tensor):
            raise TypeError("TrellisMxWeights.expert_sign_patterns must be a tensor")


def _require(
    tensors: Mapping[str, torch.Tensor],
    name: str,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if name not in tensors:
        raise ValueError(f"TrellisMX sidecar is missing tensor {name!r}")
    tensor = tensors[name]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"TrellisMX {name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise ValueError(f"TrellisMX {name} must be {dtype}; got {tensor.dtype}")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"TrellisMX {name} must have shape {tuple(shape)}; got "
            f"{tuple(tensor.shape)}"
        )
    return tensor


def _optional(
    tensors: Mapping[str, torch.Tensor],
    name: str,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor | None:
    if name not in tensors or tensors[name] is None:
        return None
    return _require(tensors, name, dtype=dtype, shape=shape)


def validate_trellismx_tensors(
    tensors: Mapping[str, object],
    meta: TrellisMxMetadata,
    *,
    verify_sha256: bool = False,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Shape/dtype gate for one sidecar, before any byte is used."""

    if not isinstance(tensors, Mapping):
        raise TypeError("TrellisMX tensors must be a mapping")
    known = set(CARRIER_NAMES) | set(PLAIN_SCALE_NAMES) | set(CSF_SCALE_NAMES)
    known |= set(CSF_EXCEPTION_NAMES) | set(BOUNDARY_NAMES) | set(OPTIONAL_NAMES)
    unknown = sorted(set(tensors) - known)
    if unknown:
        raise ValueError(
            "TrellisMX sidecar carries unexpected tensors: " + ", ".join(unknown)
        )
    experts = meta.num_experts
    hidden = meta.hidden_size
    local = meta.intermediate_size
    bits = meta.bits
    hidden_tiles = hidden // 16
    local_tiles = local // 16
    stream_words = 16 * bits
    _require(
        tensors,
        "w13_trellis",
        dtype=torch.int16,
        shape=(2, experts, hidden_tiles, local_tiles, stream_words),
    )
    _require(
        tensors,
        "w2_trellis",
        dtype=torch.int16,
        shape=(experts, local_tiles, hidden_tiles, stream_words),
    )
    _require(
        tensors,
        "intermediate_scales_fp16",
        dtype=torch.float16,
        shape=(experts, 3 * local),
    )
    _require(
        tensors,
        "gate_up_suh_fp16",
        dtype=torch.float16,
        shape=(hidden,),
    )
    _require(
        tensors,
        "down_svh_fp16",
        dtype=torch.float16,
        shape=(hidden,),
    )
    _optional(
        tensors,
        "coupled_sign_draw_u8",
        dtype=torch.uint8,
        shape=(experts,),
    )
    has_plain = any(name in tensors for name in PLAIN_SCALE_NAMES)
    has_csf = any(name in tensors for name in CSF_SCALE_NAMES)
    if has_plain and has_csf:
        raise ValueError(
            "TrellisMX sidecar carries both plain and CSF weight scales; a "
            "release stores one or the other"
        )
    if not has_plain and not has_csf:
        raise ValueError(
            "TrellisMX sidecar is missing its weight scales "
            f"({PLAIN_SCALE_NAMES[0]!r} or {CSF_SCALE_NAMES[0]!r})"
        )
    if has_plain:
        _require(
            tensors,
            "w13_scale_ue8m0",
            dtype=torch.uint8,
            shape=(experts, 2 * local, hidden // 32),
        )
        _require(
            tensors,
            "w2_scale_ue8m0",
            dtype=torch.uint8,
            shape=(experts, hidden, local // 32),
        )
        stray = sorted(name for name in CSF_EXCEPTION_NAMES if name in tensors)
        if stray:
            raise ValueError(
                "TrellisMX sidecar carries CSF exception tensors without a "
                "CSF slab: " + ", ".join(stray)
            )
    else:
        for name in CSF_EXCEPTION_NAMES:
            if name not in tensors:
                raise ValueError(f"TrellisMX sidecar is missing tensor {name!r}")
        stride = 16 * (1 + ((hidden // 32) + 7) // 8)
        _require(
            tensors,
            "w13_scale.mxfp4_csf_fixed",
            dtype=torch.uint8,
            shape=(experts, (2 * local) // 16, stride),
        )
        stride = 16 * (1 + ((local // 32) + 7) // 8)
        _require(
            tensors,
            "w2_scale.mxfp4_csf_fixed",
            dtype=torch.uint8,
            shape=(experts, hidden // 16, stride),
        )
        for name in ("w13_exception_offsets", "w2_exception_offsets"):
            tensor = tensors[name]
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.int64:
                raise ValueError(f"TrellisMX {name} must be int64")
            if tuple(tensor.shape) != (experts + 1,):
                raise ValueError(
                    f"TrellisMX {name} must have shape {(experts + 1,)}; got "
                    f"{tuple(tensor.shape)}"
                )
        for name in ("w13_exceptions", "w2_exceptions"):
            tensor = tensors[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"TrellisMX {name} must be a torch.Tensor")
            if tensor.dtype not in (torch.int32, torch.uint32):
                raise ValueError(f"TrellisMX {name} must be int32/uint32")
            if tensor.ndim != 1:
                raise ValueError(f"TrellisMX {name} must be one-dimensional")
    if verify_sha256:
        if metadata is None:
            raise ValueError("verify_sha256 requires the sidecar metadata")
        for name in sorted(tensors):
            key = f"sha256_{name}"
            expected = metadata.get(key)
            if expected is None:
                raise ValueError(
                    f"verify_sha256 was requested but metadata has no {key!r}"
                )
            tensor = tensors[name]
            if not isinstance(tensor, torch.Tensor):
                continue
            digest = hashlib.sha256(
                tensor.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            if digest != str(expected):
                raise ValueError(
                    f"TrellisMX {name} sha256 {digest} does not match the "
                    f"metadata value {expected}"
                )


def _ue8m0_plane(
    tensors: Mapping[str, object],
    name: str,
    *,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Return the logical UE8M0 scale plane for one projection."""

    tensor = tensors[name]
    assert isinstance(tensor, torch.Tensor)
    if tensor.dtype != torch.uint8:
        raise ValueError(f"TrellisMX {name} must be uint8; got {tensor.dtype}")
    if tuple(tensor.shape)[1:] != (rows, columns):
        raise ValueError(
            f"TrellisMX {name} must be [E, {rows}, {columns}]; got "
            f"{tuple(tensor.shape)}"
        )
    return tensor


def build_trellismx_weights(
    tensors: Mapping[str, object],
    meta: TrellisMxMetadata,
    *,
    verify_sha256: bool = False,
    metadata: Mapping[str, object] | None = None,
) -> TrellisMxWeights:
    """Validate one sidecar and assemble its container-independent bundle."""

    validate_trellismx_tensors(
        tensors, meta, verify_sha256=verify_sha256, metadata=metadata
    )
    experts = meta.num_experts
    hidden = meta.hidden_size
    local = meta.intermediate_size
    if "w13_scale_ue8m0" in tensors:
        w13_scale: object = _ue8m0_plane(
            tensors, "w13_scale_ue8m0", rows=2 * local, columns=hidden // 32
        )
        w2_scale: object = _ue8m0_plane(
            tensors, "w2_scale_ue8m0", rows=hidden, columns=local // 32
        )
    else:
        w13_scale = csf_scale_source(
            tensors["w13_scale.mxfp4_csf_fixed"],
            tensors["w13_exceptions"].view(torch.uint32),
            tensors["w13_exception_offsets"],
            rows=2 * local,
            columns=hidden // 32,
        )
        w2_scale = csf_scale_source(
            tensors["w2_scale.mxfp4_csf_fixed"],
            tensors["w2_exceptions"].view(torch.uint32),
            tensors["w2_exception_offsets"],
            rows=hidden,
            columns=local // 32,
        )
    signs = tensors.get("coupled_sign_draw_u8")
    if signs is None:
        # A container without a sign draw uses draw 0 (all +1); follow the
        # carriers' device so a CUDA-staged sidecar stays on device.
        signs = torch.zeros(
            experts, dtype=torch.uint8, device=tensors["w13_trellis"].device
        )
    return TrellisMxWeights(
        w13=tensors["w13_trellis"],
        w2=tensors["w2_trellis"],
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        input_scales=ScaleFactors(tensors["gate_up_suh_fp16"]),
        intermediate_scales=ScaleFactors(
            tensors["intermediate_scales_fp16"].reshape(experts, 3, local)
        ),
        output_scales=ScaleFactors(tensors["down_svh_fp16"]),
        expert_sign_patterns=signs,
    )


def plan_trellismx(meta: TrellisMxMetadata):
    """Plan one rank-local TrellisMX layer onto the W4A8 trellis recipe.

    The activation is the one the release declares and every TrellisMX arm
    implements: ``silu-cap10`` with BF16 activations. It is not configurable.
    """

    activation = ActivationSpec(
        mode=ActivationMode.A8,
        nonlinearity="silu",
        io_dtype=torch.bfloat16,
        swiglu_limit=TRELLISMX_SWIGLU_LIMIT,
    )
    geometry = MoEGeometry(
        num_experts=meta.num_experts,
        hidden_size=meta.hidden_size,
        intermediate_size=meta.intermediate_size,
    )
    return plan_weights(
        source=meta.source(),
        activation=activation,
        geometry=geometry,
        constraints=WeightPlanConstraints(
            required_packing=WeightPacking.TRELLIS_NATIVE
        ),
    )


def prepare_trellismx_experts(
    tensors: Mapping[str, object],
    metadata: Mapping[str, object],
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    device: torch.device | str | None = None,
    global_intermediate_size: int | None = None,
    verify_sha256: bool = False,
    csf_inline: bool = False,
) -> PreparedExperts:
    """Prepare one released TrellisMX rank layer for ``fused_moe`` plan/bind/run.

    ``tensors`` holds the sidecar tensors under their released names;
    ``metadata`` is the sidecar ``__metadata__`` (or an equivalent mapping with
    ``law``/``compander_scale``/``alphabet``/``scale``/``boundary``/``bits``).
    Everything that is not a released TrellisMX sidecar raises ``ValueError``
    naming the tensor or field; there is no fallback to another format.
    """

    meta = TrellisMxMetadata.from_sidecar(
        metadata,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        global_intermediate_size=global_intermediate_size,
    )
    if csf_inline:
        raise NotImplementedError(
            "inline CSF scales require the W4A8 trellis arm of "
            "MoEDynamicKernelBackend to accept csf_inline_scales (kernel gate "
            "at dynamic/kernel.py:207); until it does, TrellisMX expands CSF "
            "losslessly to UE8M0 bytes"
        )
    weights = build_trellismx_weights(
        tensors, meta, verify_sha256=verify_sha256, metadata=metadata
    )
    plan = plan_trellismx(meta)
    return prepare_weights(plan=plan, weights=weights, device=device)


__all__ = [
    "ACTIVATION",
    "ALPHABET",
    "BOUNDARY",
    "COMPANDER_SCALE",
    "LAW_ALIASES",
    "LAW_PROCEDURAL_NPK",
    "SCALE_GRID",
    "SUPPORTED_BITS",
    "TrellisMxBoundaryOps",
    "TrellisMxMetadata",
    "TrellisMxWeights",
    "build_trellismx_weights",
    "csf_scale_source",
    "decode_csf_scale_plane",
    "encode_csf_scale_plane",
    "plan_trellismx",
    "prepare_trellismx_experts",
    "validate_trellismx_tensors",
]
