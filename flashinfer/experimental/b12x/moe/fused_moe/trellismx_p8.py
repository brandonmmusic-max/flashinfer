"""Serve a prepared TrellisMX bundle through an external P8 runtime.

The P8 kernels (trellis stream decoded to E4M3 inside the MMA, UE8M0/32 scales
consumed by the tensor core) are not part of this module; b12x's own runtime is
``trellismx_p8_runtime.P8NativeTPMoE``. A consumer builds the runtime from a
backend it pins with :class:`TrellisMxP8Backend` (``in_tree_p8_backend()`` for
b12x's) and attaches it to the prepared bundle with
:func:`attach_trellismx_p8_runtime`, which returns a new bundle that owns the
runtime. There is no module-level registry: the runtime
lives exactly as long as the attached bundle and what references it.

Attachment verifies the execution contract before anything runs:

* the backend files hash to their pinned values (read and hashed on every
  attach) and contain the runtime's class;
* geometry, bitrate, SwiGLU limit 10, BF16 I/O and the coupled, deterministic
  reducer configuration, against the prepared record and the owning bundle's
  plan;
* every tensor the runtime hands its kernels as a raw pointer is contiguous,
  16-byte aligned, on the bundle's device and exactly as long as expected;
* the runtime reads the bundle's own W13/W2 carriers and logical UE8M0 planes in
  place (same storage, byte offset and length);
* its private repacked scale planes and its packed coupled-boundary vectors are
  byte-equal to the bundle's.

The runtime's resolved behavior settings (the attributes named by
``P8_OPTION_ATTRIBUTES``: RP2 row packing, direct-kernel limit and tile, FC1
schedule, grid policy and the rest) are recorded as
:class:`TrellisMxP8Options`. They are part of the attachment's identity and of
the P8 plan's query, so runtimes built from identical files under different
settings never share a plan identity. File hashes alone do not identify them.

Immutability. While attached, the verified tensors, the runtime's tensor
handles and its settings must not change. ``dataclasses.replace`` of the record
or of the expert bundle keeps the runtime only while every verified tensor,
record field and owner-plan field is unchanged; planning, binding and dispatch
reject a runtime whose handles or settings changed (host-side checks, no device
synchronization). In-place writes to tensor *contents* are not visible without
reading device memory: :func:`verify_trellismx_p8_attachment` re-checks them on
request. To change weights, scales or boundary vectors, detach the runtime,
prepare again, attach again and recapture every graph made from the bundle.

Serialization. An attachment owns live device programs and never leaves its
process. Pickling (or deep-copying) an attached bundle, a binding or a prepared
P8 state yields a :class:`TrellisMxP8Descriptor` in the attachment's place: it
keeps the backend declaration, contract, options, identity and program
declaration, owns no runtime and no programs, plans the same fixed-backend P8
plan with the same compile jobs, and raises :class:`TrellisMxP8Unavailable` on
any launch instead of falling back to this package's arms. Attaching a runtime
to a bundle that carries a descriptor rebinds it and requires the same identity.

``fused_moe.plan_execution`` on an attached bundle returns a fixed-backend plan
without a tuning race. A runtime that compiles through b12x's compiler, as
b12x's in-tree runtime does, states a ``program_declaration``: a compile-job
factory in its pinned backend and the metadata it compiles from. Each capacity
of the plan then carries that compile job, so preparation compiles the
runtime's programs and checks that every program a call launches was declared,
as for any b12x plan, and a compile the runtime needs after kernel resolution
is frozen raises. A runtime with its own compiler declares nothing: its
programs are outside b12x's discovery and freeze guard, so the consumer must
compile and install every arm it will run before capture. Either way, install
every program before capturing graphs that launch it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import torch

from b12x._lib.compile_plan import load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.scratch import scratch_buffer_spec, scratch_tensor
from b12x.preparation import (
    FrozenMapping,
    Knob,
    ParameterBinding,
    ParameterSpace,
    TuningContract,
)
from b12x.preparation._efficiency import capture_autotune_effort
from b12x.preparation.types import (
    MemoryRequirements,
    PersistentMemory,
    Plan,
    _CompositePlan,
)

from b12x.moe._shared.p8.launch import P8LaunchConfig
from b12x.moe._shared.p8.plan import (
    P8ProgramSpec,
    launch_spec,
    p8_capacity_scratch_nbytes,
)
from b12x.moe._shared.p8.scratch import P8_ARENA_ALIGNMENT

from ._impl import (
    B12XFP4ExpertWeights,
    TPMoEFP4Binding,
    _e8m0_scale_to_w4a8_sfb_inplace,
    _PreparedW4A8TrellisWeights,
    _PreparedWeightRepresentation,
)
from .weights import PreparedExperts

__all__ = [
    "P8_BACKEND",
    "P8_OPTION_ATTRIBUTES",
    "P8_TUNING",
    "TrellisMxP8Attachment",
    "TrellisMxP8Backend",
    "TrellisMxP8Contract",
    "TrellisMxP8Descriptor",
    "TrellisMxP8IncompatibleError",
    "TrellisMxP8Options",
    "TrellisMxP8Query",
    "TrellisMxP8Unavailable",
    "attach_trellismx_p8_runtime",
    "detach_trellismx_p8_runtime",
    "release_trellismx_execution_state",
    "trellismx_p8_attachment",
    "verify_trellismx_p8_attachment",
]

#: Backend name of the fixed execution contract.
P8_BACKEND = "trellismx_p8"
#: GLM-5.3 ``silu-cap10``: the only activation the TrellisMX release declares.
_SWIGLU_LIMIT = 10.0
_REATTACH = (
    "detach it, prepare and attach again, and recapture every graph made from "
    "the bundle"
)


class TrellisMxP8IncompatibleError(ValueError):
    """A P8 runtime, backend or replacement does not match the prepared bundle."""


class TrellisMxP8Unavailable(RuntimeError):
    """A bundle deserialized from an attached one was asked to launch."""


def _digest(value) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# backend declaration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrellisMxP8Backend:
    """The executed P8 backend, pinned by file content.

    ``package`` is the backend's import root and ``files`` maps package-relative
    paths to the SHA-256 of their bytes; it must include the module that
    defines the runtime's class, and should cover every file the runtime can
    execute (its import closure, including modules a package loads by scanning
    its directory). ``upstream`` records where the bytes come from (upstream
    change, image digest, local patches) and is not interpreted.
    """

    package: str
    files: Mapping[str, str] | tuple[tuple[str, str], ...]
    upstream: str = ""

    def __post_init__(self) -> None:
        package = self.package
        if (
            not isinstance(package, str)
            or not package
            or not all(part.isidentifier() for part in package.split("."))
        ):
            raise ValueError(f"backend package must be a module path; got {package!r}")
        items = self.files.items() if isinstance(self.files, Mapping) else self.files
        pinned = {}
        for path, digest in items:
            text = str(path)
            relative = PurePosixPath(text)
            if (
                not text
                or "\\" in text
                or relative.is_absolute()
                or any(part in ("", ".", "..") for part in relative.parts)
            ):
                raise ValueError(
                    f"backend file paths must be package-relative; got {text!r}"
                )
            digest = str(digest).lower()
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError(f"backend file {text!r} needs a SHA-256 hex digest")
            pinned[relative.as_posix()] = digest
        if not pinned:
            raise ValueError("a backend pins at least the runtime's own module")
        object.__setattr__(self, "files", tuple(sorted(pinned.items())))
        object.__setattr__(self, "upstream", str(self.upstream))

    @property
    def identity(self) -> str:
        """SHA-256 of the canonical ``{package, files}`` declaration.

        The file pin only: the runtime's resolved settings are part of the
        attachment's identity (:attr:`TrellisMxP8Attachment.identity`).
        """

        return _digest(
            {"package": self.package, "files": [list(item) for item in self.files]}
        )

    def root(self) -> Path:
        """The backend package's directory."""

        try:
            spec = importlib.util.find_spec(self.package)
        except (ImportError, ValueError) as error:
            raise TrellisMxP8IncompatibleError(
                f"P8 backend package {self.package!r} is not importable: {error}"
            ) from error
        locations = None if spec is None else spec.submodule_search_locations
        if not locations:
            raise TrellisMxP8IncompatibleError(
                f"P8 backend {self.package!r} is not an importable package"
            )
        return Path(next(iter(locations))).resolve()

    def verify(self) -> None:
        """Read and hash every pinned file now; raise naming the first mismatch.

        Nothing is cached between calls, so a file replaced after one attach
        (whatever its size or modification time) fails the next.
        """

        root = self.root()
        for path, expected in self.files:
            try:
                data = (root / path).read_bytes()
            except OSError as error:
                raise TrellisMxP8IncompatibleError(
                    f"P8 backend file {path!r} is missing under {root}"
                ) from error
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected:
                raise TrellisMxP8IncompatibleError(
                    f"P8 backend file {path!r} hashes to {actual}, but the "
                    f"declaration pins {expected}"
                )

    def relative_path(self, module_name: str) -> str:
        """The package-relative source file of ``module_name``."""

        module = sys.modules.get(module_name)
        source = getattr(module, "__file__", None)
        if source is None:
            raise TrellisMxP8IncompatibleError(
                f"the runtime's module {module_name!r} has no source file to pin"
            )
        try:
            return Path(source).resolve().relative_to(self.root()).as_posix()
        except ValueError as error:
            raise TrellisMxP8IncompatibleError(
                f"the runtime's module {module_name!r} ({source}) is not part of the "
                f"declared backend package {self.package!r}"
            ) from error


# --------------------------------------------------------------------------- #
# options, contract and identities
# --------------------------------------------------------------------------- #


#: Runtime attributes that select the P8 runtime's kernels, schedules and
#: storage. b12x's ``P8NativeTPMoE`` resolves them at construction from its
#: ``P8RuntimeOptions``; the runtime of local-inference-lab/b12x#342 resolved
#: them from its arguments and its environment. ``rp2_direct_tile_m``,
#: ``fc1_pipeline_stages`` and ``fc1_warps`` are not options: they are the
#: serving launch configuration's geometry, which a call that names no
#: ``P8LaunchConfig`` uses.
P8_OPTION_ATTRIBUTES = (
    "p8_down_remainder",
    "_dx2_phases",
    "p8_dx2_rowpack",
    "p8_input_rowpack",
    "rp2_direct_max_tokens",
    "rp2_direct_tile_m",
    "rp2_skip_input_clear",
    "p8_epi_par",
    "small_m_scheduler",
    "fc1_tile_n",
    "fc1_pipeline_stages",
    "fc1_warps",
    "fc1_warp_quant",
    "fc1_broadcast_a",
    "fc1_a_swizzle_rotate",
    "fc1_exact_staging",
    "prefill_chunk_tokens",
    "grid_policy",
    "tile_major_tasks",
    "compact_input_storage",
    "force_materialized",
    "mac_override",
)
_OPTION_TYPES = (bool, int, float, str, type(None))


@dataclass(frozen=True)
class TrellisMxP8Options:
    """Resolved behavior settings of a P8 runtime, by attribute name.

    ``values`` maps names from ``P8_OPTION_ATTRIBUTES`` to plain values (bool,
    int, float, str or None). The attachment records the complete set read from
    the runtime; a consumer may pass a subset to
    :func:`attach_trellismx_p8_runtime` as the values it requires.
    """

    values: Mapping[str, object] | tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        items = dict(
            self.values.items() if isinstance(self.values, Mapping) else self.values
        )
        unknown = sorted(set(items) - set(P8_OPTION_ATTRIBUTES))
        if unknown:
            raise ValueError(
                f"unknown P8 options {unknown}; the recorded ones are "
                f"{list(P8_OPTION_ATTRIBUTES)}"
            )
        for name, value in items.items():
            if type(value) not in _OPTION_TYPES:
                raise TypeError(
                    f"P8 option {name} must be a bool, int, float, str or None; "
                    f"got {type(value).__name__}"
                )
        object.__setattr__(self, "values", tuple(sorted(items.items())))

    @classmethod
    def from_runtime(cls, runtime: object) -> "TrellisMxP8Options":
        """Every option in ``P8_OPTION_ATTRIBUTES`` as ``runtime`` resolved it."""

        values = {}
        for name in P8_OPTION_ATTRIBUTES:
            value = _attribute(runtime, name)
            _require(
                type(value) in _OPTION_TYPES,
                f"the P8 runtime's {name} must be a bool, int, float, str or None; "
                f"got {type(value).__name__}",
            )
            values[name] = value
        return cls(values)

    def as_dict(self) -> dict[str, object]:
        return dict(self.values)

    @property
    def identity(self) -> str:
        """SHA-256 of the canonical option values."""

        return _digest([list(item) for item in self.values])


@dataclass(frozen=True)
class TrellisMxP8Contract:
    """What the attached runtime executes; checked at attach, plan and dispatch."""

    device: torch.device
    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    bits: int
    io_dtype: torch.dtype = torch.bfloat16
    activation: str = "silu"
    swiglu_limit: float = _SWIGLU_LIMIT
    swiglu_clamp_after_silu: bool = False
    codebook: str = "procedural_npk"
    scale_format: str = "e8m0_k32"

    def computation(self) -> dict[str, object]:
        """Every field but the device (a descriptor may be relocated)."""

        return {
            "num_experts": int(self.num_experts),
            "hidden_size": int(self.hidden_size),
            "intermediate_size": int(self.intermediate_size),
            "top_k": int(self.top_k),
            "bits": int(self.bits),
            "io_dtype": str(self.io_dtype).removeprefix("torch."),
            "activation": self.activation,
            "swiglu_limit": float(self.swiglu_limit),
            "swiglu_clamp_after_silu": bool(self.swiglu_clamp_after_silu),
            "codebook": self.codebook,
            "scale_format": self.scale_format,
        }


@dataclass(frozen=True)
class _TensorIdentity:
    storage: int
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def of(cls, tensor: torch.Tensor) -> "_TensorIdentity":
        size = tensor.element_size()
        return cls(
            storage=tensor.untyped_storage().data_ptr(),
            offset=tensor.storage_offset() * size,
            nbytes=tensor.numel() * size,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            device=tensor.device,
        )

    def extent(self) -> tuple[int, int, int, torch.device]:
        return (self.storage, self.offset, self.nbytes, self.device)

    def geometry(self) -> tuple:
        """What stays meaningful in another process: no address, no device."""

        return (self.offset, self.nbytes, self.shape, self.stride, self.dtype)


#: Prepared-record tensors the runtime depends on (directly or by value).
_RECORD_TENSORS = (
    "w13",
    "w2",
    "w13_scale",
    "w2_scale",
    "w13_sfb",
    "w2_sfb",
    "intermediate_rotations",
    "gate_suh",
    "up_suh",
    "down_svh",
)
#: Prepared-record fields that define the computation.
_RECORD_FIELDS = (
    "num_experts",
    "hidden_size",
    "intermediate_size",
    "params_dtype",
    "trellis_bits",
    "trellis_codebook",
    "trellis_exact_hadamard",
    "trellis_fp32_boundary",
    "trellis_pre_quantized_a",
    "source_format",
    "scale_format",
    "weight_layout",
    "w13_layout",
)
#: Bundle fields that must stay the prepared record's tensors.
_BUNDLE_ALIASES = (
    ("w1_fp4", "w13"),
    ("w2_fp4", "w2"),
    ("w1_blockscale", "w13_sfb"),
    ("w2_blockscale", "w2_sfb"),
)
#: Runtime tensors its kernels receive as raw 16-byte aligned pointers.
_RUNTIME_TENSORS = (
    "w13_stream",
    "w2_stream",
    "w13_scale_mx",
    "w2_scale_mx",
    "w13_sfb",
    "w2_sfb",
    "scale_component_packed",
)
#: Runtime settings the attachment verifies and then holds fixed.
_RUNTIME_SETTINGS = (
    "device",
    "experts",
    "hidden",
    "intermediate",
    "topk",
    "trellis_bits",
    "swiglu_limit",
    "full_coupled",
    "deterministic_output",
    "debug_capture",
    "compact_scale_storage",
)
_MISSING = object()


def _same_setting(current: object, recorded: object) -> bool:
    return type(current) is type(recorded) and bool(current == recorded)


class _P8Entry:
    """What a live attachment and its deserialized descriptor share."""

    backend: TrellisMxP8Backend
    contract: TrellisMxP8Contract
    options: TrellisMxP8Options
    _fields: tuple[tuple[str, object], ...]
    programs: tuple[str, object] | None

    def program_spec(self) -> P8ProgramSpec:
        """The declaration a capacity's programs, arms and arena are derived from.

        A runtime that declares no programs compiles outside b12x's discovery,
        so the plan cannot declare the scratch its calls need; it raises here
        rather than planning a capacity it cannot size.
        """

        if self.programs is None:
            raise ValueError(
                "a P8 runtime must declare its programs (a program_declaration "
                "compile job) so the plan can declare the caller-owned scratch "
                "its calls need"
            )
        return self.programs[1]

    def compile_jobs(
        self, tokens: int, config: P8LaunchConfig | None = None
    ) -> tuple[CompileJob, ...]:
        """The compile job declaring the programs any call with 1 to ``tokens``
        rows launches under ``config``, or none for a runtime with its own
        compiler."""

        if self.programs is None:
            return ()
        factory, metadata = self.programs
        return (
            CompileJob.create(
                factory,
                launch_spec(metadata, config),
                int(tokens),
                CompileJob.DEVICE_ORDINAL,
            ),
        )

    @property
    def identity(self) -> str:
        """SHA-256 of the backend pin, the contract (less the device) and the
        resolved options: equal identities execute the same computation with
        the same kernel selection."""

        return _digest(
            {
                "backend": self.backend.identity,
                "contract": self.contract.computation(),
                "options": [list(item) for item in self.options.values],
            }
        )

    def _check_fields(self, record: _PreparedW4A8TrellisWeights) -> None:
        for name, expected in self._fields:
            if getattr(record, name, None) != expected:
                raise TrellisMxP8IncompatibleError(
                    f"the attached P8 runtime was verified for {name}={expected!r}; got "
                    f"{getattr(record, name, None)!r}"
                )

    def check_owner(
        self, owner: B12XFP4ExpertWeights, record: _PreparedW4A8TrellisWeights
    ) -> None:
        """Raise unless the expert bundle and its plan execute this record."""

        plan = owner.plan
        contract = self.contract
        for name, actual, required in (
            ("source_format", owner.source_format, "trellismx"),
            ("io_dtype", plan.io_dtype, "bfloat16"),
            ("activation", plan.activation, contract.activation),
            ("quant_modes", plan.quant_modes, frozenset({"w4a8_mx"})),
            ("trellis_bits", plan.trellis_bits, contract.bits),
            ("trellis_codebook", plan.trellis_codebook, contract.codebook),
            ("num_experts", plan.num_experts, contract.num_experts),
            ("hidden_size", plan.hidden_size, contract.hidden_size),
            ("intermediate_size", plan.intermediate_size, contract.intermediate_size),
            ("intermediate_hadamard", plan.intermediate_hadamard, True),
            (
                "intermediate_hadamard_blocks",
                plan.intermediate_hadamard_blocks,
                (128, 128),
            ),
            ("w13_layout", plan.w13_layout, record.w13_layout),
        ):
            if actual != required:
                raise TrellisMxP8IncompatibleError(
                    f"the attached P8 runtime executes {name}={required!r}; the "
                    f"expert bundle's plan declares {actual!r}"
                )
        for bundle_name, record_name in _BUNDLE_ALIASES:
            if _TensorIdentity.of(getattr(owner, bundle_name)) != _TensorIdentity.of(
                getattr(record, record_name)
            ):
                raise TrellisMxP8IncompatibleError(
                    f"expert bundle field {bundle_name!r} is not the prepared "
                    f"{record_name!r} the P8 runtime was verified against"
                )


@dataclass(frozen=True, eq=False)
class TrellisMxP8Attachment(_P8Entry):
    """A verified runtime owned by one prepared TrellisMX record (live)."""

    runtime: object = field(repr=False)
    backend: TrellisMxP8Backend
    contract: TrellisMxP8Contract
    options: TrellisMxP8Options
    _tensors: tuple[tuple[str, _TensorIdentity], ...] = field(repr=False)
    _fields: tuple[tuple[str, object], ...] = field(repr=False)
    _handles: tuple[tuple[str, torch.Tensor], ...] = field(repr=False)
    _settings: tuple[tuple[str, object], ...] = field(repr=False)
    #: Device bytes the runtime holds beyond the bundle's own storage (its
    #: packed boundary vectors).
    resident_nbytes: int = 0
    #: Unique per attach, kept by its descriptors; keys the runtime's resident
    #: memory in every plan made from the bundle.
    token: str = ""
    #: ``(factory, metadata)`` of the compile job declaring the runtime's
    #: programs, or None for a runtime with its own compiler.
    programs: tuple[str, object] | None = None

    #: A launchable attachment (a :class:`TrellisMxP8Descriptor` is not).
    live = True

    def require_live(self) -> "TrellisMxP8Attachment":
        return self

    def check_record(self, record: _PreparedW4A8TrellisWeights) -> None:
        """Raise unless ``record`` still holds every verified tensor and field."""

        for name, identity in self._tensors:
            value = getattr(record, name, None)
            if (
                not isinstance(value, torch.Tensor)
                or _TensorIdentity.of(value) != identity
            ):
                raise TrellisMxP8IncompatibleError(
                    f"the attached P8 runtime was verified against a different {name!r}; "
                    "detach it before replacing the prepared tensors"
                )
        self._check_fields(record)

    def check_runtime(self) -> None:
        """Raise unless the runtime still holds its verified handles and settings.

        Host-side attribute reads only (no device access, no synchronization).
        """

        runtime = self.runtime
        for name, handle in self._handles:
            if getattr(runtime, name, None) is not handle:
                raise TrellisMxP8IncompatibleError(
                    f"the attached P8 runtime's {name} was replaced after attachment; "
                    f"{_REATTACH}"
                )
        for name, recorded in self._settings:
            current = getattr(runtime, name, _MISSING)
            if not _same_setting(current, recorded):
                raise TrellisMxP8IncompatibleError(
                    f"the attached P8 runtime's {name} changed from {recorded!r} to "
                    f"{'<missing>' if current is _MISSING else repr(current)} after "
                    f"attachment; {_REATTACH}"
                )

    def __reduce__(self):
        # Never the runtime or its programs: another process gets the inert
        # descriptor (``__copy__`` keeps a shallow copy live in this process).
        return (
            TrellisMxP8Descriptor,
            (
                self.backend,
                self.contract,
                self.options,
                tuple((name, ident.geometry()) for name, ident in self._tensors),
                self._fields,
                self.resident_nbytes,
                self.token,
                self.programs,
            ),
        )

    def __copy__(self) -> "TrellisMxP8Attachment":
        return self


@dataclass(frozen=True, eq=False)
class TrellisMxP8Descriptor(_P8Entry):
    """An attachment as it exists outside its process: identity, no runtime.

    Produced by pickling or deep-copying a :class:`TrellisMxP8Attachment` (and
    so anything that holds one). It plans the attached bundle's fixed-backend
    P8 plan, with the attachment's compile jobs, and binds. Planning fails at
    materialization with :class:`TrellisMxP8Unavailable`, because the runtime
    lives only in the process that attached it. It never falls back to this
    package's arms.
    """

    backend: TrellisMxP8Backend
    contract: TrellisMxP8Contract
    options: TrellisMxP8Options
    _geometry: tuple[tuple[str, tuple], ...] = field(repr=False)
    _fields: tuple[tuple[str, object], ...] = field(repr=False)
    resident_nbytes: int = 0
    token: str = ""
    programs: tuple[str, object] | None = None

    live = False

    def _unavailable(self) -> TrellisMxP8Unavailable:
        return TrellisMxP8Unavailable(
            "this TrellisMX bundle was deserialized from one whose P8 runtime lives "
            "in the process that attached it; the descriptor it carries plans and "
            "binds but never launches. Run the call in that process, or attach a "
            "runtime here with attach_trellismx_p8_runtime (the same identity is "
            "required)"
        )

    @property
    def runtime(self):
        raise self._unavailable()

    def require_live(self):
        raise self._unavailable()

    def check_runtime(self) -> None:
        """Nothing to check: the descriptor owns no runtime and never launches."""

    def check_record(self, record: _PreparedW4A8TrellisWeights) -> None:
        """Raise unless ``record`` keeps the verified tensor layouts and fields."""

        for name, geometry in self._geometry:
            value = getattr(record, name, None)
            if (
                not isinstance(value, torch.Tensor)
                or _TensorIdentity.of(value).geometry() != geometry
            ):
                raise TrellisMxP8IncompatibleError(
                    f"the P8 descriptor was made for a different {name!r}"
                )
        self._check_fields(record)

    def __reduce__(self):
        return (
            TrellisMxP8Descriptor,
            (
                self.backend,
                self.contract,
                self.options,
                self._geometry,
                self._fields,
                self.resident_nbytes,
                self.token,
                self.programs,
            ),
        )


def trellismx_p8_attachment(
    experts: PreparedExperts | B12XFP4ExpertWeights | None,
) -> TrellisMxP8Attachment | TrellisMxP8Descriptor | None:
    """What serves ``experts`` through P8: the live attachment, the inert
    descriptor of a deserialized bundle (``.live`` is False), or ``None``."""

    impl = experts._impl if isinstance(experts, PreparedExperts) else experts
    representation = getattr(impl, "representation", None)
    value = getattr(representation, "value", None)
    return getattr(value, "p8", None)


# --------------------------------------------------------------------------- #
# attach / detach / verify / release
# --------------------------------------------------------------------------- #


def _record(experts: PreparedExperts) -> _PreparedW4A8TrellisWeights:
    if not isinstance(experts, PreparedExperts):
        raise TypeError("a P8 runtime attaches to PreparedExperts from prepare_weights")
    impl = experts._impl
    if impl.source_format != "trellismx":
        raise TrellisMxP8IncompatibleError(
            "a P8 runtime serves prepared TrellisMX experts only; got "
            f"source_format={impl.source_format!r}"
        )
    record = getattr(impl.representation, "value", None)
    if not isinstance(record, _PreparedW4A8TrellisWeights):
        raise TrellisMxP8IncompatibleError(
            "the bundle does not carry the prepared TrellisMX W4A8 record"
        )
    return record


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrellisMxP8IncompatibleError(message)


def _attribute(runtime: object, name: str):
    try:
        return getattr(runtime, name)
    except AttributeError as error:
        raise TrellisMxP8IncompatibleError(
            f"the P8 runtime lacks the required attribute {name!r}"
        ) from error


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _launch_tensor(runtime: object, name: str, device: torch.device) -> torch.Tensor:
    """A runtime tensor its kernels read through a raw pointer, layout-checked."""

    tensor = _attribute(runtime, name)
    _require(
        isinstance(tensor, torch.Tensor), f"the P8 runtime's {name} must be a tensor"
    )
    _require(
        tensor.device == device,
        f"the P8 runtime's {name} lives on {tensor.device}, the bundle on {device}",
    )
    _require(
        tensor.is_contiguous(),
        f"the P8 runtime's {name} must be contiguous: its kernels read it through "
        "a raw pointer",
    )
    _require(
        tensor.data_ptr() % 16 == 0,
        f"the P8 runtime's {name} must start 16-byte aligned; it starts "
        f"{tensor.data_ptr() % 16} bytes past a boundary",
    )
    return tensor


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _same_extent(
    name: str, runtime_tensor: torch.Tensor, bundle_tensor: torch.Tensor
) -> None:
    _require(
        _TensorIdentity.of(runtime_tensor).extent()
        == _TensorIdentity.of(bundle_tensor).extent(),
        f"the P8 runtime's {name} does not read the bundle's own bytes in place "
        "(storage, byte offset and length must match)",
    )


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def _same_bytes(
    name: str, runtime_tensor: torch.Tensor, expected: torch.Tensor
) -> None:
    # The runtime tensor is contiguous (``_launch_tensor``), so its linear bytes
    # are exactly what its kernels read.
    _require(
        _nbytes(runtime_tensor) == _nbytes(expected)
        and torch.equal(_bytes(runtime_tensor), _bytes(expected)),
        f"the P8 runtime's {name} differs from the prepared bundle's bytes",
    )


def _packed_boundary(record: _PreparedW4A8TrellisWeights) -> torch.Tensor:
    """``cat(gate_up_suh, intermediate_scales, down_svh, coupled_signs)`` (fp16)."""

    local = int(record.intermediate_size)
    rotations = record.intermediate_rotations
    signs = rotations[:, 3 * local :]
    _require(
        bool(torch.equal(signs, signs[:1].expand_as(signs))),
        "the P8 runtime applies one coupled sign vector to every expert; the "
        "bundle's per-expert sign rows differ",
    )
    _require(
        bool(torch.equal(_bytes(record.gate_suh), _bytes(record.up_suh))),
        "the P8 runtime shares gate_up_suh between gate and up; the bundle's differ",
    )
    return torch.cat(
        (
            record.gate_suh.reshape(-1),
            rotations[:, : 3 * local].reshape(-1),
            record.down_svh.reshape(-1),
            signs[0].reshape(-1),
        )
    ).to(torch.float16)


def _repacked(record: _PreparedW4A8TrellisWeights, name: str) -> torch.Tensor:
    """The W4A8 repack of the record's logical UE8M0 plane, computed again."""

    experts, hidden = int(record.num_experts), int(record.hidden_size)
    local = int(record.intermediate_size)
    if name == "w13_sfb":
        return _e8m0_scale_to_w4a8_sfb_inplace(
            record.w13_scale.clone(),
            weight_E=experts,
            rows=2 * local,
            k_dim=hidden,
            gated_half_rows=local,
        )
    return _e8m0_scale_to_w4a8_sfb_inplace(
        record.w2_scale.clone(), weight_E=experts, rows=hidden, k_dim=local
    )


def _verify_contents(
    runtime: object,
    record: _PreparedW4A8TrellisWeights,
    device: torch.device,
    *,
    repack: bool,
) -> dict[str, torch.Tensor]:
    """Layout-check the runtime's raw-pointer tensors and compare their bytes.

    ``repack`` recomputes the repacked planes from the logical ones instead of
    trusting the record's (after attach the record holds the runtime's own).
    Returns the runtime tensors by name.
    """

    tensors = {name: _launch_tensor(runtime, name, device) for name in _RUNTIME_TENSORS}
    _same_extent("w13_stream", tensors["w13_stream"], record.w13)
    _same_extent("w2_stream", tensors["w2_stream"], record.w2)
    _same_extent("w13_scale_mx", tensors["w13_scale_mx"], record.w13_scale)
    _same_extent("w2_scale_mx", tensors["w2_scale_mx"], record.w2_scale)
    for name in ("w13_sfb", "w2_sfb"):
        expected = _repacked(record, name) if repack else getattr(record, name)
        _same_bytes(name, tensors[name], expected)
    _same_bytes(
        "scale_component_packed",
        tensors["scale_component_packed"],
        _packed_boundary(record),
    )
    return tensors


def _check_activation(experts: PreparedExperts) -> None:
    activation = experts.plan.activation
    _require(
        activation.nonlinearity == "silu"
        and activation.swiglu_limit == _SWIGLU_LIMIT
        and not activation.swiglu_clamp_after_silu
        and activation.swiglu_alpha in (None, 1.0)
        and activation.swiglu_beta in (None, 0.0)
        and activation.io_dtype == torch.bfloat16,
        "a P8 runtime serves BF16 silu-cap10 TrellisMX plans; the bundle declares "
        f"{activation}",
    )


def _check_declared_options(
    declared: TrellisMxP8Options | Mapping[str, object],
    resolved: TrellisMxP8Options,
) -> None:
    if not isinstance(declared, TrellisMxP8Options):
        declared = TrellisMxP8Options(declared)
    actual = resolved.as_dict()
    for name, value in declared.values:
        _require(
            _same_setting(actual[name], value),
            f"the P8 runtime resolved {name}={actual[name]!r}; the consumer "
            f"declares {value!r}",
        )


def attach_trellismx_p8_runtime(
    experts: PreparedExperts,
    runtime: object,
    *,
    backend: TrellisMxP8Backend,
    options: TrellisMxP8Options | Mapping[str, object] | None = None,
) -> PreparedExperts:
    """Return a copy of ``experts`` that owns ``runtime`` and executes through it.

    ``experts`` is not modified. ``options``, when given, are runtime settings
    (names from ``P8_OPTION_ATTRIBUTES``) the runtime must have resolved; the
    complete resolved set is recorded either way. A bundle that carries the
    descriptor of a deserialized attachment is rebound, and the new attachment
    must have the descriptor's identity. Raises
    :class:`TrellisMxP8IncompatibleError` naming the first mismatch. Compares
    device tensors once, so it must run outside CUDA graph capture.
    """

    if not isinstance(backend, TrellisMxP8Backend):
        raise TypeError("backend must be a TrellisMxP8Backend")
    record = _record(experts)
    impl = experts._impl
    previous = record.p8
    if isinstance(previous, TrellisMxP8Attachment):
        raise TrellisMxP8IncompatibleError(
            "the bundle already owns a P8 runtime; detach it first"
        )
    if _capturing():
        raise RuntimeError("attach a P8 runtime outside CUDA graph capture")
    _check_activation(experts)

    backend.verify()
    module_path = backend.relative_path(type(runtime).__module__)
    _require(
        module_path in dict(backend.files),
        f"the runtime's module {module_path!r} is not among the pinned backend files",
    )

    device = impl.w1_fp4.device
    contract = TrellisMxP8Contract(
        device=device,
        num_experts=impl.num_experts,
        hidden_size=impl.hidden_size,
        intermediate_size=impl.intermediate_size,
        top_k=int(_attribute(runtime, "topk")),
        bits=int(record.trellis_bits),
    )
    _require(
        torch.device(_attribute(runtime, "device")) == device,
        f"the P8 runtime lives on {_attribute(runtime, 'device')}, the bundle on {device}",
    )
    for name, expected in (
        ("experts", contract.num_experts),
        ("hidden", contract.hidden_size),
        ("intermediate", contract.intermediate_size),
        ("trellis_bits", contract.bits),
    ):
        actual = _attribute(runtime, name)
        _require(
            int(actual) == int(expected),
            f"the P8 runtime's {name}={actual} does not match the bundle's {expected}",
        )
    _require(
        float(_attribute(runtime, "swiglu_limit")) == _SWIGLU_LIMIT,
        "the P8 runtime must clamp SwiGLU at 10.0 (silu-cap10); got "
        f"{_attribute(runtime, 'swiglu_limit')}",
    )
    for name, expected in (
        ("full_coupled", True),
        ("deterministic_output", True),
        ("debug_capture", False),
        ("compact_scale_storage", False),
    ):
        _require(
            bool(_attribute(runtime, name)) is expected,
            f"the P8 runtime needs {name}={expected} to serve fused_moe bindings",
        )
    _require(callable(runtime), "the P8 runtime must be callable")
    resolved = TrellisMxP8Options.from_runtime(runtime)
    if options is not None:
        _check_declared_options(options, resolved)
    programs = _program_declaration(runtime, backend)

    tensors = _verify_contents(runtime, record, device, repack=False)

    # The runtime repacked its own copy of both UE8M0 planes, byte-identical to
    # the bundle's (checked above). The attached bundle holds the runtime's
    # copies instead of its own, so one copy stays resident once the caller
    # drops the unattached bundle.
    adopted = {
        name: _adopt(name, tensors[name], getattr(record, name))
        for name in ("w13_sfb", "w2_sfb")
    }
    record = replace(record, p8=None, **adopted)
    shared = {
        _TensorIdentity.of(getattr(record, name)).storage for name in _RECORD_TENSORS
    }
    private = {}
    for name in ("w13_sfb", "w2_sfb", "scale_component_packed"):
        storage = tensors[name].untyped_storage()
        if storage.data_ptr() not in shared:
            private[storage.data_ptr()] = storage.nbytes()

    attachment = TrellisMxP8Attachment(
        runtime=runtime,
        backend=backend,
        contract=contract,
        options=resolved,
        _tensors=tuple(
            (name, _TensorIdentity.of(getattr(record, name)))
            for name in _RECORD_TENSORS
        ),
        _fields=tuple((name, getattr(record, name)) for name in _RECORD_FIELDS),
        _handles=tuple((name, tensors[name]) for name in _RUNTIME_TENSORS),
        _settings=tuple((name, _attribute(runtime, name)) for name in _RUNTIME_SETTINGS)
        + resolved.values,
        resident_nbytes=sum(private.values()),
        token=uuid.uuid4().hex,
        programs=programs,
    )
    if isinstance(previous, TrellisMxP8Descriptor):
        _require(
            previous.identity == attachment.identity,
            "the runtime differs from the one this deserialized bundle was attached "
            f"to (identity {attachment.identity}, the descriptor's "
            f"{previous.identity})",
        )
    return _with_record(experts, replace(record, p8=attachment))


def _program_declaration(
    runtime: object, backend: TrellisMxP8Backend
) -> tuple[str, object] | None:
    """The compile job by which ``runtime`` declares its programs, if any.

    b12x preparation records every program compiled through b12x's compiler
    that a call launches, and rejects one its plan did not declare. A runtime
    compiling that way states ``program_declaration``, a ``(factory,
    metadata)`` pair: an importable compile-job factory from its pinned backend
    and the metadata it compiles from. A runtime with its own compiler states
    none.
    """

    declaration = getattr(runtime, "program_declaration", None)
    if declaration is None:
        return None
    _require(
        isinstance(declaration, tuple)
        and len(declaration) == 2
        and isinstance(declaration[0], str),
        "the P8 runtime's program_declaration must be a (factory, metadata) pair",
    )
    factory, metadata = declaration
    try:
        CompileJob.create(factory, metadata, 1, CompileJob.DEVICE_ORDINAL)
    except (TypeError, ValueError) as error:
        raise TrellisMxP8IncompatibleError(
            f"the P8 runtime's program declaration is not a compile job: {error}"
        ) from error
    _require(
        backend.relative_path(factory.partition(":")[0]) in dict(backend.files),
        f"the P8 runtime's program factory {factory!r} is not among the pinned "
        "backend files",
    )
    return (factory, metadata)


def _adopt(
    name: str, runtime_tensor: torch.Tensor, bundle_tensor: torch.Tensor
) -> torch.Tensor:
    """``runtime_tensor`` viewed as ``bundle_tensor`` (same dtype and shape)."""

    _require(
        bundle_tensor.is_contiguous(),
        f"the bundle's {name} must be contiguous to share the runtime's copy",
    )
    try:
        view = runtime_tensor.reshape(-1).view(torch.uint8)
        return view.view(bundle_tensor.dtype).view(bundle_tensor.shape)
    except RuntimeError as error:
        raise TrellisMxP8IncompatibleError(
            f"the P8 runtime's {name} cannot stand in for the bundle's: {error}"
        ) from error


def _with_record(
    experts: PreparedExperts, record: _PreparedW4A8TrellisWeights
) -> PreparedExperts:
    impl = experts._impl
    representation = _PreparedWeightRepresentation(
        quant_mode=impl.representation.quant_mode,
        layout=impl.representation.layout,
        value=record,
    )
    # The bundle's canonical scale handles are the record's repacked planes.
    return replace(
        experts,
        _impl=replace(
            impl,
            representation=representation,
            w1_blockscale=record.w13_sfb,
            w2_blockscale=record.w2_sfb,
        ),
    )


def detach_trellismx_p8_runtime(experts: PreparedExperts) -> PreparedExperts:
    """Return a copy of ``experts`` without its P8 runtime (or descriptor)."""

    record = _record(experts)
    if record.p8 is None:
        return experts
    return _with_record(experts, replace(record, p8=None))


def verify_trellismx_p8_attachment(experts: PreparedExperts) -> None:
    """Verify an attached runtime against its bundle again, reading device memory.

    Checks the runtime's handles and settings, the record and owner, every
    raw-pointer layout and extent, the repacked planes recomputed from the
    logical ones, and the packed boundary vectors. Synchronizing: call it
    outside CUDA graph capture, after anything that may have written the
    bundle's or the runtime's tensors in place, and before capturing graphs
    that launch the runtime. Raises :class:`TrellisMxP8IncompatibleError`
    naming the first difference.
    """

    record = _record(experts)
    if record.p8 is None:
        raise TrellisMxP8IncompatibleError("the bundle owns no P8 runtime")
    attachment = record.p8.require_live()
    if _capturing():
        raise RuntimeError("verify a P8 attachment outside CUDA graph capture")
    attachment.check_runtime()
    attachment.check_record(record)
    attachment.check_owner(experts._impl, record)
    _verify_contents(
        attachment.runtime, record, attachment.contract.device, repack=True
    )


def release_trellismx_execution_state(
    experts: PreparedExperts | B12XFP4ExpertWeights,
) -> None:
    """Drop the small-M plans cached on this bundle.

    A plan holds no mutable buffer: every binding carves its own views out of
    the caller's scratch, and a captured graph keeps the plan it was built from
    alive. Releasing the cache therefore only makes the next call plan afresh;
    graphs captured earlier keep replaying against the plan they captured.
    """

    impl = experts._impl if isinstance(experts, PreparedExperts) else experts
    if not isinstance(impl, B12XFP4ExpertWeights):
        raise TypeError("expected prepared experts")
    impl._trellismx_smallm_plans.clear()


# --------------------------------------------------------------------------- #
# execution plan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, kw_only=True)
class TrellisMxP8Query:
    """Identity of one P8-served capacity: the backend, its resolved options and
    what it computes.

    ``direct_tokens`` is the largest row count the runtime keeps on its RP2
    direct arm; the launch axes that only specialize that arm are raced for a
    capacity whose own call reaches it, and kept at their serving values
    otherwise. ``mac_pin`` is a resident-grid cap the runtime pins at
    construction: such a runtime's calls never use a raced cap, so the race
    leaves the cap axis at ``0`` (the runtime's own cap) and rejects a
    configuration that names another.
    """

    backend: str
    package: str
    options: str
    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    num_tokens: int
    bits: int
    direct_tokens: int
    mac_pin: int = 0
    io_dtype: str = "bfloat16"
    activation: str = "silu"
    swiglu_limit: float = _SWIGLU_LIMIT
    autotune_effort: str = field(
        default_factory=lambda: capture_autotune_effort().value
    )


#: The launch axes a P8 capacity races, at their serving values. ``parameters``
#: widens the axes that can change the call a capacity is measured at.
_P8_KNOBS = (
    Knob(name="backend", values=(P8_BACKEND,)),
    Knob(
        name="direct_tile_m",
        values=(0,),
        binding=ParameterBinding.COMPILE,
    ),
    Knob(
        name="grouped_tile_m",
        values=(0,),
        binding=ParameterBinding.COMPILE,
    ),
    Knob(
        name="fc1_pipeline_stages",
        values=(2,),
        binding=ParameterBinding.COMPILE,
    ),
    Knob(name="fc1_warps", values=(4,), binding=ParameterBinding.COMPILE),
    Knob(name="rp2_stride", values=(192,), binding=ParameterBinding.COMPILE),
    Knob(name="mac", values=(0,), binding=ParameterBinding.RUNTIME),
)


def _mac_ladder(sm_count: int) -> tuple[int, ...]:
    """The resident-grid caps a race may try: the policy and rungs of the bound."""

    if type(sm_count) is not int or sm_count <= 0:
        return (0,)
    rungs = {1 << exponent for exponent in range(sm_count.bit_length())}
    rungs |= {
        max(1, sm_count // 2),
        max(1, 3 * sm_count // 4),
        max(1, 7 * sm_count // 8),
        sm_count,
    }
    return tuple(sorted({0, *(rung for rung in rungs if rung <= sm_count)}))


def _p8_parameter_space(query: TrellisMxP8Query, device) -> ParameterSpace:
    """The launch configurations one capacity races.

    The direct arm's axes are raced when the capacity's own call lands on that
    arm; otherwise they keep their serving values, because a race measures the
    capacity's token count and those axes cannot change it there. The grouped
    tile races the policy against one alternative, which is the other tile the
    kernels accept for that row count. A runtime that pins its resident-grid cap
    keeps the cap axis at ``0`` (the runtime's own cap), because a raced cap
    could not change what such a runtime launches.
    """

    direct = 0 < int(query.num_tokens) <= int(query.direct_tokens)
    grouped_tile = 32 if int(query.num_tokens) > 128 else 64
    macs = (
        (0,)
        if int(query.mac_pin)
        else _mac_ladder(0 if device is None else device.sm_count)
    )
    top = macs[-1]
    routed = min(int(query.num_tokens), int(query.direct_tokens)) * int(query.top_k)

    def stride_fits(assignment) -> bool:
        # A direct call's high and residual carriers must not overlap.
        return routed <= 0 or assignment["rp2_stride"] > routed

    return ParameterSpace.create(
        _P8_KNOBS,
        values={
            "backend": (P8_BACKEND,),
            "direct_tile_m": (0, 32) if direct else (0,),
            "grouped_tile_m": (0, grouped_tile) if not direct else (0,),
            "fc1_pipeline_stages": (2, 3) if direct else (2,),
            "fc1_warps": (4, 8) if direct else (4,),
            "rp2_stride": (192, 128) if direct else (192,),
            "mac": macs,
        },
        predicates=(stride_fits,),
        efficiency_predicates=(
            # The serving grid policy and the full residency bound; the audit
            # effort races every rung between them.
            lambda p: p["mac"] in (0, top),
        ),
        autotune_effort=query.autotune_effort,
    )


def _validate_p8_query(query, device) -> None:
    del device
    if not isinstance(query, TrellisMxP8Query):
        raise TypeError("query must be a TrellisMxP8Query")


def _validate_p8_config(query, config, device) -> None:
    del device
    if not isinstance(config, P8LaunchConfig):
        raise TypeError("config must be a P8LaunchConfig")
    if config.backend != P8_BACKEND:
        raise ValueError(f"unsupported moe.decode backend {config.backend!r}")
    routed = min(int(query.num_tokens), int(query.direct_tokens)) * int(query.top_k)
    if routed and config.rp2_stride <= routed:
        raise ValueError(
            f"the RP2 carrier stride {config.rp2_stride} must exceed the {routed} "
            "routed rows of a direct call"
        )
    if int(query.mac_pin) and config.mac:
        raise ValueError(
            f"the runtime pins its resident-grid cap to {int(query.mac_pin)}; a "
            f"launch configuration cannot name another ({config.mac})"
        )


#: The P8 component's tuning contract: the launch configuration a capacity
#: races, measured on the complete invocation like every other b12x component.
P8_TUNING = TuningContract(
    component_id="moe.decode",
    query_schema_version=36,
    config_schema_version=32,
    query_fields=frozenset(field.name for field in fields(TrellisMxP8Query)),
    config_fields=frozenset(field.name for field in fields(P8LaunchConfig)),
    encode_query=lambda query: {
        field.name: getattr(query, field.name) for field in fields(TrellisMxP8Query)
    },
    encode_config=P8LaunchConfig.to_dict,
    decode_config=P8LaunchConfig.from_config,
    validate_query=_validate_p8_query,
    validate_config=_validate_p8_config,
    default_config=lambda query, device: P8LaunchConfig.serving(),
    knobs=_P8_KNOBS,
    parameters=_p8_parameter_space,
)


@dataclass(frozen=True)
class _P8Scratch:
    specs: tuple

    def scratch_specs(self) -> tuple:
        return self.specs


def _scratch_specs(
    device: torch.device,
    spec: P8ProgramSpec,
    tokens: int,
    config: P8LaunchConfig | None,
) -> tuple:
    """The caller-owned arena one capacity's calls need, from the runtime's layout."""

    return (
        scratch_buffer_spec(
            "trellismx_p8",
            nbytes=p8_capacity_scratch_nbytes(launch_spec(spec, config), int(tokens)),
            device=device,
            alignment=P8_ARENA_ALIGNMENT,
        ),
    )


_BIND_ARGUMENTS = frozenset(
    {
        "experts",
        "a",
        "topk_ids",
        "topk_weights",
        "output",
        "scratch",
        "input_scales_static",
        "fast_math",
        "activation_amax",
        "layer_idx",
        "route_expert_map",
        "output_expert_map",
        "scales_expanded",
        "a4_prefill",
    }
)


@dataclass(frozen=True, eq=False)
class _TrellisMxP8State:
    """Prepared state of one P8-served capacity.

    ``config`` is the launch configuration the race selected (the serving one
    when the plan is not tuned), and ``scratch`` is the caller-owned arena every
    call of this capacity carves its buffers out of.
    """

    experts: PreparedExperts
    max_tokens: int
    config: P8LaunchConfig
    scratch: _P8Scratch

    @property
    def attachment(self) -> TrellisMxP8Attachment | TrellisMxP8Descriptor:
        return trellismx_p8_attachment(self.experts)

    def bind(self, **kwargs) -> TPMoEFP4Binding:
        unknown = set(kwargs) - _BIND_ARGUMENTS
        if unknown:
            raise TypeError(f"unexpected P8 binding arguments: {sorted(unknown)}")
        experts = kwargs.get("experts", self.experts)
        if experts is not self.experts and experts is not self.experts._impl:
            raise ValueError("prepared experts differ from the plan declaration")
        if kwargs.get("a4_prefill") is True:
            raise ValueError("the P8 runtime has no A4 prefill precision")
        if kwargs.get("fast_math") is False:
            raise ValueError(
                "the P8 runtime's arithmetic is fixed; fast_math=False cannot be honored"
            )
        for name in ("activation_amax", "route_expert_map", "output_expert_map"):
            if kwargs.get(name) is not None:
                raise ValueError(f"the P8 runtime does not accept {name}")
        entry = self.attachment
        if entry is None:
            raise RuntimeError("the prepared P8 state's experts own no P8 runtime")
        entry.check_runtime()
        contract = entry.contract
        a = kwargs["a"]
        if not isinstance(a, torch.Tensor) or a.ndim != 2:
            raise TypeError("fused MoE binding requires a rank-two activation tensor")
        if int(a.shape[0]) > self.max_tokens:
            raise ValueError(
                f"input tokens {int(a.shape[0])} exceed the planned P8 capacity "
                f"{self.max_tokens}"
            )
        if kwargs.get("scratch") is None:
            raise ValueError(
                "the P8 runtime takes its scratch from the caller: bind the "
                "plan's scratch buffer"
            )
        arena = scratch_tensor(
            kwargs["scratch"], self.scratch.scratch_specs(), owner="TrellisMX P8"
        )
        impl = self.experts._impl
        binding = TPMoEFP4Binding(
            a=a,
            experts=impl,
            topk_weights=kwargs["topk_weights"],
            topk_ids=kwargs["topk_ids"],
            implementation=P8_BACKEND,
            state_E=impl.num_experts,
            weight_E=impl.num_experts,
            max_rows=self.max_tokens * contract.top_k,
            k=impl.hidden_size,
            n=impl.intermediate_size,
            num_topk=contract.top_k,
            device=a.device,
            dtype=torch.bfloat16,
            output=kwargs.get("output"),
            input_scales_static=bool(kwargs.get("input_scales_static", False)),
            fast_math=kwargs.get("fast_math"),
            quant_mode="w4a8_mx",
            swiglu_limit=_SWIGLU_LIMIT,
            layer_idx=kwargs.get("layer_idx"),
            routed_rows_capacity=self.max_tokens * contract.top_k,
            scales_expanded=bool(kwargs.get("scales_expanded", False)),
            compiler_scratch=arena,
            p8_launch=self.config,
        )
        from ._impl import _validate_trellismx_binding

        _validate_trellismx_binding(binding, top_k=contract.top_k)
        return binding

    def run(self, binding):
        return binding.run()

    def uses_expanded_nvfp4_scales(self, **kwargs) -> bool:
        if not 0 < int(kwargs.get("num_tokens", 0)) <= self.max_tokens:
            raise ValueError("token count exceeds prepared MoE capacity")
        return False

    def route(self, binding):
        raise NotImplementedError("the P8 runtime serves fused_moe.run bindings only")

    def run_sparse(self, binding):
        raise NotImplementedError("the P8 runtime serves fused_moe.run bindings only")


@dataclass(frozen=True, eq=False)
class _TrellisMxP8CapacityState:
    variants: MappingProxyType

    def for_tokens(self, tokens, **kwargs):
        from ._preparation import variant_for

        return variant_for(self.variants, tokens)

    def bind(self, **kwargs):
        activations = kwargs.get("a")
        if not isinstance(activations, torch.Tensor) or activations.ndim != 2:
            raise TypeError("fused MoE binding requires a rank-two activation tensor")
        return self.for_tokens(activations.shape[0]).bind(**kwargs)

    def run(self, binding):
        return binding.run()


def _direct_tokens(spec: P8ProgramSpec) -> int:
    """The largest row count the runtime keeps on its RP2 direct arm."""

    if not spec.small_m_scheduler:
        return 0
    if spec.p8_input_rowpack:
        return max(1, int(spec.rp2_direct_max_tokens))
    return 1


def plan_p8_execution(
    experts: PreparedExperts,
    *,
    capacity,
    routing,
    invocation: FrozenMapping,
    override,
    scale_expansion,
):
    """``fused_moe.plan_execution`` for a bundle served by a P8 runtime.

    The same plan for a live attachment and for the descriptor of a
    deserialized bundle: one compile job per raced launch configuration, the
    caller-owned scratch each capacity's calls need (sized from the runtime's
    own layout), the runtime's resident memory keyed by its attach token, and a
    query carrying the backend pin and the resolved options. ``override`` is a
    :class:`P8LaunchConfig` pinning the launch configuration, which skips the
    race.
    """

    entry = trellismx_p8_attachment(experts)
    if entry is None:
        raise ValueError("the bundle owns no P8 runtime")
    entry.check_runtime()
    contract = entry.contract
    spec = entry.program_spec()
    if override is not None and not isinstance(override, P8LaunchConfig):
        raise ValueError(
            "MoE decode configs do not apply to a P8-served bundle; pass a "
            f"P8LaunchConfig (got {type(override).__name__})"
        )
    if scale_expansion is not None:
        raise ValueError("a P8-served bundle has no compressed scales to expand")
    if capacity.top_k != contract.top_k:
        raise ValueError(
            f"the P8 runtime was built for top_k={contract.top_k}; the capacity "
            f"declares {capacity.top_k}"
        )
    if capacity.route_num_experts not in (None, 0, contract.num_experts):
        raise ValueError("the P8 runtime does not accept expert route maps")
    if routing is not None:
        if routing.apply_router_weight_on_input:
            raise ValueError(
                "the P8 runtime applies router weights in its reducer only"
            )
        if routing.collect_activation_amax:
            raise ValueError("the P8 runtime cannot collect activation amax")
    if invocation.get("a4_prefill") is True:
        raise ValueError("the P8 runtime has no A4 prefill precision")
    if invocation.get("fast_math", True) is False:
        raise ValueError(
            "the P8 runtime's arithmetic is fixed; fast_math=False cannot be honored"
        )
    activation = experts.plan.activation
    if (
        activation.nonlinearity != contract.activation
        or activation.swiglu_limit != contract.swiglu_limit
        or activation.swiglu_clamp_after_silu != contract.swiglu_clamp_after_silu
        or activation.swiglu_alpha not in (None, 1.0)
        or activation.swiglu_beta not in (None, 0.0)
        or activation.io_dtype != contract.io_dtype
    ):
        raise ValueError(
            "the bundle's activation contract differs from the attached P8 runtime"
        )

    counts = tuple(sorted({capacity.max_tokens, *capacity.warmup_token_counts}))
    direct_tokens = _direct_tokens(spec)

    def child(tokens: int) -> Plan:
        query = TrellisMxP8Query(
            backend=entry.backend.identity,
            package=entry.backend.package,
            options=entry.options.identity,
            num_experts=contract.num_experts,
            hidden_size=contract.hidden_size,
            intermediate_size=contract.intermediate_size,
            top_k=contract.top_k,
            num_tokens=int(tokens),
            bits=contract.bits,
            direct_tokens=direct_tokens,
            mac_pin=0 if spec.mac_override is None else int(spec.mac_override),
        )

        def jobs(config, device):
            del device
            return entry.compile_jobs(tokens, config)

        def memory(config, device):
            resident = entry.resident_nbytes
            return MemoryRequirements(
                # The arena lives where the runtime executes: on the bundle's
                # device, which is the one the attachment verified.
                scratch=_scratch_specs(contract.device, spec, tokens, config),
                # Already allocated by the runtime; one key per attachment, so
                # every capacity planned for this layer counts it once.
                persistent=(
                    PersistentMemory(
                        ("moe.trellismx_p8", entry.backend.identity, entry.token),
                        resident,
                        resident,
                    ),
                ),
            )

        def materialize(selection, device):
            config = P8LaunchConfig.serving() if selection is None else selection.config
            if not isinstance(config, P8LaunchConfig):
                raise TypeError("a P8 plan materializes a P8LaunchConfig")
            attachment = entry.require_live()
            if attachment.programs is not None:
                factory = attachment.programs[0]
                ordinal = device.ordinal
                programs = CompileJob.create(
                    factory,
                    launch_spec(spec, config),
                    int(tokens),
                    CompileJob.DEVICE_ORDINAL if ordinal is None else int(ordinal),
                ).invoke()
                load_programs(programs)
                attachment.runtime.install_compiled_programs(programs, tokens, config)
            return _TrellisMxP8State(
                experts=experts,
                max_tokens=int(tokens),
                config=config,
                scratch=_P8Scratch(
                    _scratch_specs(contract.device, spec, tokens, config)
                ),
            )

        return Plan(
            contract=P8_TUNING,
            query=query,
            invocation=invocation,
            _compile_jobs=jobs,
            _memory_requirements=memory,
            _materialize=materialize,
            override=override,
            _device=experts.device,
        )

    children = {tokens: child(tokens) for tokens in counts}
    if len(counts) == 1:
        return children[counts[0]]

    def assemble(states, device):
        del device
        return _TrellisMxP8CapacityState(
            MappingProxyType({tokens: states[tokens] for tokens in counts})
        )

    return _CompositePlan(
        component_id="moe.decode",
        capacity_metadata=FrozenMapping(
            {
                "token_counts": counts,
                "top_k": capacity.top_k,
                "backend": P8_BACKEND,
                "invocation": invocation,
            }
        ),
        variants=children,
        _assemble=assemble,
    )
