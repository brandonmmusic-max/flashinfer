"""Experimental TP-local P8 procedural-NPK MoE runtime for GLM-5.3.

The P8 kernels and this runtime are Brandon M. Music's, from
local-inference-lab/b12x#342 (brandonmmusic-max/b12x, branch
review/trellismx-p8-native-20260907), ported into this package with the same
kernels, numerics, tile policy and arms. The port changed imports, takes its
behavior settings from :class:`P8RuntimeOptions` instead of the process
environment, and adds :class:`P8TensorSource`, the caller-owned ``scratch`` and
``out`` of ``__call__``, and the per-call :class:`P8LaunchConfig` a prepared
plan races.

This is a direct device path: a K3, K4 or K5 trellis stream is decoded to E4M3 inside
the MMA kernel and physical UE8M0/32 scales are consumed by the tensor core.
It implements the frozen TP4 identity-boundary P8 contract and an opt-in M1
H128 suh/svh scale component for a GLM routed layer whose sidecar carries the
matching immutable layer identity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.compiler import OptLevel
from cutlass.cute.runtime import make_ptr
from safetensors import safe_open
from .launch import P8LaunchConfig
from .plan import (
    P8ProgramSpec,
    arm_key as _arm_key,
    launch_spec,
    p8_capacity_launch_rows,
    p8_capacity_scratch_nbytes,
    p8_scratch_layout_for,
    select_arm as _select_arm,
)
from .p8_coupled_scales import (
    COUPLED_SCHEMA as P8_COUPLED_SCHEMA,
    SCALE_NAMES,
    SCHEMA as P8_SCALE_COMPONENT_SCHEMA,
    validate_coupled_component,
    validate_scale_component,
)
from .scratch import (
    INTERMEDIATE,
    P8_ARENA_ALIGNMENT,
    ROUTE_OUTPUT,
    p8_clear_spans,
)
from .tile_policy import resolve_grid_cap

from b12x._lib.compiler import (
    KernelCompileSpec,
    compile as b12x_compile,
)
from b12x._lib.compile_plan import attach_programs, program_keys
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import scratch_buffer_spec, scratch_tensor
from b12x.moe._shared.p8.kernels.route_hoist_dynamic import P8DynamicKernelBackend
from b12x.moe._shared.p8.kernels.w4a8_launch import P8DynamicMoEW4A8Launch
from b12x.moe.fused_moe._impl import (
    _e8m0_scale_to_w4a8_sfb_inplace,
    _launch_dynamic_topk_sum,
)
from b12x._lib.utils.device import current_cuda_stream, get_hardware_info
from b12x.moe._shared.p8.options import GLM53_SERVING_OPTIONS, P8RuntimeOptions

_LOG = logging.getLogger(__name__)


def get_max_active_clusters(cluster_size: int) -> int:
    """Occupancy-bounded active clusters, queried as the P8 kernels were tuned.

    b12x's shared helper answers a one-CTA cluster with the SM count instead;
    the monolithic P8 arms keep the occupancy query.
    """
    return get_hardware_info().get_max_active_clusters(cluster_size)


def _gptr(dtype, tensor: torch.Tensor, align: int = 16):
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


def _fake_i32(shape: tuple[int, ...]):
    return cute.runtime.make_fake_compact_tensor(cutlass.Int32, shape, assumed_align=4)


def _fake_f32(shape: tuple[int, ...]):
    return cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, shape, assumed_align=16
    )


@dataclass(frozen=True)
class _CompiledArm:
    compiled: object
    tile_m: int
    materialized: bool
    mac: int


@dataclass(frozen=True)
class _CompiledPrograms:
    arms: tuple[tuple[tuple[bool, bool, int, int, int, int], _CompiledArm], ...]
    reducer: object


#: Importable compile-job factory for the programs a runtime call launches.
P8_PROGRAM_FACTORY = __name__ + ":compile_p8_programs"


def _build_arm(
    spec: P8ProgramSpec, key: tuple[bool, bool, int, int, int, int]
) -> _CompiledArm:
    """Compile (or, while b12x plans compilations, describe) one arm."""

    materialized, small_m, tile_m, rp2_stride, _, _ = key
    mac = (
        spec.mac_override
        if spec.mac_override is not None
        else (64 if materialized else int(get_max_active_clusters(1)))
    )
    kernel = P8DynamicKernelBackend(
        16,
        (tile_m, 128),
        activation="silu",
        quant_recipe="w4a8_trellis",
        w4a8_repacked=True,
        num_topk=spec.topk,
        trellis_bits=spec.trellis_bits,
        trellis_codebook="procedural_npk",
        trellis_scaled=True,
        trellis_identity_boundary=not spec.full_coupled,
        direct_routing=small_m,
        materialize_intermediate=materialized,
        p8_small_m=small_m,
        p8_fc1_tile_n=spec.fc1_tile_n if small_m else 128,
        p8_scale_sandwich=spec.scale_sandwich,
        p8_full_coupled=spec.full_coupled,
        share_input_across_experts=materialized,
        deterministic_output=spec.deterministic_output,
        swiglu_limit=spec.swiglu_limit,
    )
    if spec.p8_down_remainder:
        if not spec.full_coupled:
            raise RuntimeError("D-x2 is implemented for the full-coupled P8 paths only")
        if spec.fc1_warp_quant:
            raise RuntimeError(
                "D-x2 needs fc1_warp_quant=False: the warp-quant FC1 epilogue has no remainder writer"
            )
        from b12x.moe._shared.p8.kernels.p8_down_remainder import (
            enable_down_remainder,
        )

        enable_down_remainder(kernel, spec.dx2_phases)
    if spec.p8_dx2_rowpack and small_m:
        if not spec.full_coupled:
            raise RuntimeError(
                "D-x2-RP is implemented for the full-coupled P8 paths only"
            )
        if spec.fc1_warp_quant:
            raise RuntimeError(
                "D-x2-RP/RP2 need fc1_warp_quant=False: the warp-quant FC1 epilogue has no remainder writer"
            )
        if spec.tile_major_tasks:
            raise RuntimeError(
                "D-x2-RP/RP2 need one route per direct tile (tile_major_tasks=False)"
            )
        if spec.p8_input_rowpack and not spec.fc1_broadcast_a:
            raise RuntimeError(
                "D-x2-RP2 needs fc1_broadcast_a=True: otherwise ordinary FC1 staging also writes A row 8"
            )
        from b12x.moe._shared.p8.kernels.p8_down_remainder import enable_rowpack

        enable_rowpack(kernel, input_hop=spec.p8_input_rowpack)
        # Keep high and residual carriers disjoint for every enabled row.
        kernel.p8_input_rowpack_stride = rp2_stride
        kernel.materialized_phase1_kernel.p8_input_rowpack_stride = rp2_stride
    if spec.full_coupled:
        # Both M1 and grouped owners share the scale/sign plane geometry.
        kernel.materialized_phase1_kernel.p8_intermediate = spec.intermediate
        kernel.materialized_phase2_kernel.p8_intermediate = spec.intermediate
        if small_m:
            kernel.materialized_phase1_kernel.p8_tile_major = spec.tile_major_tasks
            kernel.materialized_phase2_kernel.p8_tile_major = spec.tile_major_tasks
            fc1 = kernel.materialized_phase1_kernel
            fc1.p8_a_swizzle_rotate = spec.fc1_a_swizzle_rotate
            fc1.p8_warp_quant = spec.fc1_warp_quant
            fc1.p8_epi_par = spec.p8_epi_par
            fc1.p8_exact_staging = spec.fc1_exact_staging
            # Direct owner invokes _run_task with valid_rows=1. Never
            # apply this specialization to grouped multi-row owners.
            fc1.p8_broadcast_a = spec.fc1_broadcast_a
            fc1.num_warps = spec.fc1_warps
            fc1.threads_per_cta = 32 * spec.fc1_warps
            fc1.p8_n8_per_warp = 16 // spec.fc1_warps
            fc1.owned_row_groups = fc1.tile_m // spec.fc1_warps
            fc1.p8_pipeline_stages = spec.fc1_pipeline_stages
            fc1.shared_bytes = max(
                fc1.shared_bytes, spec.fc1_pipeline_stages * fc1.stage_bytes
            )
            fc1.shared_words = (fc1.shared_bytes + 3) // 4
            fc1.trellis_lut_offset = fc1.shared_bytes
    launch = P8DynamicMoEW4A8Launch(
        kernel,
        k=spec.hidden,
        n=spec.intermediate,
        w1_n=2 * spec.intermediate,
        num_topk=spec.topk,
    )

    def ptr(dtype, address: int, align: int = 16):
        return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)

    def fake_ptr_u8():
        return ptr(cutlass.Uint8, 16)

    def fake_ptr_i32():
        return ptr(cutlass.Int32, 4, 4)

    def fake_ptr_u32():
        return ptr(cutlass.Uint32, 16)

    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float4E2M1FN,
        (2 * spec.intermediate, spec.hidden, spec.experts),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    b_w2_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float4E2M1FN,
        (spec.hidden, spec.intermediate, spec.experts),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    compiled = b12x_compile(
        launch,
        ptr(cutlass.BFloat16, 16),
        fake_ptr_i32(),
        ptr(cutlass.Float32, 4, 4),
        ptr(cutlass.Float4E2M1FN, 16),
        ptr(cutlass.Float8E4M3FN, 16),
        fake_ptr_u8(),
        fake_ptr_u8(),
        fake_ptr_u32(),
        _fake_i32((1,)),
        _fake_i32((1,)),
        _fake_i32((1,)),
        _fake_i32((1,)),
        _fake_i32((1,)),
        _fake_i32((1,)),
        _fake_i32((1,)),
        fake_ptr_i32(),
        fake_ptr_i32(),
        fake_ptr_i32(),
        fake_ptr_i32(),
        fake_ptr_i32(),
        fake_ptr_i32(),
        fake_ptr_i32(),
        b_w13_fake,
        ptr(cutlass.Float8E4M3FN, 16),
        b_w2_fake,
        ptr(cutlass.Float8E4M3FN, 16),
        fake_ptr_u8(),
        fake_ptr_u8(),
        fake_ptr_u8(),
        fake_ptr_u8(),
        fake_ptr_u32(),
        fake_ptr_u32(),
        fake_ptr_u32(),
        fake_ptr_u32(),
        _fake_i32((spec.experts,)),
        _fake_i32((spec.experts,)),
        _fake_i32((spec.experts + 1,)),
        _fake_f32((spec.experts,)),
        _fake_f32((spec.experts,)),
        _fake_f32((spec.experts,)),
        _fake_f32((spec.experts,)),
        ptr(
            cutlass.Float32 if spec.full_coupled else cutlass.BFloat16,
            16,
        ),
        fake_ptr_i32(),
        ptr(cutlass.Float32, 16),
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        fake_ptr_u8(),
        ptr(cutlass.Float16, 16),
        # The compile spec is the JIT cache key and includes every specialized
        # field, especially stored rate, topology and epilogue dimensions. It
        # carries no tensor-parallel rank: no arm kernel receives one, so the
        # four ranks of a layer compile and race one identical program.
        compile_spec=KernelCompileSpec.from_fields(
            "glm53.p8.native.tp",
            4,
            ("fc1_row_alias376", 1),
            ("requested_m_regime_direct_policy", 1),
            ("fc1_route_hoist", 1),
            ("tile_m", tile_m),
            ("trellis_bits", spec.trellis_bits),
            ("npk_k5_funnel", int(small_m and spec.trellis_bits == 5)),
            ("fc2_carveout100_grid564", int(small_m)),
            ("fc2_k5_funnel", int(small_m and spec.trellis_bits == 5)),
            ("grouped_fc2_grid376", int(not small_m)),
            ("tile_major_tasks", int(spec.tile_major_tasks and small_m)),
            ("fc1_pipeline_stages", spec.fc1_pipeline_stages if small_m else 2),
            ("fc1_warps", spec.fc1_warps if small_m else 4),
            ("fc1_a_swizzle_rotate", int(spec.fc1_a_swizzle_rotate and small_m)),
            ("fc1_warp_quant", int(spec.fc1_warp_quant and small_m)),
            ("fc1_exact_staging", int(spec.fc1_exact_staging and small_m)),
            ("fc1_broadcast_a", int(spec.fc1_broadcast_a and small_m)),
            ("materialized", int(materialized)),
            ("small_m_scheduler", int(small_m)),
            ("fc1_tile_n", spec.fc1_tile_n if small_m else 128),
            ("experts", spec.experts),
            ("hidden", spec.hidden),
            ("intermediate", spec.intermediate),
            ("topk", spec.topk),
            ("scaled", 1),
            ("identity", int(not spec.full_coupled)),
            ("scale_sandwich", int(spec.scale_sandwich)),
            ("full_coupled", int(spec.full_coupled)),
            ("codebook", "procedural_npk"),
            ("deterministic_output", int(spec.deterministic_output)),
            ("down_remainder", int(spec.p8_down_remainder)),
            (
                "down_remainder_phases",
                spec.dx2_phases if spec.p8_down_remainder else "none",
            ),
            ("dx2_rowpack", int(spec.p8_dx2_rowpack and small_m)),
            ("dx2_input_rowpack", int(spec.p8_input_rowpack and small_m)),
            (
                "rp2_input_row_stride",
                rp2_stride if spec.p8_input_rowpack and small_m else 0,
            ),
            ("epi_par", int(spec.p8_epi_par and small_m)),
        ),
        # Every arm is a pure function of this key.
    )
    return attach_programs(
        _CompiledArm(
            compiled=compiled, tile_m=tile_m, materialized=materialized, mac=mac
        ),
        compiled,
    )


def _build_coupled_reducer(spec: P8ProgramSpec):
    """Compile (or describe) the full-coupled deterministic top-k reducer."""

    from b12x.moe._shared.p8.kernels.p8_coupled_topk import P8CoupledTopKSumKernel

    compiled = b12x_compile(
        P8CoupledTopKSumKernel(topk=spec.topk, hidden=spec.hidden),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        1,
        current_cuda_stream(),
        # Rank-free for the same reason as the arm above: the reducer reduces
        # a rank-local route output and receives no rank.
        compile_spec=KernelCompileSpec.from_fields(
            "glm53.p8.native.coupled_topk",
            2,
            ("topk", spec.topk),
            ("hidden", spec.hidden),
            ("route_dtype", "fp32"),
        ),
    )
    return compiled


@program_cache(scope="preparation")
def compile_p8_programs(spec: P8ProgramSpec, tokens: int, ordinal: int):
    """Compile-job factory: the programs any runtime call with 1 to ``tokens`` rows launches.

    The arm of every such launch and the deterministic reducer, built by the code
    the runtime builds them with, so b12x preparation declares, compiles and checks
    the programs the runtime launches for every call a capacity of ``tokens`` rows
    accepts. Only full-coupled deterministic runtimes are attached to prepared
    experts, so only they declare programs.
    """

    if int(tokens) < 1:
        raise ValueError("a capacity declares at least one row")

    if not spec.full_coupled or not spec.deterministic_output:
        raise ValueError(
            "compile_p8_programs declares only the programs of full-coupled "
            "deterministic runtimes"
        )
    with torch.cuda.device(int(ordinal)):
        arms = {}
        for rows in p8_capacity_launch_rows(spec, tokens):
            materialized, small_m = _select_arm(spec, rows)
            key = _arm_key(spec, materialized, small_m, rows)
            arms[key] = _build_arm(spec, key)
        reducer = _build_coupled_reducer(spec)
    return attach_programs(
        _CompiledPrograms(tuple(arms.items()), reducer),
        *(arm for _, arm in arms.items()),
        reducer,
    )


class P8TensorSource:
    """A sidecar's tensors and metadata, already read by their host.

    ``safe_open`` reopens the file and copies every tensor; a host that has
    already loaded a sidecar hands the tensors here instead, and the runtime
    reads them through the same interface. Nothing is copied or moved, so a
    tensor on a device stays on it.
    """

    def __init__(self, tensors, metadata):
        self._tensors = dict(tensors)
        self._metadata = None if metadata is None else dict(metadata)

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        return False

    def keys(self):
        return tuple(self._tensors)

    def metadata(self):
        return self._metadata

    def get_tensor(self, name):
        return self._tensors[name]


def _validate_routing_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, int],
    device: torch.device,
    dtypes: tuple[torch.dtype, ...],
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device != device
        or tensor.dtype not in dtypes
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
        or tensor.data_ptr() % 4
    ):
        raise RuntimeError(
            f"P8 native routing contract mismatch: {name} "
            f"{getattr(tensor, 'dtype', type(tensor).__name__)} "
            f"{tuple(getattr(tensor, 'shape', ()))} on "
            f"{getattr(tensor, 'device', None)}"
        )


class P8NativeTPMoE:
    """Own one TP rank's physical P8 payload and launch compiled MoE kernels.

    ``options`` (:class:`P8RuntimeOptions`; default the GLM-5.3 Flash
    serving configuration) holds every behavior setting. They are resolved once, here,
    into the attributes b12x's P8 attachment records
    (``trellismx_p8.P8_OPTION_ATTRIBUTES``); nothing is read from the process
    environment. The launch geometry is not an option: a call names it with a
    :class:`P8LaunchConfig`, and a call that names none runs the serving one.
    """

    def __init__(
        self,
        sidecar: Path | tuple[Path, Path] | P8TensorSource,
        *,
        device: torch.device,
        tp_rank: int,
        layer: int = 3,
        expected_design_sha256: str | None = None,
        expected_transform_sha256: str | None = None,
        topk: int = 8,
        hidden: int = 4096,
        intermediate: int = 512,
        swiglu_limit: float = 10.0,
        world_size: int = 4,
        tp4_parent_sha256: tuple[str, str] | None = None,
        options: P8RuntimeOptions | None = None,
    ) -> None:
        options = GLM53_SERVING_OPTIONS if options is None else options
        if not isinstance(options, P8RuntimeOptions):
            raise TypeError(
                f"options must be a P8RuntimeOptions; got {type(options).__name__}"
            )
        self.options = options
        force_materialized = options.force_materialized
        mac_override = options.mac_override
        deterministic_output = options.deterministic_output
        small_m_scheduler = options.small_m_scheduler
        fc1_tile_n = options.fc1_tile_n
        debug_capture = options.debug_capture
        compact_scale_storage = options.compact_scale_storage
        compact_input_storage = options.compact_input_storage
        prefill_chunk_tokens = options.prefill_chunk_tokens
        grid_policy = options.grid_policy
        tile_major_tasks = options.tile_major_tasks
        fc1_a_swizzle_rotate = options.fc1_a_swizzle_rotate
        fc1_warp_quant = options.fc1_warp_quant
        fc1_exact_staging = options.fc1_exact_staging
        fc1_broadcast_a = options.fc1_broadcast_a
        self.device = torch.device(device)
        # D-x2: captured once; joins the compile spec below.
        self.p8_down_remainder = options.down_remainder == "dx2"
        self._dx2_planes = 2 if self.p8_down_remainder else 1
        self._dx2_phases = options.down_remainder_phases
        self.p8_dx2_rowpack = options.down_remainder in ("rp", "rp2")
        self.p8_input_rowpack = options.down_remainder == "rp2"
        # Keep speculative verify batches on the RP2 direct kernels when enabled.
        self.rp2_direct_max_tokens = int(options.rp2_direct_max_tokens)
        self.rp2_skip_input_clear = options.rp2_skip_input_clear
        # The launch geometry belongs to ``P8LaunchConfig``: a prepared plan
        # races it and every prepared call names the winner, so the runtime
        # carries no geometry option. These attributes are the serving
        # configuration's values, which a call that names no configuration uses.
        serving = P8LaunchConfig.serving()
        self.rp2_direct_tile_m = serving.direct_tile_m
        self.fc1_pipeline_stages = serving.fc1_pipeline_stages
        self.fc1_warps = serving.fc1_warps
        if not 0 <= self.rp2_direct_max_tokens <= 192:
            raise ValueError("rp2_direct_max_tokens must be in 0..192")
        # EPI-PAR: bit-identical parallel small-M FC1 epilogue; joins the compile spec below.
        self.p8_epi_par = options.epi_par
        if self.p8_epi_par:
            _LOG.debug("P8_EPI_PAR_ACTIVE layer=%s rank=%s", layer, tp_rank)
        if self.p8_dx2_rowpack:
            _LOG.debug(
                "P8_DX2_ROWPACK_ACTIVE layer=%s rank=%s input_hop=%d",
                layer,
                tp_rank,
                int(self.p8_input_rowpack),
            )
        self.tile_major_tasks = bool(tile_major_tasks)
        self.fc1_exact_staging = bool(fc1_exact_staging)
        self.fc1_broadcast_a = bool(fc1_broadcast_a)
        self.fc1_a_swizzle_rotate = bool(fc1_a_swizzle_rotate)
        self.fc1_warp_quant = bool(fc1_warp_quant)
        self.grid_policy = bool(grid_policy)
        self.prefill_chunk_tokens = int(prefill_chunk_tokens)
        if self.prefill_chunk_tokens < 0 or self.prefill_chunk_tokens % 64:
            raise ValueError(
                "P8 prefill chunk must be zero or a positive multiple of 64"
            )
        self.compact_scale_storage = bool(compact_scale_storage)
        self.compact_input_storage = bool(compact_input_storage)
        self.tp_rank = int(tp_rank)
        self.world_size = int(world_size)
        if self.world_size != 4 or self.tp_rank not in range(4):
            raise ValueError(
                "P8 native supports TP4 only; TP2 validators are unsupported"
            )
        self.layer = int(layer)
        if not 3 <= self.layer <= 45:
            raise ValueError(
                "P8 native layer must be in GLM routed layers 3..44 or MTP45"
            )
        self.topk = int(topk)
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.swiglu_limit = float(swiglu_limit)
        self.force_materialized = force_materialized
        self.mac_override = None if mac_override is None else int(mac_override)
        self.deterministic_output = bool(deterministic_output)
        # Explicit developmental opt-in. M2/M3 and prefill retain baseline
        # selection; this is not enabled through a serving environment flag.
        self.small_m_scheduler = bool(small_m_scheduler)
        self.fc1_tile_n = int(fc1_tile_n)
        self.debug_capture = bool(debug_capture)
        self.debug_tensors = {}
        if self.fc1_tile_n not in (32, 64, 128):
            raise ValueError("FC1 tile N must be 32, 64, or 128")
        if self.fc1_tile_n != 128 and (
            not self.small_m_scheduler or self.swiglu_limit != 10.0
        ):
            raise ValueError("Narrow FC1 requires small-M and SwiGLU limit 10")
        if self.small_m_scheduler and (
            not self.deterministic_output
            or force_materialized is not None
            or (topk, hidden, intermediate) != (8, 4096, 2048 // self.world_size)
        ):
            raise ValueError(
                "P8 small-M requires deterministic GLM TP4 and automatic fallback"
            )
        if self.mac_override is not None and self.mac_override <= 0:
            raise ValueError("mac_override must be positive")
        if tp4_parent_sha256 is not None:
            if self.world_size != 2:
                raise ValueError("Parent-pair adapter requires TP2")
            from glm53_nvfp4.p8_tp2_repack import open_tp2_pair

            source = open_tp2_pair(
                sidecar, tp4_parent_sha256, layer=self.layer, rank=self.tp_rank
            )
        elif isinstance(sidecar, P8TensorSource):
            source = sidecar
        else:
            source = safe_open(sidecar, framework="pt", device="cpu")
        with source as src:
            metadata = src.metadata() or {}
            schema = metadata.get("schema")
            source_design_sha256 = metadata.get("source_design_sha256")
            bits_text = metadata.get("bits", "")
            if bits_text not in {"3", "4", "5"}:
                raise RuntimeError(f"invalid P8 trellis rate: {bits_text!r}")
            self.trellis_bits = int(bits_text)
            base_required = {
                "layer": str(self.layer),
                "rank": str(self.tp_rank),
                "world_size": str(self.world_size),
                "bits": bits_text,
                "alphabet": "e4m3",
                "scale": "ue8m0-k32",
                "law": "procedural-mcg-alpha2",
                "ldlq": "false",
            }
            identity_schema = schema in {
                "glm53-p8-identity-mcg-tp4-rank.v1",
                "glm53-p8-mcg-tp4-rank.v2",
            }
            scale_component_schema = schema == P8_SCALE_COMPONENT_SCHEMA
            full_coupled_schema = schema == P8_COUPLED_SCHEMA.replace(
                "tp4", f"tp{self.world_size}"
            )
            if self.world_size == 2 and not full_coupled_schema:
                raise RuntimeError("TP2 port only supports the full-coupled schema")
            if (
                not (identity_schema or scale_component_schema or full_coupled_schema)
                or any(
                    metadata.get(key) != value for key, value in base_required.items()
                )
                or (identity_schema and metadata.get("boundary") != "identity")
            ):
                raise RuntimeError(f"invalid P8 native sidecar metadata: {metadata}")
            if full_coupled_schema or schema in {
                "glm53-p8-mcg-tp4-rank.v2",
                P8_SCALE_COMPONENT_SCHEMA,
                P8_COUPLED_SCHEMA,
            }:
                if (
                    not isinstance(source_design_sha256, str)
                    or len(source_design_sha256) != 64
                    or any(
                        char not in "0123456789abcdef" for char in source_design_sha256
                    )
                ):
                    raise RuntimeError("v2 P8 sidecar lacks a valid source design hash")
                if (
                    expected_design_sha256 is not None
                    and source_design_sha256 != expected_design_sha256
                ):
                    raise RuntimeError("P8 sidecar does not match the expected design")
            elif expected_design_sha256 is not None:
                raise RuntimeError(
                    "historical P8 sidecars cannot satisfy a v2 design pin"
                )
            w13 = src.get_tensor("w13_trellis")
            w2 = src.get_tensor("w2_trellis")
            w13_scale = src.get_tensor("w13_scale_ue8m0")
            w2_scale = src.get_tensor("w2_scale_ue8m0")
            scale_tensors = (
                {name: src.get_tensor(name) for name in SCALE_NAMES}
                if scale_component_schema or full_coupled_schema
                else None
            )
        self.source_design_sha256 = source_design_sha256
        experts = int(w13.shape[1])
        stream_words = 16 * self.trellis_bits
        if tuple(w13.shape) != (
            2,
            experts,
            hidden // 16,
            intermediate // 16,
            stream_words,
        ):
            raise RuntimeError(f"unexpected W13 trellis shape {tuple(w13.shape)}")
        if tuple(w2.shape) != (experts, intermediate // 16, hidden // 16, stream_words):
            raise RuntimeError(f"unexpected W2 trellis shape {tuple(w2.shape)}")
        if tuple(w13_scale.shape) != (experts, 2 * intermediate, hidden // 32):
            raise RuntimeError(f"unexpected W13 scale shape {tuple(w13_scale.shape)}")
        if tuple(w2_scale.shape) != (experts, hidden, intermediate // 32):
            raise RuntimeError(f"unexpected W2 scale shape {tuple(w2_scale.shape)}")
        self.experts = experts
        self.scale_component = None
        self.full_coupled = bool(full_coupled_schema)
        if self.compact_scale_storage and not self.full_coupled:
            raise ValueError("compact scales require the external full-coupled owners")
        if self.full_coupled and expected_transform_sha256 is None:
            raise RuntimeError(
                "full-coupled P8 requires an externally pinned encoder transform"
            )
        if scale_tensors is not None:
            validator = (
                validate_coupled_component
                if self.full_coupled
                else validate_scale_component
            )
            # Preserve compatibility with the pinned TP4 validator during
            # isolated storage diagnostics. TP2 requires the ported validator.
            validator_kwargs = (
                {} if self.world_size == 4 else {"world_size": self.world_size}
            )
            if self.full_coupled:
                validator_kwargs["expected_transform_sha256"] = (
                    expected_transform_sha256
                )
            self.scale_component = validator(
                metadata,
                scale_tensors,
                layer=self.layer,
                rank=self.tp_rank,
                experts=experts,
                hidden=hidden,
                intermediate=intermediate,
                **validator_kwargs,
            )
            if not self.small_m_scheduler or self.fc1_tile_n != 128:
                raise RuntimeError("P8 scale component requires the M1 N128 owner path")
        # The trellis storage is byte-for-byte the same size as the packed
        # E2M1 descriptor carrier expected by the inherited W4A8 launch ABI.
        # Alias it for the descriptor-only arguments instead of allocating a
        # second ~0.9 GiB of unread dummy weights per layer and TP rank.  The
        # kernel reads the procedural stream through the uint32 pointers below;
        # it never dereferences the descriptor carrier values.
        w13_stream_storage = w13.to(device=self.device).contiguous()
        w2_stream_storage = w2.to(device=self.device).contiguous()
        self.w13_stream = w13_stream_storage.view(torch.int32).reshape(-1)
        if self.fc1_exact_staging:
            if (
                self.world_size,
                self.experts,
                self.hidden,
                self.intermediate,
                self.fc1_tile_n,
            ) != (4, 288, 4096, 512, 128) or not self.full_coupled:
                raise ValueError(
                    "exact staging requires full-coupled TP4 E288/H4096/I512 N128"
                )
            expected_words = 2 * 288 * 4096 * 512 * self.trellis_bits // 32
            if self.w13_stream.numel() != expected_words:
                raise ValueError("exact staging FC1 stream extent mismatch")
        self.w2_stream = w2_stream_storage.view(torch.int32).reshape(-1)
        w13_scale = w13_scale.to(device=self.device).contiguous()
        w2_scale = w2_scale.to(device=self.device).contiguous()
        # The monolithic kernel consumes the logical [E, N, K/32] UE8M0
        # plane through the sfb_*_mx ABI slots.  The split materialized
        # kernels consume a separately repacked copy through *_sfb_rp.
        # Keep both representations: a one-byte sentinel in the logical slots
        # is an out-of-bounds scale read, not an identity scale.
        self.w13_scale_mx = w13_scale.reshape(-1)
        self.w2_scale_mx = w2_scale.reshape(-1)
        self.w13_sfb = _e8m0_scale_to_w4a8_sfb_inplace(
            w13_scale.clone(),
            weight_E=experts,
            rows=2 * intermediate,
            k_dim=hidden,
            gated_half_rows=intermediate,
        ).reshape(-1)
        self.w2_sfb = _e8m0_scale_to_w4a8_sfb_inplace(
            w2_scale.clone(),
            weight_E=experts,
            rows=hidden,
            k_dim=intermediate,
        ).reshape(-1)
        if self.compact_scale_storage:
            # Full coupling forces external materialized FC1/FC2 for every M.
            # Logical-scale ABI arguments remain non-null aliases, but are
            # not read by those owners. Never enable for a monolithic arm.
            # This is opt-in pending device closure against separate storage.
            self.w13_scale_mx = self.w13_sfb
            self.w2_scale_mx = self.w2_sfb
        # These are descriptor carriers only; they alias the trellis storage
        # above and therefore add zero payload bytes.  Their trailing extent is
        # the PACKED row length, which is bits/8 bytes per weight: hidden // 2
        # only at K4.  Deriving it from the stored rate keeps K4 byte-identical
        # while letting K3 and K5 describe their own shorter or longer rows.
        w13_row_bytes = hidden * self.trellis_bits // 8
        w2_row_bytes = intermediate * self.trellis_bits // 8
        w13_dummy_bytes = experts * 2 * intermediate * w13_row_bytes
        w2_dummy_bytes = experts * hidden * w2_row_bytes
        self.w13_dummy = (
            w13_stream_storage.view(torch.uint8)
            .reshape(-1)[:w13_dummy_bytes]
            .reshape(experts, 2 * intermediate, w13_row_bytes)
        )
        self.w2_dummy = (
            w2_stream_storage.view(torch.uint8)
            .reshape(-1)[:w2_dummy_bytes]
            .reshape(experts, hidden, w2_row_bytes)
        )
        self.sentinel = torch.zeros(1, dtype=torch.uint8, device=self.device)
        # NPK never dereferences the LUT pointer; one zero byte stands in for it.
        self.zero_lut = torch.zeros(1, dtype=torch.uint8, device=self.device)
        self.zero_rotation = torch.zeros(1, dtype=torch.float16, device=self.device)
        self.scale_component_packed = (
            self.scale_component.packed.to(device=self.device)
            if self.scale_component is not None
            else self.zero_rotation
        )
        self.ones = torch.ones(experts, dtype=torch.float32, device=self.device)
        if self.small_m_scheduler and self.experts != 288:
            raise ValueError("P8 small-M requires 288 experts")
        # Every stored rate compiles both owners: the small-M (M1) arm for the
        # row counts it serves and the fused grouped arm for the rest. The
        # grouped owner is the M32-or-M64/N128 arm the tile policy or a raced
        # or pinned ``P8LaunchConfig`` selects: ``select_tile`` gives M32
        # through 128 rows and M64 above, and each tile has its own FC1/FC2
        # programs. Grouped work is never replayed row by row through the M1
        # kernel, and a non-K4 layer takes the same native path as K4 (see
        # ``__call__``).
        self._compiled: dict[tuple[bool, bool, int, int, int, int], _CompiledArm] = {}
        self._coupled_reducer = None
        # Every setting that selects or specializes a launched program, as the
        # metadata a compile job carries: a change to any field is a different
        # program, so planned keys equal launched keys only when both are built
        # from this spec. ``launch_spec`` derives the spec of one raced
        # ``P8LaunchConfig`` from it. ``self.tp_rank`` is deliberately not one:
        # it selects and verifies this rank's sidecar slice, and no launched
        # kernel receives it (see ``P8ProgramSpec``).
        self.program_spec = P8ProgramSpec(
            trellis_bits=self.trellis_bits,
            experts=self.experts,
            hidden=self.hidden,
            intermediate=self.intermediate,
            topk=self.topk,
            swiglu_limit=self.swiglu_limit,
            full_coupled=self.full_coupled,
            scale_sandwich=self.scale_component is not None,
            mac_override=self.mac_override,
            deterministic_output=self.deterministic_output,
            small_m_scheduler=self.small_m_scheduler,
            force_materialized=self.force_materialized,
            rp2_direct_max_tokens=self.rp2_direct_max_tokens,
            rp2_skip_input_clear=self.rp2_skip_input_clear,
            prefill_chunk_tokens=self.prefill_chunk_tokens,
            compact_input_storage=self.compact_input_storage,
            p8_down_remainder=self.p8_down_remainder,
            dx2_phases=self._dx2_phases,
            p8_dx2_rowpack=self.p8_dx2_rowpack,
            p8_input_rowpack=self.p8_input_rowpack,
            p8_epi_par=self.p8_epi_par,
            tile_major_tasks=self.tile_major_tasks,
            fc1_tile_n=self.fc1_tile_n,
            rp2_direct_tile_m=self.rp2_direct_tile_m,
            grouped_tile_m=0,
            rp2_stride=192,
            fc1_warps=self.fc1_warps,
            fc1_pipeline_stages=self.fc1_pipeline_stages,
            fc1_a_swizzle_rotate=self.fc1_a_swizzle_rotate,
            fc1_warp_quant=self.fc1_warp_quant,
            fc1_exact_staging=self.fc1_exact_staging,
            fc1_broadcast_a=self.fc1_broadcast_a,
        )
        # The compile job that declares every program a call launches.
        self.program_declaration = (P8_PROGRAM_FACTORY, self.program_spec)

    def launch_spec(self, config: P8LaunchConfig | None = None) -> P8ProgramSpec:
        """The program spec of one launch configuration."""

        return launch_spec(self.program_spec, config)

    def scratch_specs(
        self, device: torch.device, tokens: int, config: P8LaunchConfig | None = None
    ) -> tuple:
        """The one caller-owned buffer every call of 1 to ``tokens`` rows fits."""

        return (
            scratch_buffer_spec(
                "trellismx_p8",
                nbytes=p8_capacity_scratch_nbytes(self.launch_spec(config), tokens),
                device=torch.device(device),
                alignment=P8_ARENA_ALIGNMENT,
            ),
        )

    def _workspace(self, scratch, layout):
        """The caller's arena, checked against the extents this call needs.

        Missing, wrongly typed, wrongly placed, too small or misaligned storage
        raises; the runtime never allocates one. The launch hands each region's
        pointer to the kernels with ``assumed_align=16``, so an arena whose
        start is not 16-byte aligned would make every region misaligned.
        """

        if scratch is None:
            raise RuntimeError(
                "a P8 call takes its scratch from the caller: bind the plan's "
                "scratch buffer (P8NativeTPMoE.scratch_specs) and pass it"
            )
        return scratch_tensor(
            scratch,
            (
                scratch_buffer_spec(
                    "trellismx_p8",
                    nbytes=layout.nbytes,
                    device=self.device,
                    alignment=P8_ARENA_ALIGNMENT,
                ),
            ),
            owner="P8 native",
        ).narrow(0, 0, layout.nbytes)

    def _compile(
        self,
        spec: P8ProgramSpec,
        materialized: bool,
        small_m: bool = False,
        expected_m: int | None = None,
    ) -> _CompiledArm:
        cache_key = _arm_key(spec, materialized, small_m, expected_m)
        try:
            return self._compiled[cache_key]
        except KeyError as error:
            raise RuntimeError(
                f"P8 arm {cache_key} was not installed before launch"
            ) from error

    def _compile_full_coupled_reducer(self):
        if not self.full_coupled:
            raise RuntimeError("coupled reducer requested for non-coupled P8")
        if self._coupled_reducer is None:
            raise RuntimeError("the P8 coupled reducer was not installed before launch")
        return self._coupled_reducer

    def install_compiled_programs(self, programs, tokens: int, config=None) -> None:
        if not isinstance(programs, _CompiledPrograms):
            raise TypeError(
                "P8 programs must come from compile_p8_programs; got "
                f"{type(programs).__name__}"
            )
        if int(tokens) < 1:
            raise ValueError("a P8 program installation covers at least one row")
        spec = self.launch_spec(config)
        expected = set()
        for rows in p8_capacity_launch_rows(spec, int(tokens)):
            materialized, small_m = _select_arm(spec, rows)
            expected.add(_arm_key(spec, materialized, small_m, expected_m=rows))
        installed = {}
        for key, arm in programs.arms:
            if key not in expected:
                raise RuntimeError(f"P8 program bundle declares unknown arm {key}")
            if key in installed:
                raise RuntimeError(f"P8 program bundle repeats arm {key}")
            if arm.tile_m != key[2] or arm.materialized != key[0]:
                raise RuntimeError(f"P8 program carrier disagrees with arm {key}")
            if not callable(arm.compiled) or not program_keys(arm):
                raise RuntimeError(f"P8 program carrier lacks arm {key}")
            installed[key] = arm
        if set(installed) != expected:
            missing = sorted(expected - set(installed), key=repr)
            raise RuntimeError(f"P8 program bundle lacks arms {missing}")
        if not callable(programs.reducer) or not program_keys(programs.reducer):
            raise RuntimeError("P8 program bundle lacks the coupled reducer")
        self._compiled.update(installed)
        self._coupled_reducer = programs.reducer

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        out: torch.Tensor,
        *,
        scratch: torch.Tensor | object = None,
        config: P8LaunchConfig | None = None,
    ) -> torch.Tensor:
        """Run one call through the arm its row count selects.

        ``scratch`` is the caller's arena (:meth:`scratch_specs` sizes it): every
        buffer the call touches is carved out of it, so a call allocates neither
        scratch nor an output, and a call without a large enough or 16-byte
        aligned arena raises. ``out`` is the caller's output storage and is
        required: the arena holds no result, so a call never returns storage a
        later call overwrites. ``config`` is the launch configuration the
        prepared plan selected.
        """
        if (
            x.dtype != torch.bfloat16
            or x.ndim != 2
            or x.shape[1] != self.hidden
            or x.device != self.device
            or not x.is_contiguous()
            or x.data_ptr() % 16
        ):
            raise RuntimeError(
                f"P8 native input contract mismatch: {x.dtype} {tuple(x.shape)} "
                f"on {x.device}"
            )
        m = int(x.shape[0])
        _validate_routing_tensor(
            "topk_ids",
            topk_ids,
            shape=(m, self.topk),
            device=self.device,
            dtypes=(torch.int32,),
        )
        _validate_routing_tensor(
            "topk_weights",
            topk_weights,
            shape=(m, self.topk),
            device=self.device,
            dtypes=(torch.float32,),
        )
        if out is None:
            raise RuntimeError(
                "a P8 call writes a caller-owned output; pass out= (the arena "
                "holds scratch only)"
            )
        if (
            out.dtype != torch.bfloat16
            or tuple(out.shape) != (m, self.hidden)
            or out.device != self.device
            or not out.is_contiguous()
            or out.data_ptr() % 16
        ):
            raise RuntimeError(
                f"P8 native output contract mismatch: {out.dtype} {tuple(out.shape)}"
            )
        if self.full_coupled and not self.deterministic_output:
            # The kernel's grouped output is FP32 and only the coupled
            # deterministic reducer casts it to the BF16 result.
            raise RuntimeError(
                "a full-coupled P8 runtime without its deterministic reducer "
                "has no BF16 output stage"
            )
        if self.prefill_chunk_tokens and m > self.prefill_chunk_tokens:
            if not self.full_coupled or self.debug_capture:
                raise RuntimeError(
                    "bounded prefill requires full coupling without debug capture"
                )
            # MoE is token-local. Bound route-output and materialized carrier
            # storage without changing the scheduler batch or attention work.
            # The chunks reuse the caller's arena on the same stream; no host
            # readback or sync.
            for start in range(0, m, self.prefill_chunk_tokens):
                stop = min(start + self.prefill_chunk_tokens, m)
                self(
                    x[start:stop],
                    topk_weights[start:stop],
                    topk_ids[start:stop],
                    out[start:stop],
                    scratch=scratch,
                    config=config,
                )
            return out
        # Every stored rate now has a fused grouped owner, the M32-or-M64/N128
        # arm the tile policy or the raced configuration selects, so a non-K4
        # layer at M>1 runs the same native path as K4 rather than looping the
        # M1 kernel row by row. The row-by-row fallback is deliberately gone: a
        # rate without a grouped specialization must fail closed instead of
        # silently serving at a fraction of the speed.
        if self.scale_component is not None and not self.full_coupled and m != 1:
            raise RuntimeError("P8 scale sandwich currently supports M=1 only")
        spec = self.launch_spec(config)
        materialized, small_m = _select_arm(spec, m)
        arm = self._compile(spec, materialized, small_m=small_m, expected_m=m)
        tile_m = arm.tile_m
        flat_ids = topk_ids.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        # The same declaration the plan sized the capacity with: the arena the
        # caller bound and the regions this call carves cannot drift apart.
        layout = p8_scratch_layout_for(spec, m)
        arena = self._workspace(scratch, layout)
        # One GPU fill initializes the regions and their alignment padding; the
        # views add no casts, copies, or device kernels. Two spans are skipped,
        # and only because every element of them is written before anything
        # reads it:
        #
        # * The direct RP2 prologue overwrites every high/residual input
        #   payload and scale that a valid route consumes, so only the state
        #   from the intermediate carrier onward is cleared there.
        # * The per-route partial sum (``route_output``, m * top-k * hidden
        #   floats) is written in full by the split FC2 owners before the
        #   reducer reads it: every valid route owns one physical row, and the
        #   scale-sandwich epilogue stores ``hidden`` FP32 columns of it with
        #   plain (non-accumulating) stores, four per lane over all 128-column
        #   output tiles. The monolithic arm never writes it without first
        #   clearing it in its own phase 0, which is the same rule the shared
        #   kernel states where it skips that clear.
        #
        # No other output, intermediate, queue, or routing clear is skipped.
        # The region is declared last, so the clear is the prefix before it
        # plus any alignment padding after it (none at the serving geometry);
        # that padding lies outside every region and no kernel addresses it.
        clear_start = (
            layout.region(INTERMEDIATE).offset
            if (
                self.rp2_skip_input_clear
                and small_m
                and self.full_coupled
                and self.p8_input_rowpack
                and self.fc1_broadcast_a
            )
            else 0
        )
        for offset, nbytes in p8_clear_spans(
            layout,
            start=clear_start,
            skip=(ROUTE_OUTPUT,) if self.deterministic_output else (),
        ):
            arena.narrow(0, offset, nbytes).zero_()
        buffers = {name: layout.view(arena, name) for name in layout.names()}
        packed_a = buffers["packed_a"]
        scale_flat = buffers["scale_flat"]
        intermediate_u32 = buffers[INTERMEDIATE]
        barrier_count = buffers["barrier_count"]
        barrier_epoch = buffers["barrier_epoch"]
        pair_head = buffers["pair_head"]
        producers_done = buffers["producers_done"]
        all_published = buffers["all_published"]
        task_head = buffers["task_head"]
        task_tail = buffers["task_tail"]
        task_ready = buffers["task_ready"]
        task_expert = buffers["task_expert"]
        task_m_tile = buffers["task_m_tile"]
        task_slice_begin = buffers["task_slice_begin"]
        task_slice_count = buffers["task_slice_count"]
        task_valid_rows = buffers["task_valid_rows"]
        tile_write_count = buffers["tile_write_count"]
        row_counts = buffers["row_counts"]
        expert_write_rows = buffers["expert_write_rows"]
        expert_tile_base = buffers["expert_tile_base"]
        token_map = buffers["token_map"]
        token_weights = buffers["token_weights"]
        # The reducer writes the caller's storage directly.
        output = out
        kernel_output = buffers[ROUTE_OUTPUT] if self.deterministic_output else output
        launch_mac = resolve_grid_cap(
            m,
            arm=arm.mac,
            config=0 if config is None else config.mac,
            policy=self.grid_policy and self.world_size == 4,
            pin=self.mac_override,
        )
        arm.compiled(
            _gptr(cutlass.BFloat16, x),
            _gptr(cutlass.Int32, flat_ids, 4),
            _gptr(cutlass.Float32, flat_weights, 4),
            _gptr(cutlass.Float4E2M1FN, packed_a),
            _gptr(cutlass.Float8E4M3FN, scale_flat),
            _gptr(cutlass.Uint8, packed_a),
            _gptr(cutlass.Uint8, scale_flat),
            _gptr(cutlass.Uint32, intermediate_u32),
            barrier_count,
            barrier_epoch,
            pair_head,
            producers_done,
            all_published,
            task_head,
            task_tail,
            _gptr(cutlass.Int32, task_ready, 4),
            _gptr(cutlass.Int32, task_expert, 4),
            _gptr(cutlass.Int32, task_m_tile, 4),
            _gptr(cutlass.Int32, task_slice_begin, 4),
            _gptr(cutlass.Int32, task_slice_count, 4),
            _gptr(cutlass.Int32, task_valid_rows, 4),
            _gptr(cutlass.Int32, tile_write_count, 4),
            self.w13_dummy,
            _gptr(cutlass.Float8E4M3FN, self.sentinel),
            self.w2_dummy,
            _gptr(cutlass.Float8E4M3FN, self.sentinel),
            _gptr(cutlass.Uint8, self.w13_scale_mx),
            _gptr(cutlass.Uint8, self.w2_scale_mx),
            _gptr(cutlass.Uint8, self.sentinel),
            _gptr(cutlass.Uint8, self.sentinel),
            _gptr(cutlass.Uint32, self.w13_stream),
            _gptr(cutlass.Uint32, self.w13_sfb),
            _gptr(cutlass.Uint32, self.w2_stream),
            _gptr(cutlass.Uint32, self.w2_sfb),
            row_counts,
            expert_write_rows,
            expert_tile_base,
            self.ones,
            self.ones,
            self.ones,
            self.ones,
            _gptr(
                cutlass.Float32 if self.full_coupled else cutlass.BFloat16,
                kernel_output,
            ),
            _gptr(cutlass.Int32, token_map, 4),
            _gptr(cutlass.Float32, token_weights, 4),
            m,
            m * self.topk,
            m * self.topk if self.deterministic_output else m,
            # Padded rows, task capacity and physical tiles: the arena's
            # extents, so a launch can never address outside its own regions.
            layout.region("token_map").shape[0],
            layout.region("task_ready").shape[0],
            layout.region("tile_write_count").shape[0],
            launch_mac,
            current_cuda_stream(),
            _gptr(cutlass.Uint8, self.zero_lut),
            _gptr(cutlass.Float16, self.scale_component_packed),
        )
        if self.deterministic_output:
            if self.full_coupled:
                reducer = self._compile_full_coupled_reducer()
                reducer(
                    _gptr(cutlass.Float32, kernel_output),
                    _gptr(cutlass.Float32, flat_weights, 4),
                    _gptr(cutlass.BFloat16, output),
                    m,
                    current_cuda_stream(),
                )
            else:
                _launch_dynamic_topk_sum(
                    route_output=kernel_output,
                    output=output,
                    m=m,
                    num_topk=self.topk,
                    k=self.hidden,
                    stream=current_cuda_stream(),
                )
        if self.debug_capture:
            self.debug_tensors = {
                "packed_a": packed_a,
                "scale_flat": scale_flat,
                "intermediate_u32": intermediate_u32,
                "route_output": kernel_output,
                "token_map": token_map,
                "row_counts": row_counts,
                "expert_tile_base": expert_tile_base,
            }
            self.debug_dispatch = {
                "small_m": small_m,
                "materialized": materialized,
                "fc1_tile_n": self.fc1_tile_n if small_m else 128,
                "tile_m": tile_m,
                "scratch_nbytes": layout.nbytes,
            }
        return output
