"""Startup selection for unquantized BF16 and FP32 projections."""

from dataclasses import asdict, dataclass, field, fields

from b12x.preparation import Knob, ParameterBinding, ParameterSpace, TuningContract
from b12x.preparation._efficiency import (
    AutotuneEffort,
    capture_autotune_effort,
)


@dataclass(frozen=True, kw_only=True)
class GemvQuery:
    source_dtype: str
    weight_dtype: str
    max_rows: int
    in_features: int
    out_features: int
    source_contiguous: bool
    source_aligned: bool
    weight_contiguous: bool
    weight_aligned: bool
    output_dtype: str = "bfloat16"
    output_contiguous: bool = True
    output_aligned: bool = True
    bias_dtype: str | None = None
    autotune_effort: str = field(default_factory=capture_autotune_effort)

    def __post_init__(self):
        object.__setattr__(
            self, "autotune_effort", AutotuneEffort(self.autotune_effort).value
        )


@dataclass(frozen=True, kw_only=True)
class GemvConfig:
    backend: str
    rows_per_tile: int = 8
    block_threads: int = 128
    split_k: int = 1


def validate_query(query):
    if not isinstance(query, GemvQuery):
        raise TypeError("projection query must be GemvQuery")
    if any(
        dtype not in ("bfloat16", "float32")
        for dtype in (
            query.source_dtype,
            query.weight_dtype,
            query.output_dtype,
        )
    ) or query.bias_dtype not in (None, "bfloat16", "float32"):
        raise ValueError("unquantized projection requires BF16 or FP32 operands")
    if (
        not 0 < query.max_rows < 2**31
        or min(query.in_features, query.out_features) <= 0
    ):
        raise ValueError(
            "projection geometry must be positive and row capacity fit Int32"
        )
    for name in (
        "source_contiguous",
        "source_aligned",
        "weight_contiguous",
        "weight_aligned",
        "output_contiguous",
        "output_aligned",
    ):
        if type(getattr(query, name)) is not bool:
            raise TypeError(f"projection {name} must be a boolean")


def _mma_eligible(query):
    return query.source_dtype == query.weight_dtype == "bfloat16"


def _mma_preferred(query):
    return query.out_features >= 256 and query.in_features >= 16


def _tc_eligible(query):
    from ._tensor_core import (
        TC_K_MULTIPLE,
        TC_MAX_ROWS,
        TC_SMALL_K,
        TC_SMALL_K_MAX_ROWS,
        TC_SMALL_K_N_MULTIPLE,
    )

    if not (
        query.source_dtype == query.weight_dtype == "bfloat16"
        and query.source_contiguous
        and query.source_aligned
        and query.weight_contiguous
        and query.weight_aligned
        and query.output_contiguous
    ):
        return False
    if (
        query.in_features == TC_SMALL_K
        and query.out_features % TC_SMALL_K_N_MULTIPLE == 0
        and query.max_rows <= TC_SMALL_K_MAX_ROWS
        and query.output_dtype == "bfloat16"
        and query.bias_dtype is None
    ):
        return True
    return (
        query.output_dtype in ("bfloat16", "float32")
        and query.bias_dtype in (None, "bfloat16", "float32")
        and query.max_rows <= TC_MAX_ROWS
        and query.in_features % TC_K_MULTIPLE == 0
    )


def _torch_eligible(query):
    if query.source_dtype != "bfloat16" or query.weight_dtype != "bfloat16":
        return False
    if query.output_dtype == "bfloat16":
        return query.bias_dtype in (None, "bfloat16")
    import torch

    return (
        query.output_dtype == "float32"
        and query.bias_dtype is None
        and hasattr(torch.ops.aten.mm, "dtype_out")
    )


def _prefill_eligible(query):
    return (
        query.bias_dtype is None
        and query.source_dtype == query.weight_dtype == "bfloat16"
        and query.in_features % 64 == 0
        and query.source_contiguous
        and query.source_aligned
        and query.weight_contiguous
        and query.weight_aligned
        and query.output_contiguous
        and query.output_aligned
    )


def default_config(query, device):
    if _torch_eligible(query) and query.max_rows > 8:
        return GemvConfig(backend="torch")
    n_tiles = (query.out_features + 63) // 64
    minimum_mma_rows = 24 if n_tiles >= 64 else ((64 + n_tiles - 1) // n_tiles) * 32
    if (
        _mma_eligible(query)
        and _mma_preferred(query)
        and query.max_rows >= minimum_mma_rows
    ):
        return GemvConfig(backend="mma")
    rows_per_tile = 8
    if (
        query.source_dtype == query.weight_dtype == "bfloat16"
        and query.in_features == 5120
        and query.out_features in (32, 384, 512)
        and query.max_rows <= 8
        and query.source_contiguous
        and query.source_aligned
        and query.weight_contiguous
        and query.weight_aligned
    ):
        rows_per_tile = 2 if query.out_features == 32 else 4
    return GemvConfig(backend="simt", rows_per_tile=rows_per_tile)


def _tc_split_k_candidates(query):
    from ._tensor_core import TC_SMALL_K, TC_SPLIT_K_VALUES, tc_chunk_k

    if query.in_features == TC_SMALL_K:
        return (1,)
    n_blocks = (query.out_features + 15) // 16
    chunks = query.in_features // tc_chunk_k(query.in_features)
    cands = []
    if n_blocks >= 32:
        cands.append(1)
    if n_blocks < 256:
        for sk in TC_SPLIT_K_VALUES:
            if sk == 1:
                continue
            if chunks % sk == 0 and 32 <= n_blocks * sk <= 1024:
                cands.append(sk)
    if not cands:
        cands.append(1)
    return tuple(cands)


def _validate_query(query, device):
    validate_query(query)


def validate_config(query, config, device):
    from ._tensor_core import TC_SMALL_K, TC_SPLIT_K_VALUES, tc_chunk_k

    if not isinstance(config, GemvConfig):
        raise TypeError("projection config must be GemvConfig")
    if config.backend in ("simt", "mixed_vector"):
        if config.backend == "mixed_vector" and not (
            query.source_dtype == "bfloat16" and query.weight_dtype == "float32"
        ):
            raise ValueError(
                "mixed vector projection requires BF16 source and FP32 weights"
            )
        if type(config.rows_per_tile) is not int or config.rows_per_tile not in (
            1,
            2,
            4,
            8,
        ):
            raise ValueError("SIMT projection rows_per_tile must be 1, 2, 4 or 8")
        if type(config.block_threads) is not int or config.block_threads not in (
            128,
            256,
        ):
            raise ValueError("SIMT projection block_threads must be 128 or 256")
        if config.split_k != 1:
            raise ValueError("split_k only configures tensor-core projection")
        return
    if config.rows_per_tile != 8:
        raise ValueError("rows_per_tile only configures SIMT projection")
    if config.block_threads != 128:
        raise ValueError("block_threads only configures SIMT projection")
    if config.backend == "tc":
        if not _tc_eligible(query):
            raise ValueError("projection backend 'tc' is ineligible for this query")
        if query.in_features == TC_SMALL_K:
            if config.split_k != 1:
                raise ValueError("invalid split_k for small-K16 tensor-core projection")
            return
        chunks = query.in_features // tc_chunk_k(query.in_features)
        if (
            type(config.split_k) is not int
            or config.split_k not in TC_SPLIT_K_VALUES
            or chunks % config.split_k != 0
        ):
            raise ValueError("invalid split_k for tensor-core projection")
        return
    if config.split_k != 1:
        raise ValueError("split_k only configures tensor-core projection")
    if config.backend == "mma" and _mma_eligible(query):
        return
    if config.backend == "torch" and _torch_eligible(query):
        return
    if config.backend == "prefill" and _prefill_eligible(query):
        return
    raise ValueError(
        f"projection backend {config.backend!r} is ineligible for this query"
    )


def _parameters(query, device):
    backends = ["simt"]
    if query.source_dtype == "bfloat16" and query.weight_dtype == "float32":
        backends.append("mixed_vector")
    if _mma_eligible(query):
        backends.append("mma")
    tc_splits = (1,)
    if _tc_eligible(query):
        backends.append("tc")
        tc_splits = _tc_split_k_candidates(query)
    if _prefill_eligible(query):
        backends.append("prefill")
    if _torch_eligible(query):
        backends.append("torch")
    split_values = tuple(dict.fromkeys((1, *tc_splits)))

    def eligible(choice):
        if choice["backend"] in ("simt", "mixed_vector"):
            return choice["split_k"] == 1
        if choice["backend"] == "tc":
            return (
                choice["rows_per_tile"] == 8
                and choice["block_threads"] == 128
                and choice["split_k"] in tc_splits
            )
        return (
            choice["rows_per_tile"] == 8
            and choice["block_threads"] == 128
            and choice["split_k"] == 1
        )

    return ParameterSpace.create(
        TUNING.knobs,
        values={"backend": tuple(backends), "split_k": split_values},
        predicates=(eligible,),
        # The prefill GEMM is raced only by the audit; it never came within 2%
        # of a winner on the shapes it was preferred for.  Likewise, for
        # small-M decode shapes (M <= 16, N <= 4096) where single-launch
        # TC-GEMV is eligible, hot-L2 benchmarking over-favors cuBLASLt
        # two-kernel split-K which is slower in cold DRAM across 93 layers.
        # At M >= 17 cuBLASLt TMA beats TC-GEMV even in cold DRAM.
        efficiency_predicates=(
            lambda choice: (
                (choice["backend"] != "mma" or _mma_preferred(query))
                and choice["backend"] != "prefill"
                and not (
                    choice["backend"] == "torch"
                    and _tc_eligible(query)
                    and query.max_rows <= 16
                    and query.out_features <= 4096
                )
            ),
        ),
        autotune_effort=query.autotune_effort,
    )


TUNING = TuningContract(
    component_id="gemm.bf16_gemv",
    query_schema_version=23,
    config_schema_version=24,
    query_fields=frozenset(field.name for field in fields(GemvQuery)),
    config_fields=frozenset(field.name for field in fields(GemvConfig)),
    encode_query=asdict,
    encode_config=asdict,
    decode_config=lambda payload: GemvConfig(**dict(payload)),
    validate_query=_validate_query,
    validate_config=validate_config,
    default_config=default_config,
    knobs=(
        Knob(name="backend", values=None, binding=ParameterBinding.COMPILE),
        Knob(
            name="rows_per_tile", values=(1, 2, 4, 8), binding=ParameterBinding.COMPILE
        ),
        Knob(name="block_threads", values=(128, 256), binding=ParameterBinding.COMPILE),
        Knob(
            name="split_k",
            values=(1, 2, 3, 4, 6, 7, 8, 12, 14, 16, 28, 56),
            binding=ParameterBinding.COMPILE,
        ),
    ),
    candidate_contract_version=17,
    parameters=_parameters,
    materialize=lambda query, device, choice: GemvConfig(**dict(choice)),
)
