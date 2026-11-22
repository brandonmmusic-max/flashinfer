"""Configuration contract for SM120 4D-TMA relative prefill attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from b12x.preparation import (
    FrozenMapping,
    Knob,
    ParameterBinding,
    ParameterSpace,
    TuningContract,
)
from b12x.preparation._efficiency import capture_autotune_effort

from ._kernel import SMEM_LIMIT

MAX_SPLITS = 32


@dataclass(frozen=True, kw_only=True)
class RelativePrefillQuery:
    """Geometry and capacity contract of one paged full or staged-window prefill layer family."""

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    rel_extent: int
    max_batch: int
    max_total_q: int
    sm_count: int
    window_left: int = -1
    autotune_effort: str = field(default_factory=capture_autotune_effort)

    @property
    def group_size(self) -> int:
        return self.num_q_heads // self.num_kv_heads


@dataclass(frozen=True, kw_only=True)
class RelativePrefillConfig:
    """Kernel knobs: CTA tile shape, head grouping, TMA pipeline, and split-KV budget."""

    q_tile_tokens: int
    grouped: bool
    stage_rows: int
    bn_sub: int
    num_stages: int
    split_kv: bool
    max_splits: int
    min_stages_per_split: int
    q_in_smem: bool = False

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "RelativePrefillConfig":
        expected = {f.name for f in fields(cls)}
        if set(payload) != expected:
            raise ValueError(
                f"relative_prefill config fields must be {sorted(expected)}, got {sorted(payload)}"
            )
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_query(query: RelativePrefillQuery) -> None:
    if not isinstance(query, RelativePrefillQuery):
        raise TypeError("relative_prefill query must be RelativePrefillQuery")
    if query.head_dim != 128:
        raise ValueError(f"head_dim must be 128, got {query.head_dim}")
    if (
        query.num_kv_heads <= 0
        or query.num_q_heads <= 0
        or query.num_q_heads % query.num_kv_heads
    ):
        raise ValueError("num_q_heads must be a positive multiple of num_kv_heads")
    if query.page_size <= 0 or query.page_size % 64 != 0:
        raise ValueError("page_size must be a positive multiple of 64")
    if query.rel_extent <= 0:
        raise ValueError("rel_extent must be positive")
    if query.window_left < -1:
        raise ValueError("window_left must be -1 (full) or non-negative")
    if query.max_batch <= 0 or query.max_total_q <= 0 or query.sm_count <= 0:
        raise ValueError("max_batch, max_total_q and sm_count must be positive")


def _default_config(query: RelativePrefillQuery) -> RelativePrefillConfig:
    group_size = query.group_size
    is_window = query.window_left >= 0
    if is_window:
        if group_size <= 4 and query.max_total_q >= 2048:
            q_tile_tokens = 64
            grouped = True
            q_in_smem = True
        elif group_size <= 4 and query.max_total_q >= 1024:
            q_tile_tokens = 32
            grouped = True
            q_in_smem = False
        elif group_size <= 4 and query.max_total_q >= 512:
            q_tile_tokens = 16
            grouped = True
            q_in_smem = False
        else:
            q_tile_tokens = 16
            grouped = False
            q_in_smem = False
        return RelativePrefillConfig(
            q_tile_tokens=q_tile_tokens,
            grouped=grouped,
            stage_rows=64,
            bn_sub=64,
            num_stages=3,
            split_kv=False,
            max_splits=1,
            min_stages_per_split=16,
            q_in_smem=q_in_smem,
        )
    if group_size <= 4 and query.max_total_q >= 64:
        q_tile_tokens = 64
        grouped = True
        bn_sub = 64
    elif group_size <= 6 and query.max_total_q >= 32:
        q_tile_tokens = 32
        grouped = True
        bn_sub = 64
    else:
        q_tile_tokens = 64 if query.max_total_q >= 64 else 16
        grouped = False
        bn_sub = 64
    stage_rows = 64
    num_stages = 3
    q_tiles = max(1, (query.max_total_q + q_tile_tokens - 1) // q_tile_tokens)
    head_ctas = query.num_kv_heads if grouped else query.num_q_heads
    single_req_ctas = q_tiles * head_ctas
    if single_req_ctas < query.sm_count:
        splits = min(MAX_SPLITS, max(1, query.sm_count // max(single_req_ctas, 1)))
    elif query.max_total_q >= 2048 and grouped and q_tile_tokens == 64:
        splits = 2
    else:
        splits = 1
    row_bytes = query.max_total_q * query.num_q_heads * query.head_dim * 4
    if row_bytes > 0:
        splits = min(splits, max(1, (256 << 20) // row_bytes))
    split_kv = splits > 1
    return RelativePrefillConfig(
        q_tile_tokens=q_tile_tokens,
        grouped=grouped,
        stage_rows=stage_rows,
        bn_sub=bn_sub,
        num_stages=num_stages,
        split_kv=split_kv,
        max_splits=splits if split_kv else 1,
        min_stages_per_split=16 if query.max_total_q >= 2048 else 4,
        q_in_smem=False,
    )


def _validate_config(query: RelativePrefillQuery, config: RelativePrefillConfig) -> None:
    if config.q_tile_tokens < 16 or config.q_tile_tokens % 16 != 0:
        raise ValueError("q_tile_tokens must be a positive multiple of 16")
    if config.stage_rows not in (32, 64, 128) or query.page_size % config.stage_rows != 0:
        raise ValueError("stage_rows must be 32/64/128 and divide page_size")
    if config.bn_sub not in (16, 32, 64) or config.stage_rows % config.bn_sub != 0:
        raise ValueError("bn_sub must be 16/32/64 and divide stage_rows")
    if not 1 <= config.max_splits <= MAX_SPLITS:
        raise ValueError(f"max_splits must be in [1, {MAX_SPLITS}]")
    if not config.split_kv and config.max_splits != 1:
        raise ValueError("non-split relative_prefill requires max_splits == 1")
    warps_per_head = config.q_tile_tokens // 16
    ncw = (query.group_size * warps_per_head) if config.grouped else warps_per_head
    if ncw > 12:
        raise ValueError(f"compute warp count {ncw} exceeds 12")
    smem = 2 * config.num_stages * config.stage_rows * 128 + (ncw * 4096 if config.q_in_smem else 0)
    if smem + 2048 > SMEM_LIMIT:
        raise ValueError("config exceeds SM120 shared-memory limit")


def _parameters(query: RelativePrefillQuery, _device) -> ParameterSpace:
    cfg = _default_config(query)
    return ParameterSpace.create(
        TUNING.knobs,
        values={"num_stages": (cfg.num_stages,)},
        autotune_effort=query.autotune_effort,
    )


def _materialize(query: RelativePrefillQuery, _device, choice) -> RelativePrefillConfig:
    base = _default_config(query)
    return RelativePrefillConfig(**{**base.to_dict(), **dict(choice)})


TUNING = TuningContract(
    component_id="attention.relative_prefill",
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(f.name for f in fields(RelativePrefillQuery)),
    config_fields=frozenset(f.name for f in fields(RelativePrefillConfig)),
    encode_query=asdict,
    encode_config=asdict,
    decode_config=lambda payload: RelativePrefillConfig(**dict(payload)),
    validate_query=lambda query, _device=None: validate_query(query),
    validate_config=lambda query, config, _device=None: _validate_config(query, config),
    default_config=lambda query, _device=None: _default_config(query),
    knobs=(
        Knob(name="num_stages", values=None, binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=1,
    parameters=_parameters,
    materialize=_materialize,
)

__all__ = [
    "RelativePrefillConfig",
    "RelativePrefillQuery",
    "TUNING",
    "validate_query",
]
