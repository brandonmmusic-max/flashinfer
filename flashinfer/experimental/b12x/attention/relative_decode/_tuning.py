"""Configuration contract for SM120 TMA relative decode/verify attention."""

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

from ._kernel import SMEM_LIMIT, default_max_splits

MAX_Q_PER_REQ = 16
MAX_SPLITS = 64


@dataclass(frozen=True, kw_only=True)
class RelativeDecodeQuery:
    """Geometry and cache-mode contract of one relative attention layer family."""

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    is_window: bool
    commit_ring: bool
    page_or_window: int
    window_left: int
    rel_extent: int
    max_batch: int
    max_q_per_req: int
    max_total_q: int
    sm_count: int
    autotune_effort: str = field(default_factory=capture_autotune_effort)

    @property
    def group_size(self) -> int:
        return self.num_q_heads // self.num_kv_heads


@dataclass(frozen=True, kw_only=True)
class RelativeDecodeConfig:
    """Kernel knobs: split-KV budget and CTA organization."""

    split_kv: bool
    max_splits: int
    stage_rows: int
    num_stages: int
    num_consumer_warps: int
    min_tiles_per_split: int

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "RelativeDecodeConfig":
        expected = {f.name for f in fields(cls)}
        if set(payload) != expected:
            raise ValueError(
                f"relative_decode config fields must be {sorted(expected)}, got {sorted(payload)}"
            )
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_query(query: RelativeDecodeQuery) -> None:
    if not isinstance(query, RelativeDecodeQuery):
        raise TypeError("relative_decode query must be RelativeDecodeQuery")
    if query.head_dim != 128:
        raise ValueError(f"head_dim must be 128, got {query.head_dim}")
    if (
        query.num_kv_heads <= 0
        or query.num_q_heads <= 0
        or query.num_q_heads % query.num_kv_heads
    ):
        raise ValueError("num_q_heads must be a positive multiple of num_kv_heads")
    if not 1 <= query.max_q_per_req <= MAX_Q_PER_REQ:
        raise ValueError(f"max_q_per_req must be in [1, {MAX_Q_PER_REQ}]")
    if query.max_batch <= 0 or query.max_total_q < query.max_batch or query.sm_count <= 0:
        raise ValueError("max_batch, max_total_q and sm_count must be positive")
    if query.page_or_window <= 0 or query.page_or_window % 16 != 0:
        raise ValueError("page_or_window must be a positive multiple of 16")
    if query.rel_extent <= 0:
        raise ValueError("rel_extent must be positive")
    if query.is_window:
        if query.window_left < 0 or query.window_left >= query.page_or_window:
            raise ValueError("window_left must be in [0, page_or_window) for windowed layers")
    elif query.window_left != -1:
        raise ValueError("window_left must be -1 for paged full-attention layers")
    row_groups = (query.group_size * query.max_q_per_req + 15) // 16
    if row_groups > 8:
        raise ValueError("query group * max_q_per_req exceeds CTA row capacity")


def _default_config(query: RelativeDecodeQuery) -> RelativeDecodeConfig:
    max_splits = default_max_splits(
        query.max_batch, query.num_kv_heads, num_sms=query.sm_count, cap=32
    )
    split_kv = max_splits > 1
    row_groups = max(1, (query.group_size * query.max_q_per_req + 15) // 16)
    if query.is_window:
        ncw = row_groups
        num_stages = 4
    else:
        ncw = max(2, row_groups) if row_groups <= 1 else max(4, row_groups)
        num_stages = 4 if ncw <= 2 else 6
    if ncw > 8:
        ncw = 8
    return RelativeDecodeConfig(
        split_kv=split_kv,
        max_splits=max_splits if split_kv else 1,
        stage_rows=16,
        num_stages=num_stages,
        num_consumer_warps=ncw,
        min_tiles_per_split=2,
    )


def _validate_config(query: RelativeDecodeQuery, config: RelativeDecodeConfig) -> None:
    if config.stage_rows not in (16, 32) or query.page_or_window % config.stage_rows != 0:
        raise ValueError("stage_rows must be 16 or 32 and divide page_or_window")
    if not 1 <= config.max_splits <= MAX_SPLITS:
        raise ValueError(f"max_splits must be in [1, {MAX_SPLITS}]")
    if not config.split_kv and config.max_splits != 1:
        raise ValueError("non-split relative_decode requires max_splits == 1")
    row_groups = (query.group_size * query.max_q_per_req + 15) // 16
    if config.num_consumer_warps < row_groups or config.num_consumer_warps > 8:
        raise ValueError("num_consumer_warps cannot cover query rows")
    kv_bytes = 2 * config.num_stages * config.stage_rows * 128
    comb_bytes = 0 if config.num_consumer_warps == 1 else (
        config.num_consumer_warps * (8 * 8 * 32 + 4 * 32) * 4
    )
    if max(kv_bytes, comb_bytes, 4096) + 2048 > SMEM_LIMIT:
        raise ValueError("config exceeds SM120 shared-memory limit")


def _parameters(query: RelativeDecodeQuery, _device) -> ParameterSpace:
    cfg = _default_config(query)
    return ParameterSpace.create(
        TUNING.knobs,
        values={"num_consumer_warps": (cfg.num_consumer_warps,)},
        autotune_effort=query.autotune_effort,
    )


def _materialize(query: RelativeDecodeQuery, _device, choice) -> RelativeDecodeConfig:
    base = _default_config(query)
    return RelativeDecodeConfig(**{**base.to_dict(), **dict(choice)})


TUNING = TuningContract(
    component_id="attention.relative_decode",
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(f.name for f in fields(RelativeDecodeQuery)),
    config_fields=frozenset(f.name for f in fields(RelativeDecodeConfig)),
    encode_query=asdict,
    encode_config=asdict,
    decode_config=lambda payload: RelativeDecodeConfig(**dict(payload)),
    validate_query=lambda query, _device=None: validate_query(query),
    validate_config=lambda query, config, _device=None: _validate_config(query, config),
    default_config=lambda query, _device=None: _default_config(query),
    knobs=(
        Knob(name="num_consumer_warps", values=None, binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=1,
    parameters=_parameters,
    materialize=_materialize,
)

__all__ = [
    "RelativeDecodeConfig",
    "RelativeDecodeQuery",
    "TUNING",
    "validate_query",
]
