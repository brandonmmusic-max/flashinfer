"""MSA extend must not read past the request's page-table row.

The page-table load is guarded per row when a 128-token union block spans more
than one page-table entry (page 64).  At page 128 ``entries_per_block`` is 1, so
that guard compiles out, while a per-token selection list is the caller's raw
block ids and can contain an id past the visible extent.  The guard and the K
valid-rows trim therefore also apply to per-token tiles.

The same file covers the MSA/TMA exclusion in
``build_extend_forward_kernel``: the extend TMA staging helpers derive the page
index from the packed token base and never consult a block id, so an MSA block
list must not reach that path.
"""

from __future__ import annotations

import math

import pytest
import torch

from b12x.attention.paged._forward import paged_attention_forward
from b12x.attention.paged._scratch import (
    B12XPagedAttentionScratchCaps,
    plan_paged_attention_scratch,
)
from b12x.attention.paged.forward_extend_generic import build_extend_forward_kernel
from b12x.attention.paged.planner import create_paged_plan
from b12x.attention.paged.traits import select_paged_forward_traits_from_plan
from b12x.testing.reference.helpers import require_b12x
from b12x.testing.reference.paged_attention_helpers import (
    make_msa_q2k_indices,
    make_paged_inputs,
    quantize_paged_kv_cache_e4m3,
)


def _lse_base2_to_natural(lse: torch.Tensor) -> torch.Tensor:
    return lse * math.log(2.0)


def _run_msa_extend(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    q2k_indices: torch.Tensor,
    *,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    msa_union_tile: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="extend",
        msa_block_sparse=True,
        msa_union_tile=msa_union_tile,
    )
    assert plan.split_kv is False
    assert plan.cta_tile_q in (16, 128)
    assert plan.new_batch_size > 0
    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="extend",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=k_cache.shape[1],
            max_total_q=plan.total_q,
            max_batch=page_table.shape[0],
            max_page_table_width=page_table.shape[1],
            max_work_items=max(plan.new_batch_size, 1),
            max_partial_rows=0,
            num_cache_pages=k_cache.shape[0],
            msa_block_sparse=True,
            msa_union_tile=msa_union_tile,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    output = torch.empty(
        (q.shape[0], q.shape[1], v_cache.shape[3]), dtype=q.dtype, device=q.device
    )
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    out, lse_base2 = paged_attention_forward(binding=binding)
    return out, _lse_base2_to_natural(lse_base2)


def _msa_per_token_out_of_extent_case(page_size: int) -> None:
    """Per-token tiles must skip a caller block id that starts past the extent.

    The kernel walks ``ceil(visible / 128)`` blocks, so an id in one of those
    slots is read even when the list has no padding.  Replacing the first slot
    with an id far past the request's pages must be as inert as ``-1``; without
    the guard the page-table column leaves the row (and the tensor).
    """
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[129],
        cache_seqlens=[257],
        page_size=page_size,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1327,
    )
    q2k = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        topk=16,
        seed=59,
        force_block0=True,
    )
    q2k_skip = q2k.clone()
    q2k_skip[:, 0, 0] = -1
    q2k_bad = q2k.clone()
    q2k_bad[:, 0, 0] = 1_000_000
    out_skip, lse_skip = _run_msa_extend(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_skip,
        msa_union_tile=False,
    )
    out_bad, lse_bad = _run_msa_extend(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_bad,
        msa_union_tile=False,
    )
    assert torch.isfinite(out_bad).all()
    assert torch.equal(out_bad, out_skip)
    assert torch.equal(lse_bad, lse_skip)


@pytest.mark.parametrize("page_size", [64, 128])
def test_msa_extend_per_token_out_of_extent_block(page_size: int) -> None:
    """A caller block id past the extent must not leave the page-table row."""
    require_b12x()
    _msa_per_token_out_of_extent_case(page_size)


def test_msa_extend_head256_page64_is_unreachable() -> None:
    """The extend TMA staging path cannot meet an MSA plan.

    ``base_use_paged_kv_tma_extend`` requires head_dim 256/256 and
    ``cta_tile_q`` 16, while ``create_paged_plan`` rejects MSA unless both
    head dims are 128.  The helpers that issue the TMA loads derive the page
    index from the packed token base and never consult a block id, so a rotated
    (non-contiguous) MSA block list would read the wrong page.  The kernel
    rejects the combination instead of carrying an untested term.
    """
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[2],
        cache_seqlens=[200],
        page_size=64,
        q_heads=64,
        kv_heads=4,
        head_dim=256,
        dtype=torch.bfloat16,
        seed=1381,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        topk=16,
        seed=61,
        force_block0=True,
        poison_padding=True,
    )
    # 200 tokens cover blocks 0 and 1.  Rotate the list so it is neither
    # contiguous nor ordered from block 0; the helpers that issue the TMA
    # loads would read the wrong page for that list.
    q2k_indices = q2k_indices.clone()
    q2k_indices[:, :, 0] = 1
    q2k_indices[:, :, 1] = 0
    k_fp8, v_fp8, _k_descale, _v_descale = quantize_paged_kv_cache_e4m3(
        k_cache, v_cache, page_table, cache_seqlens
    )
    with pytest.raises(ValueError, match="head_dim_qk=head_dim_vo=128"):
        create_paged_plan(
            q,
            k_fp8,
            v_fp8,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            mode="extend",
            msa_block_sparse=True,
        )

    # The same head-256 geometry does take the TMA path, and the MSA guard
    # would compile, so the raise is the only check between a caller that
    # bypasses the planner and the helpers that issue the TMA loads without a
    # block id.
    plan = create_paged_plan(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, mode="extend"
    )
    traits = select_paged_forward_traits_from_plan(plan)
    kernel = build_extend_forward_kernel(
        traits, False, False, window_left=-1, page_size=64
    )
    assert kernel.use_paged_kv_tma
    with pytest.raises(ValueError, match="MSA block-sparse extend"):
        build_extend_forward_kernel(
            traits,
            False,
            False,
            window_left=-1,
            msa_block_sparse=True,
            msa_union_tile=True,
            page_size=64,
        )

    # Every reachable MSA extend plan (head_dim 128) takes the guarded
    # direct-copy path instead.
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[2],
        cache_seqlens=[200],
        page_size=64,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1387,
    )
    k_fp8, v_fp8, _k_descale, _v_descale = quantize_paged_kv_cache_e4m3(
        k_cache, v_cache, page_table, cache_seqlens
    )
    plan = create_paged_plan(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="extend",
        msa_block_sparse=True,
    )
    traits = select_paged_forward_traits_from_plan(plan)
    kernel = build_extend_forward_kernel(
        traits,
        False,
        False,
        window_left=-1,
        msa_block_sparse=True,
        msa_union_tile=plan.msa_union_tile,
        page_size=64,
    )
    assert not kernel.use_paged_kv_tma
