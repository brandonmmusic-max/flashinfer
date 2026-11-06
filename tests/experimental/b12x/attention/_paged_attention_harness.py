"""Paged-attention harness for the unowned-byte tests.

Builds one paged plan over a K/V pool whose ownership is known.  The pool marks
every row a request owns at a key position in ``[window_start, cache_len)``;
every other row and page is unowned and is filled with a byte pattern the
kernel must never read (the poison pattern).  A test compares the request's
outputs bitwise against a run whose unowned bytes are zero.

Each case names its geometry explicitly; a test that needs a different
geometry builds its own case.
"""

from __future__ import annotations

import dataclasses

import torch

from . import _unowned_cache_bytes as ucb


@dataclasses.dataclass(frozen=True)
class PagedCase:
    name: str
    hq: int
    hkv: int
    dqk: int
    dvo: int
    page: int
    window: int
    fp8: bool
    batch: int
    q_lens: tuple[int, ...]
    lens: tuple[int, ...]
    mode: str
    plan_kwargs: tuple[tuple[str, object], ...] = ()
    turbo: bool = False
    reference_min_cos: float = 0.9999

    @property
    def kv_dtype(self) -> torch.dtype:
        return torch.float8_e4m3fn if self.fp8 else torch.bfloat16


class PagedHarness:
    """One plan from ``create_paged_plan`` over a refillable K/V pool."""

    def __init__(self, case: PagedCase):
        from b12x.attention.paged._forward import paged_attention_forward
        from b12x.attention.paged._scratch import (
            B12XPagedAttentionScratchCaps,
            plan_paged_attention_scratch,
        )
        from b12x.attention.paged.planner import create_paged_plan

        self.case = case
        self.pool = ucb.build_paged_pool(
            lens=case.lens,
            page_size=case.page,
            kv_heads=case.hkv,
            dim_k=case.dqk,
            dim_v=case.dvo,
            kv_dtype=case.kv_dtype,
            q_lens=case.q_lens,
            window=case.window,
        )
        gen = torch.Generator(device="cuda").manual_seed(11)
        total_q = sum(case.q_lens)
        self.q = torch.randn(
            total_q,
            case.hq,
            case.dqk,
            generator=gen,
            device="cuda",
            dtype=torch.float32,
        ).to(torch.bfloat16)
        self.cu = torch.zeros(case.batch + 1, dtype=torch.int32, device="cuda")
        self.cu[1:] = torch.tensor(
            case.q_lens, dtype=torch.int32, device="cuda"
        ).cumsum(0)
        self.k_buf = self.pool.k.clone()
        self.v_buf = self.pool.v.clone()
        self.table_buf = self.pool.page_table.clone()
        self.output = torch.empty(
            total_q, case.hq, case.dvo, dtype=torch.bfloat16, device="cuda"
        )
        self._forward = paged_attention_forward
        from contextlib import nullcontext

        from b12x.attention.paged._controls import paged_controls

        self._turbo = case.turbo

        def _controls():
            return (
                paged_controls({"B12X_TURBO_ATTN": "1"})
                if self._turbo
                else nullcontext()
            )

        self._ctx = _controls
        plan_kwargs = dict(case.plan_kwargs)
        with self._ctx():
            self.plan = create_paged_plan(
                self.q,
                self.k_buf,
                self.v_buf,
                self.table_buf,
                self.pool.cache_seqlens,
                self.cu,
                mode=case.mode,
                window_left=case.window,
                **plan_kwargs,
            )
            self.scratch_plan = plan_paged_attention_scratch(
                B12XPagedAttentionScratchCaps(
                    device=self.q.device,
                    mode=case.mode,
                    dtype=self.q.dtype,
                    kv_dtype=self.k_buf.dtype,
                    num_q_heads=case.hq,
                    num_kv_heads=case.hkv,
                    head_dim_qk=case.dqk,
                    head_dim_vo=case.dvo,
                    page_size=case.page,
                    max_total_q=self.plan.total_q,
                    max_batch=case.batch,
                    max_page_table_width=self.table_buf.shape[1],
                    max_work_items=max(self.plan.new_batch_size, 1),
                    max_partial_rows=self.plan.total_num_partial_rows,
                    num_cache_pages=self.k_buf.shape[0],
                    use_cuda_graph=bool(plan_kwargs.get("enable_cuda_graph", False)),
                )
            )
        self.scratch = tuple(
            torch.empty(shape, dtype=dtype, device=self.q.device)
            for shape, dtype in self.scratch_plan.shapes_and_dtypes()
        )
        self._bind_kwargs = {
            key: plan_kwargs[key]
            for key in ("disable_split_kv", "fixed_split_size")
            if key in plan_kwargs
        }

    def binding(self):
        return self.scratch_plan.bind(
            scratch=self.scratch,
            q=self.q,
            k_cache=self.k_buf,
            v_cache=self.v_buf,
            output=self.output,
            page_table=self.table_buf,
            cache_seqlens=self.pool.cache_seqlens,
            cu_seqlens_q=self.cu,
            window_left=self.case.window,
            active_total_q=self.q.shape[0],
            k_descale=self.pool.k_descale,
            v_descale=self.pool.v_descale,
            **self._bind_kwargs,
        )

    def fill(self, pattern: str, plane: str) -> None:
        k_plane = pattern if plane in ("k", "all") else "ZERO"
        v_plane = pattern if plane in ("v", "all") else "ZERO"
        self.k_buf.copy_(ucb.overlay(self.pool.k, self.pool.owned, k_plane))
        self.v_buf.copy_(ucb.overlay(self.pool.v, self.pool.owned, v_plane))
        self.table_buf.copy_(self.pool.page_table_for(plane, pattern))
        if plane in ("scratch", "all"):
            for buffer in self.scratch:
                if pattern == "ZERO":
                    buffer.zero_()
                else:
                    buffer.copy_(ucb.fill_like(buffer, pattern))
        else:
            for buffer in self.scratch:
                buffer.zero_()

    def run(self, pattern: str, plane: str):
        self.fill(pattern, plane)
        with self._ctx():
            out, lse = self._forward(binding=self.binding())
        torch.cuda.synchronize()
        return out.clone(), lse.clone()

    def run_with_v_poison(self, request: int, positions):
        """Zero baseline with NaN in selected owned V rows of one request."""
        self.fill("ZERO", "all")
        for position in positions:
            page = int(self.pool.page_table[request, position // self.case.page].item())
            row = position % self.case.page
            ucb.set_owned_nan(self.v_buf, page, row)
        with self._ctx():
            out, lse = self._forward(binding=self.binding())
        torch.cuda.synchronize()
        return out.clone(), lse.clone()


def clean_reference(
    harness: PagedHarness, batch: int, window: int, causal: bool = True
):
    """Reference output for the clean pool."""
    from b12x.attention.paged.reference import paged_attention_reference

    pool = harness.pool
    descales = None
    if pool.k_descale is not None:
        descales = [
            value.expand(batch).contiguous()
            for value in (pool.k_descale, pool.v_descale)
        ]
    return paged_attention_reference(
        harness.q.contiguous(),
        pool.k,
        pool.v,
        pool.page_table,
        pool.cache_seqlens,
        harness.cu,
        k_descale=None if descales is None else descales[0],
        v_descale=None if descales is None else descales[1],
        causal=causal,
        window_left=window,
    )[0]


def route(plan, case: PagedCase) -> str:
    """A short label for the kernel configuration the plan selected."""
    return (
        f"paged:{case.mode}:split={int(plan.split_kv)},"
        f"chunks={getattr(plan, 'max_chunks_per_request', '?')},"
        f"tile_q={getattr(plan, 'cta_tile_q', '?')},"
        f"tile_kv={getattr(plan, 'cta_tile_kv', '?')}"
    )


def check_poison_matrix(
    harness: PagedHarness,
    *,
    patterns=("FF", "7F", "RAND1"),
    planes=("k", "v", "all", "scratch"),
) -> list[str]:
    """Compare every poisoned plane and pattern against the zero baseline."""
    case = harness.case
    rows = ucb.active_rows(case.q_lens)
    base, base_lse = harness.run("ZERO", "all")
    base2, base_lse2 = harness.run("ZERO", "all")
    assert ucb.compare(base, base2, rows).equal, (
        f"{case.name}: two zero-baseline runs differ; the plan is not repeatable"
    )
    assert torch.equal(base_lse, base_lse2), f"{case.name}: clean LSE is not repeatable"
    failures: list[str] = []
    for pattern in patterns:
        for plane in planes:
            out, lse = harness.run(pattern, plane)
            cmp = ucb.compare(base, out, rows)
            lse_cmp = ucb.compare(base_lse, lse, rows)
            if not (cmp.equal and lse_cmp.equal):
                failures.append(
                    f"{pattern}/{plane}: {cmp.describe()}"
                    + ("" if lse_cmp.equal else f"; lse: {lse_cmp.describe()}")
                )
    return failures
