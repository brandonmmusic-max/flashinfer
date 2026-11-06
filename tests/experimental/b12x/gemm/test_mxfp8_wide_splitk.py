"""Native MXFP8 split-K preserves live rows and captured caller-owned storage."""

from itertools import permutations

import pytest
import torch

from b12x import gemm
from b12x.preparation import PreparationSession, PreparedCall
from b12x.quantization import mxfp8
from ..conftest import require_b12x
from ._mxfp8_rows_fixture import rows_and_bases


@pytest.mark.parametrize("rows", (9, 13, 15, 16))
@pytest.mark.parametrize("atomic", (False, True))
@pytest.mark.parametrize(
    "tile_m,tile_k",
    ((16, 128), (32, 128), (64, 128), (128, 128), (16, 256), (32, 256), (64, 256)),
)
def test_wide_splitk_mutable_capture(
    rows,
    atomic,
    tile_m,
    tile_k,
    *,
    n=16384,
    tile_n=64,
    split_slices=4,
    large_m_unroll=True,
):
    device = require_b12x()
    k = 5120
    torch.manual_seed(816 + rows)
    source = torch.randint(-2, 3, (rows, k), device=device).bfloat16()
    a, bases = rows_and_bases(rows, k, device=device)
    weight = gemm.blockscaled.pack_weight(
        torch.randint(-2, 3, (n, k), device=device).to(torch.float8_e4m3fn),
        torch.randint(124, 128, (n, k // 32), device=device, dtype=torch.uint8),
    ).weight
    storage = torch.full((rows + 2, n, 1), 91, device=device, dtype=torch.bfloat16)
    output = storage[1:-1]
    workspace = torch.empty(split_slices * rows * n, device=device, dtype=torch.float32)
    quant = mxfp8.plan(
        mxfp8.query_from_call(source, a.values, a.scale_rows, a.scale_mma)
    )
    query = gemm.DenseGemmQuery(
        recipe="mxfp8",
        entry_point="gemm.mm",
        weight_storage="native",
        output_dtype="bfloat16",
        batch=1,
        max_rows=rows,
        in_features=k,
        out_features=n,
        output_mode="provided",
        alpha_mode="unit",
        workspace_form="provided",
        workspace_nbytes=workspace.numel() * 4,
        allow_bf16_atomic_reduction=atomic,
    )
    config = gemm.DenseGemmConfig(
        backend="cutedsl",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        load_path="tma",
        swap_ab=False,
        split_k_slices=split_slices,
        large_m_unroll=large_m_unroll,
        target_occupancy=None,
    )
    plan = gemm.plan(query, override=config)

    def run():
        mxfp8.quantize_rows(source, a.values, a.scale_rows, a.scale_mma, plan=quant)
        gemm.mm(
            (a.values.view(rows, k, 1), a.scale_mma),
            (weight.values.view(n, k, 1), weight.scale_mma),
            out=output,
            workspace=workspace,
            plan=plan,
        )

    def check():
        left = a.values.float() * torch.exp2(
            a.scale_rows.view(torch.uint8)[0].float() - 127
        ).repeat_interleave(32, dim=1)
        right = weight.values.float() * torch.exp2(
            weight.scale_rows.view(torch.uint8)[0].float() - 127
        ).repeat_interleave(32, dim=1)
        actual = output[:, :, 0]
        if atomic:
            partials = [
                (
                    left[:, start : start + k // split_slices]
                    @ right[:, start : start + k // split_slices].T
                ).bfloat16()
                for start in range(0, k, k // split_slices)
            ]
            distance = torch.full_like(actual.float(), float("inf"))
            reference = torch.zeros_like(actual)
            for order in permutations(range(split_slices)):
                candidate = torch.zeros_like(actual)
                for index in order:
                    candidate = (candidate.float() + partials[index].float()).bfloat16()
                error = (actual.float() - candidate.float()).abs()
                reference = torch.where(error < distance, candidate, reference)
                distance = torch.minimum(distance, error)
        else:
            reference = (left @ right.T).bfloat16()
        torch.testing.assert_close(actual, reference, rtol=0.01, atol=0.002)
        assert torch.isfinite(actual).all()
        assert torch.all(storage[0] == 91) and torch.all(storage[-1] == 91)

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare(
            (
                quant.request(
                    name="quant",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(
                            source, a.values, a.scale_rows, a.scale_mma
                        ),
                        owners=(source, *bases),
                    ),
                ),
                plan.request(
                    name="split",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(
                            a.values.view(rows, k, 1),
                            a.scale_mma,
                            weight.values.view(n, k, 1),
                            weight.scale_mma,
                            None,
                            output,
                            workspace,
                            torch.bfloat16,
                            None,
                        ),
                        owners=(*bases, weight, storage, workspace),
                    ),
                ),
            )
        )
        run()
        check()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run()
        session.freeze()
        for live in (rows, 3, 0, rows - 1):
            source.random_(-2, 3)
            source[live:].zero_()
            weight.values[:, 0].fill_(1 if live else 0)
            a.values.view(torch.uint8).fill_(255)
            a.scale_mma.view(torch.uint8).fill_(255)
            workspace.fill_(float("nan"))
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in (
                "allocation.all.allocated",
                "allocation.all.freed",
                "segment.all.allocated",
                "segment.all.freed",
            ):
                assert before[key] == after[key]
            check()
            assert torch.count_nonzero(output[live:]) == 0
        graph.reset()


@pytest.mark.parametrize("rows", (1, 16))
@pytest.mark.parametrize("split_slices", (2, 4))
@pytest.mark.parametrize("atomic", (False, True))
@pytest.mark.parametrize("large_m_unroll", (False, True))
def test_splitk_nondivisible_raster_capture(rows, split_slices, atomic, large_m_unroll):
    test_wide_splitk_mutable_capture(
        rows,
        atomic,
        128,
        64,
        n=16480,
        tile_n=128,
        split_slices=split_slices,
        large_m_unroll=large_m_unroll,
    )
