"""Native MXFP8 K256 staging preserves scale bytes and single-tile split tails."""

from dataclasses import replace

import pytest
import torch

from b12x import gemm
from b12x.preparation import PreparationSession, PreparedCall
from ..conftest import require_b12x
from ._mxfp8_rows_fixture import rows_and_bases


@pytest.mark.parametrize("rows", (1, 16))
@pytest.mark.parametrize("tile_m", (16, 32, 64))
@pytest.mark.parametrize("split", (1, 4))
def test_native_mxfp8_k256_scale_domain_capture(rows, tile_m, split):
    device = require_b12x()
    k, n = 1024, 136
    a, a_bases = rows_and_bases(rows, k, device=device)
    b, b_bases = rows_and_bases(n, k, device=device)
    storage = tuple(
        torch.full((rows + 2, n, 1), 91, dtype=torch.bfloat16, device=device)
        for _ in range(2)
    )
    outputs = tuple(value[1:-1] for value in storage)
    workspace = torch.empty(4 * rows * n, dtype=torch.float32, device=device)
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
        allow_bf16_atomic_reduction=False,
    )
    config = gemm.DenseGemmConfig(
        backend="cutedsl",
        tile_m=tile_m,
        tile_n=64,
        tile_k=128,
        load_path="tma",
        swap_ab=False,
        split_k_slices=split,
        large_m_unroll=False,
        target_occupancy=None,
    )
    plans = (
        gemm.plan(query, override=config),
        gemm.plan(query, override=replace(config, tile_k=256)),
    )

    def request(plan, output, name):
        return plan.request(
            name=name,
            prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(
                    a.values.view(rows, k, 1),
                    a.scale_mma,
                    b.values.view(n, k, 1),
                    b.scale_mma,
                    None,
                    output,
                    workspace,
                    torch.bfloat16,
                    None,
                ),
                owners=(*a_bases, *b_bases, *storage, workspace),
            ),
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare(
            tuple(
                request(plan, output, str(index))
                for index, (plan, output) in enumerate(zip(plans, outputs, strict=True))
            )
        )

        def run():
            for plan, output in zip(plans, outputs, strict=True):
                gemm.mm(
                    (a.values.view(rows, k, 1), a.scale_mma),
                    (b.values.view(n, k, 1), b.scale_mma),
                    out=output,
                    workspace=workspace,
                    plan=plan,
                )

        a.values.fill_(1)
        b.values.fill_(1)
        a_bases[2].fill_(127)
        b_bases[2].fill_(127)
        run()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run()
        session.freeze()
        for left_scale, right_scale in ((64, 190), (190, 64), (254, 254), (127, 127)):
            zero = left_scale == right_scale == 254
            a.values.fill_(0 if zero else 1)
            b.values.fill_(0 if zero else 1)
            a_bases[2].fill_(left_scale)
            b_bases[2].fill_(right_scale)
            workspace.fill_(float("nan"))
            for output in outputs:
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
            for output, backing in zip(outputs, storage, strict=True):
                assert torch.isfinite(output).all()
                assert torch.all(output == (0 if zero else k))
                assert torch.all(backing[0] == 91) and torch.all(backing[-1] == 91)
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        graph.reset()
