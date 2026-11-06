"""Native FP4 wide-K and narrow-M tiles preserve captured storage contracts."""

from dataclasses import asdict, replace

import pytest
import torch

from b12x._lib.intrinsics.reference import swizzle_block_scale
from b12x.gemm import blockscaled
from b12x.gemm.blockscaled import _a16
from b12x.gemm._tuning import TUNING as DENSE_TUNING
from b12x.gemm.blockscaled._tuning import BlockscaledConfig, quantized_query
from b12x.preparation.device import detect_device
from b12x.testing.reference.helpers import require_b12x
from .test_blockscaled_a16 import (
    assert_close,
    make_weight,
    make_workspace,
    prepared_execution,
)
from .test_nvfp4_precision_race import _quantized_source


@pytest.mark.parametrize(
    "m,n,k",
    [(1, 136, 1024), (8, 264, 2048), (16, 128, 1024), (16, 264, 2048), (16, 264, 3072)],
)
@pytest.mark.parametrize(
    "tile_m,tile_n,tile_k,swap_ab,occupancy",
    [
        (64, 64, 1024, False, 2),
        (64, 64, 1024, True, 2),
        (16, 64, 256, False, 2),
        (16, 64, 512, False, 2),
        (16, 64, 1024, False, 2),
        (16, 64, 1024, False, 1),
        (16, 32, 1024, False, 3),
        (16, 32, 1024, False, 1),
    ],
)
def test_native_nvfp4_tiles_graph_replay(
    m, n, k, tile_m, tile_n, tile_k, swap_ab, occupancy
):
    require_b12x()
    torch.manual_seed(8126)
    source = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.125
    output = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    weight, decoded, scale_storage = make_weight("nvfp4", n, k)
    activation_scale = torch.tensor([128.0], device="cuda")
    options = dict(
        activation_global_scale=activation_scale, activation_mode="quantized"
    )
    query = blockscaled.query_from_call(source, weight, out=output, **options)
    dense = DENSE_TUNING.default_config(
        quantized_query(query, expected_m=None), detect_device(source.device).identity
    )
    dense = replace(
        dense,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        load_path="tma",
        swap_ab=swap_ab,
        target_occupancy=occupancy,
    )
    config = BlockscaledConfig(mode="quantized", dense_config=asdict(dense))
    workspace = make_workspace(source, weight, out=output, config=config, **options)
    saved_scales = scale_storage.view(torch.uint8).clone()
    padded_rows = ((m + 127) // 128) * 128
    pad_mask = swizzle_block_scale(
        (torch.arange(padded_rows, device="cuda")[:, None] >= m)
        .expand(padded_rows, k // 16)
        .contiguous()
    ).flatten()
    sf_start = _a16._layout(m, n, k, True)[0]
    compact = (tile_m, tile_k, swap_ab) == (16, 1024, False) and tile_n in (32, 64)

    with prepared_execution(
        source, weight, out=output, workspace=workspace, config=config, **options
    ) as (_, plan):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            blockscaled.mm(
                source,
                weight,
                out=output,
                workspace=workspace,
                plan=plan,
                activation_global_scale=activation_scale,
            )
        for seed in (1, 2, 3):
            torch.manual_seed(seed)
            source.normal_(std=0.125)
            activation_scale.mul_(0.5)
            expected = _quantized_source(source, activation_scale) @ decoded.T
            output.fill_(float("nan"))
            workspace.fill_(0xFF)
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            assert_close(output, expected)
            activation_scales = workspace[sf_start : sf_start + pad_mask.numel()]
            assert torch.all(activation_scales[pad_mask] == (255 if compact else 0))

        torch.testing.assert_close(
            scale_storage.view(torch.uint8), saved_scales, atol=0, rtol=0
        )
        graph.reset()


@pytest.mark.parametrize("m", range(1, 16))
def test_compact_nvfp4_poisoned_inactive_scale_rows(m):
    test_native_nvfp4_tiles_graph_replay(m, 136, 1024, 16, 64, 1024, False, 2)


@pytest.mark.parametrize("m", [2, 3, 15])
def test_narrow_n_nvfp4_poisoned_inactive_scale_rows(m):
    test_native_nvfp4_tiles_graph_replay(m, 136, 1024, 16, 32, 1024, False, 3)


def test_narrow_n_nvfp4_three_stage_reuse():
    test_native_nvfp4_tiles_graph_replay(16, 264, 5120, 16, 32, 1024, False, 1)


@pytest.mark.parametrize("m", [1, 16])
@pytest.mark.parametrize("occupancy", [1, 2, 3, 4])
def test_narrow_n_nvfp4_persistent_work_replay(m, occupancy):
    test_native_nvfp4_tiles_graph_replay(m, 5120, 3072, 16, 32, 1024, False, occupancy)


@pytest.mark.parametrize("n", [136, 5120])
@pytest.mark.parametrize(
    "m,tile_k,error",
    [
        (1, 1024, None),
        (16, 1024, None),
        (17, 1024, "M<=16"),
        (16, 512, "K1024 staging"),
    ],
)
def test_narrow_n_nvfp4_direct_execution_graph_replay(m, n, tile_k, error):
    from b12x._lib.dense_gemm.execution import dense_gemm

    require_b12x()
    torch.manual_seed(9136)
    activation, decoded_a, _ = make_weight("nvfp4", m, 3072)
    weight, decoded_b, _ = make_weight("nvfp4", n, 3072)
    lhs = (
        activation.values.unsqueeze(-1),
        activation.scale_mma.view(torch.float8_e4m3fn),
    )
    rhs = weight.values.unsqueeze(-1), weight.scale_mma.view(torch.float8_e4m3fn)
    alpha = activation.global_scale * weight.global_scale
    output = torch.empty((m, n, 1), dtype=torch.bfloat16, device="cuda")

    def run():
        return dense_gemm(
            lhs,
            rhs,
            out=output,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            alpha=alpha,
            mma_tiler_mn=(16, 32),
            load_path="tma",
            swap_ab=False,
            _tile_k_override=tile_k,
            _target_occupancy_override=4,
        )

    if error is not None:
        with pytest.raises(ValueError, match=error):
            run()
        return
    expected = decoded_a @ decoded_b.T
    assert run() is output
    assert_close(output[..., 0], expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        alpha.mul_(0.5)
        expected.mul_(0.5)
        output.fill_(float("nan"))
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        assert_close(output[..., 0], expected)
    graph.reset()
