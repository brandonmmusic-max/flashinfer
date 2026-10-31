"""The dynamic workspace's A-scale plane is sized to each recipe's own row width.

The FP4 recipes carry one E4M3 activation scale per 16 values, so their plane
row is ``align_up(k // 16, 4)`` bytes; the MX recipes (``w4a8_mx``, ``w6a8_mx``,
``w8a8_mx``) carry one UE8M0 byte per 32 values, so theirs is
``align_up(k // 32, 4)``. Sizing every recipe at the FP4 width doubled the MX
plane, and ``_dynamic_rows_padded_limit``'s row budget followed the doubled
width.

The plane is consumed as a flat buffer (``_packed_input_binding_views`` hands the
launch ``packed_input_scale.view(-1)`` and its base pointer), so these tests pin
the two things that matter: the planned row width equals the width the
consuming launch declares, and the planned buffer is at least as large as the
flat extent the launch reads.
"""

from __future__ import annotations

import math

import pytest
import torch

from b12x.moe.fused_moe import _impl

#: The checkpoint format each recipe is paired with (the pair is validated).
_SOURCE = {
    "nvfp4": "modelopt_nvfp4",
    "w4a8_nvfp4": "modelopt_nvfp4",
    "w4a8_mx": "fp4_e8m0_k32",
    "w6a8_mx": "mxfp6_e2m3",
    "w8a8_mx": "mxfp8_e8m0_k32",
}

#: Activation-scale granularity: 16 values per E4M3 scale, 32 per UE8M0 byte.
_BLOCK = {
    "nvfp4": 16,
    "w4a8_nvfp4": 16,
    "w4a8_mx": 32,
    "w6a8_mx": 32,
    "w8a8_mx": 32,
}

_K = 4096
_N = 512
_E = 8
_TOPK = 4
_ROUTED_ROWS = 64


def _planned_plane(quant_mode: str):
    plan = _impl._plan_core_workspace(
        "dynamic",
        quant_mode,
        _E,
        _E,
        _K,
        _N,
        _TOPK,
        torch.device("cpu"),
        torch.bfloat16,
        routed_rows=_ROUTED_ROWS,
        max_rows=_ROUTED_ROWS,
        source_format=_SOURCE[quant_mode],
    )
    return {spec.name: spec for spec in plan.tensor_specs}["packed_input_scale"]


@pytest.mark.parametrize("quant_mode", sorted(_BLOCK))
def test_scale_plane_row_width_is_the_recipes_scale_granularity(quant_mode) -> None:
    width = _impl._activation_scale_cols_pad(_K, quant_mode)
    assert width == _impl.align_up(_K // _BLOCK[quant_mode], 4)
    spec = _planned_plane(quant_mode)
    assert int(spec.shape[-1]) == width, (quant_mode, spec.shape)


@pytest.mark.parametrize("quant_mode", sorted(_BLOCK))
def test_scale_plane_matches_the_launch_that_reads_it(quant_mode) -> None:
    """The planned row width is the row width the dynamic launch declares.

    The byte-container launch (``w6a8_mx``/``w8a8_mx``) states the MX width
    itself; the FP4 launch states the E4M3 width. The W4A8 launch declares a
    flat ``rows_padded * (k // 32)`` extent instead, so its check is the
    containment below.
    """

    spec = _planned_plane(quant_mode)
    width = int(spec.shape[-1])
    if quant_mode in _impl._MX_BYTE_CONTAINER_QUANT_MODES:
        launch = _impl._DynamicMoEByteContainerLaunch(None, _K, _N, 2 * _N, _TOPK)
        assert launch._cols_pad_k == width
    elif quant_mode == "nvfp4":
        launch = _impl._DynamicMoELaunch(None, _K, _N, 2 * _N, _TOPK)
        assert launch._cols_pad_k == width
    else:
        # w4a8_mx and w4a8_nvfp4 run _DynamicMoEW4A8Launch, which declares a
        # flat ``rows_padded * (k // 32)`` scale extent instead of a row width,
        # so the check is that the plane covers what the launch reads.
        assert quant_mode in ("w4a8_mx", "w4a8_nvfp4")
        assert width >= _K // 32
    # Every launch reads at most `width` bytes per row out of the buffer.
    rows = int(spec.shape[0])
    plane_bytes = math.prod(int(dim) for dim in spec.shape) * spec.dtype.itemsize
    assert plane_bytes >= rows * (_K // _BLOCK[quant_mode])


def test_mx_plane_is_not_the_fp4_width() -> None:
    """The MX width is half the FP4 width, which is the change under test."""

    fp4 = _impl._activation_scale_cols_pad(_K, "nvfp4")
    for quant_mode in ("w4a8_mx", "w6a8_mx", "w8a8_mx"):
        assert _impl._activation_scale_cols_pad(_K, quant_mode) == fp4 // 2
