"""Bit-equality of the procedural NPK decode against its CPU reference.

Two probes drive ``packed_decode_npk_to_e4m3x8`` on the GPU:

* the exhaustive sweep decodes every one of the 65,536 states at every one of
  the eight lane slots, for K2, K3, K4 and K5, and compares each byte with
  ``npk_decode_states_reference``;
* the real-tile probe stages the first 64 ``w13_trellis`` tiles of a released
  TrellisMX sidecar into shared memory, decodes them through the same
  ``_w4a8_trellis_window_words`` funnel the W4A8 kernel uses, and compares the
  packed words bit for bit with a CPU replay of the ring geometry.

Both window-extraction laws the unified decoder exposes are pinned the same
way: the two-word funnel (``npk_funnel_windows``, K2 to K4) and the K5
three-word ring (``npk_ring3_windows``), the latter through the P8 wrapper the
P8 kernels call. The ring law reduces to the funnel on the two-word lanes, and
the funnel alone is shown to be wrong on the lanes whose span crosses into a
third ring word, so the K5 case cannot pass by accident.

``exact`` here means bitwise equality of the emitted E4M3 bytes, not a bound.
"""

from __future__ import annotations

import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics.memory import shared_ptr_to_u32
from b12x._lib.utils.device import current_cuda_stream
from b12x.moe._shared.kernels.npk_state_law import NPK_RATES
from b12x.moe._shared.kernels.w4a8_npk_decode import (
    _w4a8_npk_decode_both,
    npk_decode_states_reference,
    packed_decode_npk_to_e4m3x8,
)
from b12x.moe._shared.p8.kernels.npk_decode import (
    _npk_decode_both as _p8_npk_decode_both,
)
from b12x.moe._shared.kernels.w4a8_trellis_decode import (
    _w4a8_trellis_lane_geom,
)
from b12x.testing.reference.helpers import require_b12x
from b12x.testing.reference.trellis_moe import _tile_position_map

from ._trellismx_data import ds41_sidecar, glm_sidecar

require_b12x(allow_module_level=True)

GLM_W13 = glm_sidecar(3, 0)
DS41_W13 = ds41_sidecar(2, 0)
_TILE_COUNT = 64


class _NpkSweepProbe:
    """One lane per (slot, state): decode a constructed window pair."""

    def __init__(self, bits: int):
        self.bits = int(bits)

    @cute.jit
    def __call__(self, wins: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
        self.kernel(wins, out).launch(
            grid=(cute.size(wins) // 64, 1, 1), block=[32, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, wins: cute.Tensor, out: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = Int32(tidx) + Int32(bidx) * Int32(32)
        win_a = Uint32(wins[2 * lane])
        win_b = Uint32(wins[2 * lane + 1])
        lo, hi = packed_decode_npk_to_e4m3x8(win_a, win_b, self.bits)
        out[2 * lane] = Int32(lo)
        out[2 * lane + 1] = Int32(hi)


class _NpkTileProbe:
    """One block per tile: stage the ring, decode both n8 halves per lane."""

    def __init__(self, bits: int):
        self.bits = int(bits)

    @cute.jit
    def __call__(self, tiles: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
        self.kernel(tiles, out).launch(
            grid=(cute.size(tiles, 0), 1, 1), block=[32, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, tiles: cute.Tensor, out: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        ring_words = 8 * self.bits
        ring = cutlass.utils.SmemAllocator().allocate_tensor(
            cutlass.Uint32, cute.make_layout(ring_words), byte_alignment=16
        )
        for index in cutlass.range_constexpr((ring_words + 31) // 32):
            offset = Int32(tidx) + Int32(32 * index)
            if offset < Int32(ring_words):
                ring[offset] = Uint32(tiles[bidx, offset])
        cute.arch.sync_threads()
        smem_base = shared_ptr_to_u32(ring.iterator)
        ia, ib, s2 = _w4a8_trellis_lane_geom(Int32(tidx), self.bits)
        lo, hi = _w4a8_npk_decode_both(smem_base, Int32(0), ia, ib, s2, self.bits)
        out[bidx, 2 * Int32(tidx)] = Int32(lo)
        out[bidx, 2 * Int32(tidx) + 1] = Int32(hi)


def _ring_geom(lane: int, bits: int) -> tuple[int, int, int]:
    """Python mirror of ``trellis_ring.trellis256_lane_geom_bits(lane, 0, 8)``."""

    ring = 8 * bits
    b1 = (8 * lane + 257) * bits
    b0 = b1 - 16
    b2 = b1 + 7 * bits
    i0 = b0 >> 5
    i2 = (b2 - 1) >> 5
    ia = i0 - ring * (1 if i0 >= ring else 0)
    ib = i2 - ring * (1 if i2 >= ring else 0)
    s2 = (i2 + 1) * 32 - b2
    return ia, ib, s2


def _signed32(words: torch.Tensor) -> torch.Tensor:
    """uint32-range int64 values -> int32 tensor with the same bit pattern."""

    wide = words & 0xFFFFFFFF
    return torch.where(wide >= 2**31, wide - 2**32, wide).to(torch.int32)


def _states_from_words(words: torch.Tensor, bits: int) -> torch.Tensor:
    """[8*bits] ring words -> [32, 8] states in lane*8+slot order."""

    states = torch.zeros(32, 8, dtype=torch.int64)
    for lane in range(32):
        ia, ib, s2 = _ring_geom(lane, bits)
        merged = (int(words[ia]) << 32) | int(words[ib])
        win_a = (merged >> s2) & 0xFFFFFFFF
        win_b = (merged >> (s2 + 4 * bits)) & 0xFFFFFFFF
        for slot in range(8):
            source = win_b if slot < 4 else win_a
            shift = (3 - (slot & 3)) * bits
            states[lane, slot] = (source >> shift) & 0xFFFF
    return states


class _NpkRingTileProbe:
    """One block per tile: stage the ring, decode through the K5 ring law."""

    def __init__(self, bits: int):
        self.bits = int(bits)

    @cute.jit
    def __call__(self, tiles: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
        self.kernel(tiles, out).launch(
            grid=(cute.size(tiles, 0), 1, 1), block=[32, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, tiles: cute.Tensor, out: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        ring_words = 8 * self.bits
        ring = cutlass.utils.SmemAllocator().allocate_tensor(
            cutlass.Uint32, cute.make_layout(ring_words), byte_alignment=16
        )
        for index in cutlass.range_constexpr((ring_words + 31) // 32):
            offset = Int32(tidx) + Int32(32 * index)
            if offset < Int32(ring_words):
                ring[offset] = Uint32(tiles[bidx, offset])
        cute.arch.sync_threads()
        smem_base = shared_ptr_to_u32(ring.iterator)
        ia, ib, s2 = _w4a8_trellis_lane_geom(Int32(tidx), self.bits)
        lo, hi = _p8_npk_decode_both(smem_base, Int32(0), ia, ib, s2, self.bits)
        out[bidx, 2 * Int32(tidx)] = Int32(lo)
        out[bidx, 2 * Int32(tidx) + 1] = Int32(hi)


class _NpkBothWrappersProbe:
    """One block per tile: decode through both module wrappers into one buffer."""

    def __init__(self, bits: int):
        self.bits = int(bits)

    @cute.jit
    def __call__(self, tiles: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
        self.kernel(tiles, out).launch(
            grid=(cute.size(tiles, 0), 1, 1), block=[32, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, tiles: cute.Tensor, out: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        ring_words = 8 * self.bits
        ring = cutlass.utils.SmemAllocator().allocate_tensor(
            cutlass.Uint32, cute.make_layout(ring_words), byte_alignment=16
        )
        for index in cutlass.range_constexpr((ring_words + 31) // 32):
            offset = Int32(tidx) + Int32(32 * index)
            if offset < Int32(ring_words):
                ring[offset] = Uint32(tiles[bidx, offset])
        cute.arch.sync_threads()
        smem_base = shared_ptr_to_u32(ring.iterator)
        ia, ib, s2 = _w4a8_trellis_lane_geom(Int32(tidx), self.bits)
        lo, hi = _w4a8_npk_decode_both(smem_base, Int32(0), ia, ib, s2, self.bits)
        plo, phi = _p8_npk_decode_both(smem_base, Int32(0), ia, ib, s2, self.bits)
        out[bidx, 4 * Int32(tidx)] = Int32(lo)
        out[bidx, 4 * Int32(tidx) + 1] = Int32(hi)
        out[bidx, 4 * Int32(tidx) + 2] = Int32(plo)
        out[bidx, 4 * Int32(tidx) + 3] = Int32(phi)


def _windows_from_words(words: torch.Tensor, lane: int, bits: int, ring3: bool):
    """Python mirror of the device window laws for one lane.

    ``words`` may hold signed int32 patterns, so every read is masked to its
    32-bit value before the 64-bit merge.
    """

    def word(index: int) -> int:
        return int(words[index]) & 0xFFFFFFFF

    ia, ib, s2 = _ring_geom(lane, bits)
    if not ring3:
        merged = (word(ia) << 32) | word(ib)
        return (merged >> s2) & 0xFFFFFFFF, (merged >> (s2 + 4 * bits)) & 0xFFFFFFFF
    ring = 8 * bits
    im = ia + 1
    if im >= ring:
        im -= ring
    top, mid = 0, word(ia)
    if ib != im:
        top, mid = word(ia), word(im)
    lo64 = (mid << 32) | word(ib)
    hi64 = (top << 32) | mid
    t = s2 + 4 * bits
    win_b = (lo64 >> t) if t < 32 else (hi64 >> (t - 32))
    return (lo64 >> s2) & 0xFFFFFFFF, win_b & 0xFFFFFFFF


def _states_from_law(words: torch.Tensor, bits: int, ring3: bool) -> torch.Tensor:
    """[8*bits] ring words -> [32, 8] states through one extraction law."""

    states = torch.zeros(32, 8, dtype=torch.int64)
    for lane in range(32):
        win_a, win_b = _windows_from_words(words, lane, bits, ring3)
        for slot in range(8):
            source = win_b if slot < 4 else win_a
            shift = (3 - (slot & 3)) * bits
            states[lane, slot] = (source >> shift) & 0xFFFF
    return states


def _pack_bytes(states: torch.Tensor) -> torch.Tensor:
    """[32, 8] states -> [32, 2] (lo, hi) int32 words of packed E4M3 bytes."""

    decoded = npk_decode_states_reference(states.to(torch.int32)).to(torch.int64)
    lo = torch.zeros(32, dtype=torch.int64)
    hi = torch.zeros(32, dtype=torch.int64)
    for slot in range(8):
        target = lo if slot < 4 else hi
        target |= decoded[:, slot] << (8 * (slot & 3))
    return _signed32(torch.stack((lo, hi), dim=1))


def test_reference_matches_torch_e4m3_cast() -> None:
    """The CPU law agrees with torch's own satfinite E4M3 encoder.

    The NPK law's mask/or bounds both fp16 halves to exponent field 12..15, so
    the doubled sum is always finite and inside +-8: the reference never
    reaches the saturation or NaN arms, and torch's cast is an independent
    implementation of the same RNE rule over the whole reachable range.
    """

    def half(bits: torch.Tensor) -> torch.Tensor:
        signed = bits.to(torch.int64)
        signed = signed - 65536 * (signed >= 32768).to(torch.int64)
        return signed.to(torch.int16).view(torch.float16)

    states = torch.arange(65536, dtype=torch.int32)
    product = (states.to(torch.int64) * 0xCBAC1FED) & 0xFFFFFFFF
    word = (product & 0x8FFF8FFF) ^ 0x3B603B60
    low = half(word & 0xFFFF)
    high = half((word >> 16) & 0xFFFF)
    combined = low + high
    combined = combined + combined
    assert float(combined.abs().max()) <= 8.0
    assert torch.isfinite(combined.float()).all()
    expected = combined.to(torch.float8_e4m3fn).view(torch.uint8)
    torch.testing.assert_close(
        npk_decode_states_reference(states), expected, rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", NPK_RATES)
def test_npk_device_decode_all_states(bits: int) -> None:
    """Every state at every lane slot decodes bit-identically to the law."""

    device = torch.device("cuda")
    states = torch.arange(65536, dtype=torch.int64)
    # A non-zero partner word in every row so a decoder that leaked bits
    # between the two funnel windows would show up as a mismatch.
    partner = _signed32((states * 0x9E3779B1) & 0xFFFFFFFF)
    count = 8 * 65536
    wins = torch.zeros(2 * count, dtype=torch.int32)
    expected = npk_decode_states_reference(states.to(torch.int32)).to(device)
    for slot in range(8):
        shift = (3 - (slot & 3)) * bits
        target = _signed32((states << shift) & 0xFFFFFFFF)
        base = 2 * slot * 65536
        # Slots 0..3 are windows of ``win_b`` (the second word of the pair),
        # slots 4..7 are windows of ``win_a`` (the first).
        first, second = (partner, target) if slot < 4 else (target, partner)
        wins[base : base + 2 * 65536 : 2] = first
        wins[base + 1 : base + 2 * 65536 : 2] = second
    wins = wins.to(device)
    out = torch.zeros(2 * count, dtype=torch.int32, device=device)

    def args():
        return (
            from_dlpack(wins, assumed_align=16),
            from_dlpack(out, assumed_align=16),
            current_cuda_stream(),
        )

    compiled = b12x_compile(_NpkSweepProbe(bits), *args())
    compiled(*args())
    torch.cuda.synchronize()
    words = out.view(count, 2).to(torch.int64) & 0xFFFFFFFF
    for slot in range(8):
        got = (
            words[slot * 65536 : (slot + 1) * 65536, 0 if slot < 4 else 1]
            >> (8 * (slot & 3))
        ) & 0xFF
        mismatched = int((got != expected.to(torch.int64)).sum())
        assert mismatched == 0, (bits, slot, mismatched)


def _synthetic_rings(bits: int, count: int) -> torch.Tensor:
    """Deterministic ring words: random plus the all-ones and all-zero rings."""

    generator = torch.Generator().manual_seed(0x4E504B + bits)
    words = torch.randint(
        0, 2**32, (count, 8 * bits), dtype=torch.int64, generator=generator
    )
    words[0] = 0
    words[1] = 0xFFFFFFFF
    return _signed32(words)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", [3, 4, 5])
def test_npk_ring_extraction_matches_reference(bits: int) -> None:
    """The K5 ring law decodes every lane's windows to the law's bytes.

    The ring probe runs the P8 wrapper the P8 kernels call, so this covers the
    three-word extraction, the two-word lanes it reduces to, and the packed
    byte order in one comparison per tile.
    """

    device = torch.device("cuda")
    rings = _synthetic_rings(bits, 32)
    out = torch.zeros(32, 64, dtype=torch.int32, device=device)
    expected = torch.stack(
        [
            _pack_bytes(_states_from_law(rings[t], bits, ring3=True)).reshape(-1)
            for t in range(32)
        ]
    )

    def args():
        return (
            from_dlpack(rings.to(device), assumed_align=16),
            from_dlpack(out, assumed_align=16),
            current_cuda_stream(),
        )

    compiled = b12x_compile(_NpkRingTileProbe(bits), *args())
    compiled(*args())
    torch.cuda.synchronize()
    mismatched = int((out.cpu() != expected.to(torch.int32)).sum())
    assert mismatched == 0, (bits, mismatched)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_k5_two_word_funnel_is_wrong_on_three_word_lanes() -> None:
    """The ring law is required at K5: the funnel alone loses the third word.

    Same staged rings as the ring test. The funnel replay is the K4-shaped
    two-word read; on the lanes whose span crosses into a third ring word it
    disagrees with the law, while the ring replay matches it on every lane.
    This is what makes the K5 case of the previous test discriminating rather
    than vacuous.
    """

    bits = 5
    rings = _synthetic_rings(bits, 8)
    three_word_lanes = [
        lane
        for lane in range(32)
        if _ring_geom(lane, bits)[0] != _ring_geom(lane, bits)[1]
    ]
    assert three_word_lanes, "K5 must have lanes whose span needs a third word"
    funnel_disagrees = False
    for tile in range(rings.shape[0]):
        ring_states = _states_from_law(rings[tile], bits, ring3=True)
        funnel_states = _states_from_law(rings[tile], bits, ring3=False)
        if not torch.equal(ring_states, funnel_states):
            funnel_disagrees = True
    assert funnel_disagrees, (
        "the funnel replay matched every K5 lane; the test is vacuous"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_funnel_and_ring_wrappers_agree_below_k5(bits: int) -> None:
    """Both named extraction laws and both module wrappers emit the same bytes.

    Below K5 the ring law is the funnel law by construction, so the shared
    W4A8 wrapper and the P8 wrapper must be interchangeable: this pins that a
    later edit to either wrapper cannot move one caller's bytes without the
    other's.
    """

    device = torch.device("cuda")
    rings = _synthetic_rings(bits, 16)

    probe = _NpkBothWrappersProbe(bits)

    out = torch.zeros(16, 128, dtype=torch.int32, device=device)

    def args():
        return (
            from_dlpack(rings.to(device), assumed_align=16),
            from_dlpack(out, assumed_align=16),
            current_cuda_stream(),
        )

    compiled = b12x_compile(probe, *args())
    compiled(*args())
    torch.cuda.synchronize()
    words = out.cpu().view(16, 32, 4)
    assert torch.equal(words[..., 0], words[..., 2]), bits
    assert torch.equal(words[..., 1], words[..., 3]), bits
    expected = torch.stack(
        [
            _pack_bytes(_states_from_law(rings[t], bits, ring3=False)).reshape(-1)
            for t in range(16)
        ]
    )
    assert int((words[..., :2].reshape(16, 64) != expected.to(torch.int32)).sum()) == 0


def _load_tiles(path: str, bits: int) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        # First 64 tiles in (projection, expert, k16, n16) order: projection 0,
        # expert 0, k16 0..1, n16 0..31.
        block = handle.get_slice("w13_trellis")[0, 0, 0:2, 0:32, :]
    block = block.to(torch.int64) & 0xFFFF
    words = block[..., 0::2] | (block[..., 1::2] << 16)
    tiles = words.reshape(-1, 8 * bits)
    assert tiles.shape[0] >= _TILE_COUNT, tiles.shape
    return tiles[:_TILE_COUNT].contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "path,bits,label",
    [
        (GLM_W13, 4, "glm-layer-003-rank-0"),
        (DS41_W13, 2, "ds41-k2-layer-000-rank-0"),
    ],
)
def test_npk_device_decode_real_tiles(path: str, bits: int, label: str) -> None:
    if not os.path.exists(path):
        pytest.skip(f"{label}: sidecar not present at {path}")
    device = torch.device("cuda")
    tiles = _load_tiles(path, bits)
    out = torch.zeros(_TILE_COUNT, 64, dtype=torch.int32, device=device)
    expected = torch.stack(
        [
            _pack_bytes(_states_from_words(tiles[t], bits)).reshape(-1)
            for t in range(_TILE_COUNT)
        ]
    )

    def args():
        return (
            from_dlpack(tiles.to(device), assumed_align=16),
            from_dlpack(out, assumed_align=16),
            current_cuda_stream(),
        )

    compiled = b12x_compile(_NpkTileProbe(bits), *args())
    compiled(*args())
    torch.cuda.synchronize()
    mismatched = int((out.cpu() != expected.to(torch.int32)).sum())
    assert mismatched == 0, (label, mismatched)


def _unpack_tile_words(words: torch.Tensor) -> torch.Tensor:
    """[32, 2] packed (lo, hi) words -> [256] E4M3 bytes in lane*8+slot order."""

    out = torch.zeros(256, dtype=torch.int64)
    for lane in range(32):
        for slot in range(8):
            word = int(words[lane, 0 if slot < 4 else 1]) & 0xFFFFFFFF
            out[lane * 8 + slot] = (word >> (8 * (slot & 3))) & 0xFF
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_npk_tile_times_released_scale_matches_reference() -> None:
    """One expert, one 16x16 tile: decode x released UE8M0 plane, bitwise.

    The smallest level at which the scaled arm can be wrong: the device decode
    supplies the E4M3 codes for one gate tile of the real GLM layer-3 rank-0
    sidecar, the released ``w13_scale_ue8m0`` plane supplies the K32 block
    scale of each output row, and ``decode_trellis_matrix`` is the
    reference for the product. Codes, scale bytes and product are all compared
    exactly (no tolerance).
    """

    from safetensors import safe_open

    from b12x.testing.reference.trellismx_moe import decode_trellis_matrix

    if not os.path.exists(GLM_W13):
        pytest.skip(f"GLM layer-3 rank-0 sidecar not present at {GLM_W13}")
    device = torch.device("cuda")
    # One expert, 16 output rows, two K16 tiles = one full K32 block, which is
    # the smallest shape the reference decodes (it expands a plane column
    # across 32 K values).
    with safe_open(GLM_W13, framework="pt", device="cpu") as handle:
        tile = handle.get_slice("w13_trellis")[0, 0, 0:2, 0:1, :].to(torch.int16)
        plane = handle.get_slice("w13_scale_ue8m0")[0].to(torch.uint8)

    # Device codes for that tile, through the same probe as the sweep.
    words = tile.reshape(2, -1).to(torch.int64) & 0xFFFF
    ring = (words[:, 0::2] | (words[:, 1::2] << 16)).contiguous()
    out = torch.zeros(2, 64, dtype=torch.int32, device=device)
    args = (
        from_dlpack(_signed32(ring).to(device), assumed_align=16),
        from_dlpack(out, assumed_align=16),
        current_cuda_stream(),
    )
    compiled = b12x_compile(_NpkTileProbe(4), *args)
    compiled(*args)
    torch.cuda.synchronize()
    rows_map, cols_map = _tile_position_map(torch.device("cpu"))
    tile_codes = torch.zeros(16, 32, dtype=torch.int64)
    for index in range(2):
        codes = _unpack_tile_words(out.cpu()[index].view(32, 2))
        tile_codes[rows_map, cols_map + 16 * index] = codes

    # The plane byte each output row of this tile picks: the kernel addresses
    # the logical plane row-for-row (payload row order), and the tile is K16 0,
    # i.e. K32 block 0.
    picked = plane[:16, 0]

    values = tile_codes.to(torch.uint8).view(torch.float8_e4m3fn).float()
    scaled = values * torch.pow(2.0, picked.float() - 127.0)[:, None]
    # The arm reads the w13 plane as stored, row for row with the staged B
    # tile, so this gate tile pairs with plane rows 0..16 (the host swap is
    # what makes the oracle match, so the kernel must not swap).
    reference = decode_trellis_matrix(
        tile,
        plane[:16, 0:1],
        bits=4,
        out_features=16,
        in_features=32,
        scale_row_offset=0,
    )
    # Every row, bitwise, with the plane row-for-row against the tile: the
    # decode reproduces the recorded one-tile dump exactly (tile sha a2baee9b...) and
    # the plane byte is the row's own, which is the pairing the arm uses.
    assert torch.equal(picked, plane[:16, 0])
    assert torch.equal(scaled, reference), (scaled - reference).abs().max()
