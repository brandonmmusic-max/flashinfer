"""TrellisMX on the released checkpoints: decode, MoE forward, original basis.

Three independent questions, each answered on real sidecar bytes rather than
synthetic payloads:

``a`` decode parity
    The procedural-NPK law in ``w4a8_npk_decode.npk_decode_states_reference``
    must reproduce, bit for bit, the release's own decode on states taken from
    real GLM and DS41 tiles of every projection.  The state extraction itself is
    pinned against b12x's tested reference (``_pack_native_tiles`` /
    ``_cyclic_states`` / ``_tile_position_map``) so a layout slip cannot hide
    behind a matching law.

``b`` MoE forward
    The released sidecar drives ``fused_moe`` ``source_format="trellismx"``
    and must agree with ``trellismx_moe_reference`` at M 1/8/32/256
    with distinct rows and mixed routing, eager and under CUDA graph replay.

``c`` original basis
    Experts decoded out of the K4 rank-0 sidecar, scaled, and un-transformed by
    the declared boundary must match the official layer-0 experts in the base
    checkpoint (MXFP4 dequantized).  A wrong order or axis shows as a cosine
    near zero, so this is the check that tests the *interpretation*, not just
    the arithmetic.

CPU parts run anywhere.  GPU parts skip without CUDA, and every part skips when
its data is absent.

Data (read only; see ``_trellismx_data.py`` for the variables that locate it):
  GLM release   ``B12X_TRELLISMX_GLM_RELEASE``
  DS41 sidecars ``B12X_TRELLISMX_DS41_TESTDATA``
  DS41 base     ``B12X_TRELLISMX_DS41_BASE``
                (lossless re-store of the official checkpoint: every scale plane
                expands back to the receipt's ``raw_sha256``; ``_csf_receipt_sha``
                proves it in-process before any weight is used)
  captures      ``B12X_TRELLISMX_CAPTURES`` (forwards recorded from the release
                runtime)
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from typing import Mapping

import numpy as np
import pytest
import torch

import b12x  # noqa: F401  (an import failure must fail, not skip, the module)

from ._trellismx_data import DS41_BASE as DS41_BASE_CSF
from ._trellismx_data import DS41_TESTDATA, GLM_RELEASE, capture

# --------------------------------------------------------------------------- #
# released data
# --------------------------------------------------------------------------- #

#: sidecar id -> (path, layer, rank, bits, hidden, intermediate_per_rank, top_k)
SIDECARS = {
    "glm-l3-r0": (
        f"{GLM_RELEASE}/sidecars/p8-layer-003-tp4-rank-0.safetensors",
        3,
        0,
        4,
        4096,
        512,
        8,
    ),
    "ds41-k2-l0-r0": (
        f"{DS41_TESTDATA}/K2/trellismx/tp2/layer-000-rank-0.safetensors",
        0,
        0,
        2,
        5120,
        1152,
        6,
    ),
    "ds41-k4-l0-r0": (
        f"{DS41_TESTDATA}/K4/trellismx/tp4/layer-000-rank-0.safetensors",
        0,
        0,
        4,
        5120,
        640,
        6,
    ),
    "ds41-k4-l0-r2": (
        f"{DS41_TESTDATA}/K4/trellismx/tp4/layer-000-rank-2.safetensors",
        0,
        2,
        4,
        5120,
        512,
        6,
    ),
}

#: sidecar id -> sha256 from the release manifest (``trellismx-manifest.json``
#: for GLM, ``K2.manifest.json`` / ``K4.manifest.json`` for DS41).  The DS41
#: entries are the ones the local manifests carry; the GLM entry is checked
#: against the sidecar's own ``__metadata__`` hash instead, because that
#: manifest is keyed by layer rather than by tensor.
MANIFEST_SHA256 = {
    "ds41-k2-l0-r0": "6b01815a5792db1c5edbf325f2178cea958dce8023a5a8a60aa44a8b4b5dcaf7",
    "ds41-k4-l0-r0": "05a29fbbb37d2c202581d487cb922293c39d96116ce89188313547038ac44085",
    "ds41-k4-l0-r2": "1b164b6fd081d0b132e0bed808c1098e5c04695e937e3eabf4630dff9aa27836",
}

#: projectors of one sidecar: (tensor, projection name) in payload order.
PROJECTIONS = (
    ("w13_trellis", 0, "gate"),
    ("w13_trellis", 1, "up"),
    ("w2_trellis", None, "down"),
)

_NPK_MULTIPLIER = 0xCBAC1FED
_NPK_MASK = 0x8FFF8FFF
_NPK_XOR = 0x3B603B60


def _missing(path: str) -> str:
    return f"released data absent: {path}"


def _load_sidecar(name: str) -> tuple[dict[str, np.ndarray], dict[str, str], str]:
    path, *_ = SIDECARS[name]
    if not os.path.exists(path):
        pytest.skip(_missing(path))
    from safetensors import safe_open

    with safe_open(path, "np") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = dict(handle.metadata() or {})
    return tensors, metadata, path


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# procedural-NPK decode, written from the law (independent of the kernel module)
# --------------------------------------------------------------------------- #
def npk_law_bytes(states: np.ndarray) -> np.ndarray:
    """``w = s*0xCBAC1FED mod 2**32; x = (w & M) ^ X; h = fp16(lo)+fp16(hi);
    h += h; e4m3 = cvt.rn.satfinite(h)`` -> uint8 E4M3FN bytes."""

    source = np.asarray(states, dtype=np.uint64) & np.uint64(0xFFFF)
    word = (source * np.uint64(_NPK_MULTIPLIER)) & np.uint64(0xFFFFFFFF)
    mixed = (word & np.uint64(_NPK_MASK)) ^ np.uint64(_NPK_XOR)
    low = (mixed & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16)
    high = (
        ((mixed >> np.uint64(16)) & np.uint64(0xFFFF))
        .astype(np.uint16)
        .view(np.float16)
    )
    total = (low + high).astype(np.float16)  # add.rn.f16x2
    total = (total + total).astype(np.float16)  # alpha-2 compander
    return torch.from_numpy(total).to(torch.float8_e4m3fn).view(torch.uint8).numpy()


# --------------------------------------------------------------------------- #
# t256 tile layout, pinned to b12x's tested reference
# --------------------------------------------------------------------------- #
def tile_edges(tile: np.ndarray, bits: int) -> np.ndarray:
    """One ``16*bits`` int16 tile -> its 256 edge symbols.

    Inverse of ``b12x.testing.reference.trellis_moe._pack_native_tiles``: the
    payload swaps each adjacent int16 word, packs sixteen symbols per span
    MSB-first, and stores ``bits`` sixteen-bit words per span.
    """

    words = np.asarray(tile, dtype=np.uint16).reshape(-1, 2)[:, ::-1].reshape(16, bits)
    symbol = np.arange(256)
    index = (symbol % 16)[:, None] * bits + np.arange(bits)[None, :]
    bit = (words[(symbol // 16)[:, None], index // 16] >> (15 - index % 16)) & 1
    return (bit << (bits - 1 - np.arange(bits))[None, :]).sum(1).astype(np.uint16)


def tile_states(edges: np.ndarray, bits: int) -> np.ndarray:
    """256 edge symbols -> 256 sixteen-bit states (cyclic, window ends at k).

    ``_cyclic_states`` rolls the symbol axis by ``+lag``, i.e. state ``k`` keeps
    symbol ``k`` in its low bits and ``k-1, k-2, ...`` above it.
    """

    states = np.zeros(256, dtype=np.uint32)
    for lag in range((16 + bits - 1) // bits):
        states |= np.asarray(edges, dtype=np.uint32)[(np.arange(256) - lag) % 256] << (
            lag * bits
        )
    return (states & 0xFFFF).astype(np.uint16)


def tile_position_map() -> tuple[np.ndarray, np.ndarray]:
    """State index ``lane*8 + j`` -> (row = N, col = K) inside one 16x16 tile."""

    rows = np.empty(256, dtype=np.int64)
    cols = np.empty(256, dtype=np.int64)
    for lane in range(32):
        quarter = lane % 4
        n0 = 2 * (lane // 8) + ((lane >> 2) & 1)
        ks = (2 * quarter, 2 * quarter + 1, 2 * quarter + 8, 2 * quarter + 9)
        for j in range(8):
            rows[lane * 8 + j] = n0 + (8 if j >= 4 else 0)
            cols[lane * 8 + j] = ks[j % 4]
    return rows, cols


TILE_ROWS, TILE_COLS = tile_position_map()


def sample_tiles(
    tensors: Mapping[str, np.ndarray], *, bits: int, count: int, seed: int = 0
) -> dict[str, np.ndarray]:
    """Sample ``count`` tiles from every projection of a sidecar."""

    generator = np.random.default_rng(seed)
    out = {}
    for tensor, matrix, label in PROJECTIONS:
        payload = tensors[tensor]
        if matrix is None:
            flat = payload.reshape(-1, payload.shape[-1])
        else:
            flat = payload[matrix].reshape(-1, payload.shape[-1])
        index = np.sort(
            generator.choice(
                flat.shape[0], size=min(count, flat.shape[0]), replace=False
            )
        )
        out[label] = flat[index]
    return out


# --------------------------------------------------------------------------- #
# MXFP4-CSF scale planes (``row-base-offset1-u24-exceptions/1``)
# --------------------------------------------------------------------------- #
def expand_csf_plane(
    fixed: np.ndarray, exceptions: np.ndarray, *, rows: int, columns: int
) -> np.ndarray:
    """Expand one CSF scale plane to its ``[rows, columns]`` UE8M0 bytes.

    One base byte per row (sixteen rows per fixed row) followed by one selector
    bit per scale, little-endian, then ``(value << 24) | position`` overrides.
    """

    group = 16
    if fixed.shape[0] * group != rows:
        raise ValueError(f"CSF plane {fixed.shape} does not cover {rows} rows")
    per_row = (fixed.shape[1] - group) // group
    if per_row * 8 < columns:
        raise ValueError(
            f"CSF plane holds {per_row * 8} selector bits, needs {columns}"
        )
    base = fixed[:, :group].reshape(rows)
    selectors = np.unpackbits(
        fixed[:, group:].reshape(rows, per_row), axis=1, bitorder="little"
    )[:, :columns]
    plane = (base[:, None].astype(np.uint16) + selectors).astype(np.uint8).reshape(-1)
    if exceptions.size:
        position = (exceptions & 0xFFFFFF).astype(np.int64)
        if int(position.max()) >= rows * columns:
            raise ValueError("CSF exception position lies outside the scale plane")
        plane[position] = (exceptions >> 24).astype(np.uint8)
    return plane.reshape(rows, columns)


def ue8m0_scale(plane: np.ndarray) -> np.ndarray:
    return np.ldexp(np.ones(plane.shape, np.float32), plane.astype(np.int32) - 127)


def sidecar_scale_plane(
    tensors: Mapping[str, np.ndarray],
    tensor: str,
    *,
    expert: int,
    rows: int,
    columns: int,
) -> np.ndarray:
    """UE8M0 plane of one sidecar scale tensor, CSF or plain."""

    plain = f"{tensor}_ue8m0"
    if plain in tensors:
        plane = tensors[plain][expert]
        if tuple(plane.shape) != (rows, columns):
            raise ValueError(f"{plain} is {plane.shape}, expected {(rows, columns)}")
        return plane
    fixed = tensors[f"{tensor}.mxfp4_csf_fixed"][expert]
    # The released DS41 chunks name the exception planes by matrix, not by the
    # scale tensor: ``w13_scale.mxfp4_csf_fixed`` pairs with ``w13_exceptions``.
    prefix = tensor.split("_")[0]
    exceptions = tensors[f"{prefix}_exceptions"]
    offsets = tensors[f"{prefix}_exception_offsets"]
    return expand_csf_plane(
        fixed,
        exceptions[offsets[expert] : offsets[expert + 1]],
        rows=rows,
        columns=columns,
    )


class _RawSafetensors:
    """Minimal reader for dtype strings numpy cannot name (``F8_E8M0``)."""

    def __init__(self, path: str) -> None:
        self._handle = open(path, "rb")
        length = struct.unpack("<Q", self._handle.read(8))[0]
        self.header = json.loads(self._handle.read(length))
        self._base = 8 + length

    def bytes_of(self, name: str) -> np.ndarray:
        entry = self.header[name]
        begin, end = entry["data_offsets"]
        self._handle.seek(self._base + begin)
        return np.frombuffer(self._handle.read(end - begin), dtype=np.uint8)

    def get(self, name: str, dtype=np.uint8) -> np.ndarray:
        return self.bytes_of(name).view(dtype).reshape(self.header[name]["shape"])

    def close(self) -> None:
        self._handle.close()


def _csf_receipt_sha(shard: str, tensor: str) -> str | None:
    """``raw_sha256`` the -CSF checkpoint recorded for one original tensor."""

    receipt = f"{DS41_BASE_CSF}/receipts/{shard}.scales.json"
    if not os.path.exists(receipt):
        return None
    planes = json.load(open(receipt, "rb"))["per_tensor"]
    for entry in planes:
        if entry["tensor"] == tensor:
            return entry["raw_sha256"]
    return None


def official_expert(projection: str, expert: int) -> np.ndarray:
    """Official DS41 layer-0 expert matrix, MXFP4 dequantized to FP32."""

    shard = "model-00003-of-00048.safetensors"
    path = f"{DS41_BASE_CSF}/tensors/{shard}"
    if not os.path.exists(path):
        pytest.skip(_missing(path))
    name = f"layers.0.ffn.experts.{expert}.{projection}"
    raw = _RawSafetensors(path)
    try:
        packed = raw.get(f"{name}.weight", np.int8).view(np.uint8)
        fixed = raw.get(f"{name}.scale.mxfp4_csf_fixed")
        exceptions = raw.get(f"{name}.scale.mxfp4_csf_exceptions", np.uint32)
    finally:
        raw.close()
    rows, byte_columns = packed.shape
    columns = 2 * byte_columns
    plane = expand_csf_plane(fixed, exceptions, rows=rows, columns=columns // 32)
    expected = _csf_receipt_sha(shard, f"{name}.scale")
    if expected is not None:
        assert hashlib.sha256(plane.tobytes()).hexdigest() == expected, (
            f"{name}.scale does not expand to the -CSF receipt hash"
        )
    codes = np.array(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=np.float32,
    )
    low = codes[packed & 0x0F]
    high = codes[(packed >> 4) & 0x0F]
    values = np.stack((low, high), axis=-1).reshape(rows, columns)
    return values * np.repeat(ue8m0_scale(plane), 32, axis=1)


# --------------------------------------------------------------------------- #
# transforms (declared orders, SPEC section 5.1)
# --------------------------------------------------------------------------- #
def sylvester(order: int) -> np.ndarray:
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return (matrix / np.sqrt(order)).astype(np.float32)


def block_right(matrix: np.ndarray, hadamard: np.ndarray) -> np.ndarray:
    rows, columns = matrix.shape
    order = hadamard.shape[0]
    if columns % order:
        raise ValueError(f"{columns} columns do not divide into {order}-blocks")
    return (matrix.reshape(rows, columns // order, order) @ hadamard).reshape(
        rows, columns
    )


def decode_stored_matrix(
    tensors: Mapping[str, np.ndarray],
    *,
    expert: int,
    matrix: int | None,
    bits: int,
    hidden: int,
    intermediate: int,
    scale_row: int,
) -> np.ndarray:
    """Decode one expert's stored-basis matrix (UE8M0 scales applied).

    ``w13`` keeps the payload's own matrix axis (0 = gate, 1 = up); ``w2`` has
    no matrix axis.  Rows are the output axis, columns the K32-scaled input.
    """

    if matrix is None:
        # w2 carrier is [K16 tile, N16 tile, 16*bits]: dim 0 walks the *input*
        # (intermediate) axis and dim 1 the output (hidden) axis, so the matrix
        # is [hidden, intermediate] with the tile axes in that order.
        payload = tensors["w2_trellis"][expert]
        input_tiles, output_tiles = payload.shape[0], payload.shape[1]
        rows, inputs = hidden, intermediate
        scales = sidecar_scale_plane(
            tensors, "w2_scale", expert=expert, rows=hidden, columns=intermediate // 32
        )
        tiles = payload.reshape(input_tiles, output_tiles, -1)
        out = np.zeros((rows, inputs), dtype=np.float32)
        for output_tile in range(output_tiles):
            for input_tile in range(input_tiles):
                tile = tiles[input_tile, output_tile]
                values = npk_law_bytes(tile_states(tile_edges(tile, bits), bits))
                decoded = (
                    torch.from_numpy(values).view(torch.float8_e4m3fn).float().numpy()
                )
                fragment = np.zeros((16, 16), dtype=np.float32)
                fragment[TILE_ROWS, TILE_COLS] = decoded
                out[
                    output_tile * 16 : output_tile * 16 + 16,
                    input_tile * 16 : input_tile * 16 + 16,
                ] = fragment
        return out * np.repeat(ue8m0_scale(scales), 32, axis=1)
    else:
        # w13 carrier is [hidden tile, row group, 16*bits]: the tile axis comes
        # first, so transpose into the same (row group, hidden tile) order.
        payload = tensors["w13_trellis"][matrix, expert]
        hidden_tiles, row_groups = payload.shape[0], payload.shape[1]
        rows, inputs = intermediate, hidden
        scales = sidecar_scale_plane(
            tensors,
            "w13_scale",
            expert=expert,
            rows=2 * intermediate,
            columns=hidden // 32,
        )[scale_row : scale_row + intermediate]
        tiles = payload.transpose(1, 0, 2).reshape(row_groups, hidden_tiles, -1)
    out = np.zeros((rows, inputs), dtype=np.float32)
    for group in range(row_groups):
        for tile_index in range(hidden_tiles):
            tile = tiles[group, tile_index]
            values = npk_law_bytes(tile_states(tile_edges(tile, bits), bits))
            decoded = torch.from_numpy(values).view(torch.float8_e4m3fn).float().numpy()
            fragment = np.zeros((16, 16), dtype=np.float32)
            fragment[TILE_ROWS, TILE_COLS] = decoded
            out[
                group * 16 : group * 16 + 16, tile_index * 16 : tile_index * 16 + 16
            ] = fragment
    return out * np.repeat(ue8m0_scale(scales), 32, axis=1)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.ravel().astype(np.float64)
    b = right.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm((left - right).astype(np.float64))
        / (np.linalg.norm(right.astype(np.float64)) + 1e-30)
    )


# --------------------------------------------------------------------------- #
# (a) decode parity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_tile_layout_matches_b12x_reference(bits: int) -> None:
    """The CPU tile reader inverts b12x's own packer for every supported rate."""

    from b12x.testing.reference.trellis_moe import _cyclic_states, _pack_native_tiles

    generator = torch.Generator().manual_seed(0xB12 + bits)
    edges = torch.randint(0, 1 << bits, (8, 256), generator=generator)
    payload = _pack_native_tiles(edges, bits)
    reference_states = _cyclic_states(edges, bits).numpy()
    for index in range(payload.shape[0]):
        assert np.array_equal(
            tile_edges(payload[index].numpy(), bits), edges[index].numpy()
        )
        assert np.array_equal(
            tile_states(tile_edges(payload[index].numpy(), bits), bits),
            reference_states[index],
        )


@pytest.mark.parametrize("name", sorted(SIDECARS))
def test_npk_law_matches_w1_on_real_tiles(name: str) -> None:
    """The kernel module's CPU law is bit-identical to the written law on real tiles."""

    from b12x.moe._shared.kernels.w4a8_npk_decode import npk_decode_states_reference

    tensors, metadata, _ = _load_sidecar(name)
    bits = int(metadata["bits"])
    sampled = sample_tiles(tensors, bits=bits, count=64)
    total = 0
    for label, tiles in sampled.items():
        for tile in tiles:
            states = tile_states(tile_edges(tile, bits), bits)
            mine = npk_law_bytes(states)
            theirs = (
                npk_decode_states_reference(torch.from_numpy(states.astype(np.int32)))
                .view(torch.uint8)
                .numpy()
            )
            assert np.array_equal(mine, theirs), f"{name}/{label}: law mismatch"
            total += states.size
    assert total >= 3 * 64 * 256


@pytest.mark.parametrize("name", sorted(SIDECARS))
def test_npk_law_is_permutation_sensitive(name: str) -> None:
    """A neighbouring state must not decode to the same byte (no degenerate law)."""

    tensors, metadata, _ = _load_sidecar(name)
    bits = int(metadata["bits"])
    tile = sample_tiles(tensors, bits=bits, count=1)["gate"][0]
    states = tile_states(tile_edges(tile, bits), bits)
    values = npk_law_bytes(states)
    shifted = npk_law_bytes((states + 1) & 0xFFFF)
    assert not np.array_equal(values, shifted)
    assert len(np.unique(values)) > 8


@pytest.mark.parametrize("name", sorted(SIDECARS))
def test_sidecar_geometry_and_manifest(name: str) -> None:
    """Shapes match the released container formula; DS41 files match the manifest."""

    tensors, metadata, path = _load_sidecar(name)
    _, layer, rank, bits, hidden, intermediate, top_k = SIDECARS[name]
    slots = intermediate // 32
    experts = 288 if name.startswith("glm") else 384
    hidden_tiles = hidden // 16
    assert int(metadata["bits"]) == bits
    assert int(metadata["layer"]) == layer
    assert int(metadata["rank"]) == rank
    assert metadata["law"] in ("procedural-npk", "procedural-mcg-alpha2")
    # DS41 chunks declare their geometry; the GLM overlay sidecar declares the
    # global intermediate size and leaves the rest to the tensor shapes.
    if "hidden" in metadata:
        assert int(metadata["hidden"]) == hidden
        assert int(metadata["topk"]) == top_k
        assert int(metadata["intermediate_per_rank"]) == intermediate
        assert int(metadata["slice_begin"]) == int(metadata["slice_end"]) - intermediate
    else:
        assert int(metadata["global_intermediate"]) == 2048
        assert int(metadata["world_size"]) == 4
        assert int(metadata["local_atom_begin"]) == 16 * rank
    assert metadata["boundary"] == "coupled-h512-h128-suh-svh-v1"
    assert tensors["w13_trellis"].shape == (
        2,
        experts,
        hidden_tiles,
        2 * slots,
        16 * bits,
    )
    assert tensors["w2_trellis"].shape == (experts, 2 * slots, hidden_tiles, 16 * bits)
    assert tensors["intermediate_scales_fp16"].shape == (experts, 3 * intermediate)
    assert tensors["gate_up_suh_fp16"].shape == (hidden,)
    assert tensors["down_svh_fp16"].shape == (hidden,)
    expected = MANIFEST_SHA256.get(name)
    if expected is not None:
        assert _sha256_file(path) == expected


def test_ds41_sidecar_sha256_matches_manifest_after_download() -> None:
    """The downloaded K4 rank-2 sidecar hashes to the manifest value."""

    name = "ds41-k4-l0-r2"
    path, *_ = SIDECARS[name]
    if not os.path.exists(path):
        pytest.skip(_missing(path))
    assert _sha256_file(path) == MANIFEST_SHA256[name]


def test_csf_scale_plane_is_lossless() -> None:
    """CSF expansion of a real DS41 plane is injective and inside the UE8M0 range."""

    tensors, _, _ = _load_sidecar("ds41-k4-l0-r0")
    plane = sidecar_scale_plane(tensors, "w13_scale", expert=0, rows=1280, columns=160)
    assert plane.dtype == np.uint8
    assert plane.shape == (1280, 160)
    assert plane.min() >= 1 and plane.max() <= 254
    # One selector bit per scale: outside the exception list every byte is its
    # row base or one above it, and the exception list is exactly the override.
    fixed = tensors["w13_scale.mxfp4_csf_fixed"][0]
    base = fixed[:, :16].reshape(1280)
    offsets = tensors["w13_exception_offsets"]
    exceptions = tensors["w13_exceptions"][offsets[0] : offsets[1]]
    overridden = np.zeros(1280 * 160, dtype=bool)
    overridden[(exceptions & 0xFFFFFF).astype(np.int64)] = True
    flat = plane.reshape(-1)
    base_flat = np.repeat(base[:, None], 160, axis=1).reshape(-1)
    assert int(overridden.sum()) == int(exceptions.size)
    assert np.all(flat[~overridden] >= base_flat[~overridden])
    assert np.all(flat[~overridden] <= base_flat[~overridden] + 1)


def test_glm_sidecar_metadata_hashes_match_tensor_bytes() -> None:
    """The GLM sidecar's own ``sha256_*`` metadata covers the raw tensor bytes."""

    tensors, metadata, _ = _load_sidecar("glm-l3-r0")
    for tensor in ("w13_trellis", "w2_trellis", "gate_up_suh_fp16", "down_svh_fp16"):
        key = f"sha256_{tensor}"
        if key not in metadata:
            continue
        digest = hashlib.sha256(
            np.ascontiguousarray(tensors[tensor]).tobytes()
        ).hexdigest()
        assert digest == metadata[key], f"{tensor} does not match {key}"


# --------------------------------------------------------------------------- #
# (c) reconstruction in the original basis
# --------------------------------------------------------------------------- #
def _reconstruct(name: str, expert: int) -> dict[str, dict[str, float]]:
    tensors, metadata, _ = _load_sidecar(name)
    bits = int(metadata["bits"])
    intermediate = int(metadata["intermediate_per_rank"])
    hidden = int(metadata["hidden"])
    slice_begin = int(metadata["slice_begin"])
    had512 = sylvester(512)
    had128 = sylvester(128)
    suh = tensors["gate_up_suh_fp16"].astype(np.float32)
    scales = tensors["intermediate_scales_fp16"].astype(np.float32)[expert]
    gate_svh = scales[:intermediate]
    up_svh = scales[intermediate : 2 * intermediate]

    report = {}
    for matrix, projection, svh, official_projection in (
        (0, "gate", gate_svh, "w1"),
        (1, "up", up_svh, "w3"),
    ):
        stored = decode_stored_matrix(
            tensors,
            expert=expert,
            matrix=matrix,
            bits=bits,
            hidden=hidden,
            intermediate=intermediate,
            scale_row=matrix * intermediate,
        )
        # Declared weight-side order (SPEC section 5.1): the stored rows carry
        # gate_svh/up_svh, and the hidden axis carries H512 -> suh -> H128.
        reconstructed = block_right(
            block_right(stored * svh[:, None], had512) * suh, had128
        )
        official = official_expert(official_projection, expert)[
            slice_begin : slice_begin + intermediate
        ]
        report[projection] = {
            "cosine": cosine(reconstructed, official),
            "relative_frobenius": relative_frobenius(reconstructed, official),
            "stored_rms": float(np.sqrt((stored.astype(np.float64) ** 2).mean())),
            "official_rms": float(np.sqrt((official.astype(np.float64) ** 2).mean())),
        }
    return report


@pytest.mark.parametrize("expert", [0])
def test_ds41_layer0_reconstructs_in_original_basis(expert: int) -> None:
    """K4 rank-0 experts un-transformed by the declared order, recorded.

    A per-plane weight-side inverse does **not** exist for this format: the
    coupled boundary applies two H128s to the *interleaved* gate/up axis with
    the svh scales between them, so the stored gate rows are a mixture of the
    official gate and up rows and no per-matrix inverse can separate them.
    The per-plane cosine therefore stays near zero (recorded -0.0004 and
    0.0028), which this test asserts: a per-plane match would mean the stored
    rows are not coupled as the format declares. The valid numerical gate is
    ``test_ds41_layer_output_matches_official``, which runs the release's own
    coupled topology end to end and reaches cosine 0.93-0.98.
    """

    report = _reconstruct("ds41-k4-l0-r0", expert)
    print(json.dumps({str(expert): report}, indent=1, sort_keys=True))
    for projection, row in report.items():
        assert np.isfinite(row["cosine"]) and np.isfinite(row["relative_frobenius"])
        assert row["stored_rms"] > 0 and row["official_rms"] > 0
        assert abs(row["cosine"]) < 0.1, f"{projection} cosine {row['cosine']:.4f}"


def test_ds41_reconstruction_is_not_a_transform_artifact() -> None:
    """Block energies are invariant to the column transform, so a near-zero
    correlation there rules the transform out as the only problem."""

    tensors, metadata, _ = _load_sidecar("ds41-k4-l0-r0")
    stored = decode_stored_matrix(
        tensors,
        expert=0,
        matrix=0,
        bits=int(metadata["bits"]),
        hidden=int(metadata["hidden"]),
        intermediate=int(metadata["intermediate_per_rank"]),
        scale_row=0,
    )
    official = official_expert("w1", 0)[: stored.shape[0]]

    def energy(matrix: np.ndarray) -> np.ndarray:
        return (matrix.astype(np.float64) ** 2).reshape(matrix.shape[0], -1, 512).sum(2)

    correlation = float(
        np.corrcoef(energy(stored).ravel(), energy(official).ravel())[0, 1]
    )
    print(f"512-column block-energy correlation: {correlation:.4f}")
    assert abs(correlation) < 0.9, (
        "block energies agree; the mismatch is a permutation, not the values"
    )


# --------------------------------------------------------------------------- #
# (c2) layer-output gate: the reference against the official layer output
# --------------------------------------------------------------------------- #
def reference_tensors(name: str) -> tuple[dict[str, np.ndarray], dict]:
    """Sidecar tensors as ``trellismx_moe_reference`` takes them.

    DS41 ships its scale planes in CSF, so they are expanded to the plain
    UE8M0 mapping the reference documents (a lossless expansion;
    ``test_csf_scale_plane_is_lossless`` pins the codec).  The reference also
    wants ``coupled_sign_draw_u8``, which the DS41 chunks do not carry: they are
    all-zero (sign draw 0) exactly as the GLM release records it.
    """

    tensors, metadata, _ = _load_sidecar(name)
    meta = _sidecar_meta(name)
    intermediate = meta["intermediate_size"]
    hidden = meta["hidden_size"]
    experts = meta["num_experts"]
    plain = {
        key: value
        for key, value in tensors.items()
        if not key.startswith("w13_scale") and not key.startswith("w2_scale")
    }
    w13 = np.empty((experts, 2 * intermediate, hidden // 32), np.uint8)
    w2 = np.empty((experts, hidden, intermediate // 32), np.uint8)
    for expert in range(experts):
        w13[expert] = sidecar_scale_plane(
            tensors,
            "w13_scale",
            expert=expert,
            rows=2 * intermediate,
            columns=hidden // 32,
        )
        w2[expert] = sidecar_scale_plane(
            tensors, "w2_scale", expert=expert, rows=hidden, columns=intermediate // 32
        )
    plain["w13_scale_ue8m0"] = w13
    plain["w2_scale_ue8m0"] = w2
    plain["coupled_sign_draw_u8"] = np.zeros(experts, np.uint8)
    return plain, meta


def official_activation(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """The checkpoint's own SwiGLU: ``inference/model.py:847-850``."""

    up = torch.clamp(up, min=-10.0, max=10.0)
    gate = torch.clamp(gate, max=10.0)
    return torch.nn.functional.silu(gate) * up


def official_rank_output(
    expert: int, tokens: torch.Tensor, slice_begin: int, slice_end: int
) -> torch.Tensor:
    """One expert's contribution through the official experts of one rank slice."""

    gate = torch.from_numpy(official_expert("w1", expert)[slice_begin:slice_end])
    up = torch.from_numpy(official_expert("w3", expert)[slice_begin:slice_end])
    down = torch.from_numpy(official_expert("w2", expert)[:, slice_begin:slice_end])
    source = tokens.float()
    return official_activation(source @ gate.T, source @ up.T) @ down.T


def block_hadamard(value: torch.Tensor, order: int) -> torch.Tensor:
    """Normalized Sylvester Hadamard over the last axis, in ``order`` blocks."""

    matrix = torch.from_numpy(sylvester(order))
    return (
        value.reshape(*value.shape[:-1], value.shape[-1] // order, order) @ matrix
    ).reshape(value.shape)


def coupled_rank_output(
    tokens: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    *,
    gate_up_suh: torch.Tensor,
    down_svh: torch.Tensor,
    intermediate_scales: torch.Tensor,
    intermediate: int,
) -> torch.Tensor:
    """One expert's rank-slice output through the release's coupled boundary.

    Transcribed from the release's own CPU topology, which is the executable
    specification of the format (``p8_coupled_scales.py:355-402`` in the
    release's runtime image, the ``full_coupled`` branch):

    ``H512 -> *gate_up_suh -> H128 -> [gate|up] -> (H128 -> *svh -> H128) on the
    interleaved 32-atom axis -> *pre_signs -> adjacent-pair split -> silu-cap10
    -> *post_signs -> H128 -> *down_suh -> H128 -> down -> H128 -> *down_svh ->
    H512``.  The sign vectors are all +1 at draw 0, so they are omitted here and
    asserted by the caller.
    """

    source = block_hadamard(block_hadamard(tokens.float(), 512) * gate_up_suh, 128)
    gate_value = (source @ gate.T).half().float()
    up_value = (source @ up.T).half().float()
    rows = gate_value.shape[0]
    interleaved = torch.stack(
        (
            gate_value.view(rows, intermediate // 32, 32),
            up_value.view(rows, intermediate // 32, 32),
        ),
        dim=2,
    ).reshape(rows, 2 * intermediate)
    gate_svh, up_svh, down_suh = intermediate_scales.split(intermediate)
    scale_axis = torch.stack(
        (gate_svh.view(intermediate // 32, 32), up_svh.view(intermediate // 32, 32)),
        dim=1,
    ).reshape(2 * intermediate)
    pre = block_hadamard(interleaved, 128)
    pre = block_hadamard(pre * scale_axis, 128)
    gate_joint = pre[:, 0::2].clamp(max=10.0)
    up_joint = pre[:, 1::2].clamp(min=-10.0, max=10.0)
    activated = gate_joint * torch.sigmoid(gate_joint) * up_joint
    down_input = block_hadamard(block_hadamard(activated, 128) * down_suh, 128)
    product = (down_input @ down.T).half().float()
    return block_hadamard(block_hadamard(product, 128) * down_svh, 512)


def _layer_output_gate(
    name: str, slice_begin: int, slice_end: int, experts, tokens: int, seed: int
):
    """Decode each expert, run the coupled boundary, compare to the official."""

    tensors, metadata, _ = _load_sidecar(name)
    meta = _sidecar_meta(name)
    hidden = meta["hidden_size"]
    intermediate = meta["intermediate_size"]
    bits = meta["bits"]
    gate_up_suh = torch.from_numpy(tensors["gate_up_suh_fp16"].astype(np.float32))
    down_svh = torch.from_numpy(tensors["down_svh_fp16"].astype(np.float32))
    scales = torch.from_numpy(tensors["intermediate_scales_fp16"].astype(np.float32))
    assert bool(torch.all(torch.abs(scales) == 1.0)), (
        "the DS41 chunks carry signed-unit scales; a magnitude would need the sign path"
    )
    generator = torch.Generator().manual_seed(seed)
    # Unit-RMS rows: the MoE input is post-RMSNorm in both models.
    x = torch.randn(tokens, hidden, generator=generator).to(torch.bfloat16)
    report = {}
    for expert in experts:
        # Scale plane is up-gate: payload matrix 0 is gate and reads the second
        # half of the plane (confirmed against the release's own kernel).
        gate = torch.from_numpy(
            decode_stored_matrix(
                tensors,
                expert=expert,
                matrix=0,
                bits=bits,
                hidden=hidden,
                intermediate=intermediate,
                scale_row=intermediate,
            )
        )
        up = torch.from_numpy(
            decode_stored_matrix(
                tensors,
                expert=expert,
                matrix=1,
                bits=bits,
                hidden=hidden,
                intermediate=intermediate,
                scale_row=0,
            )
        )
        down = torch.from_numpy(
            decode_stored_matrix(
                tensors,
                expert=expert,
                matrix=None,
                bits=bits,
                hidden=hidden,
                intermediate=intermediate,
                scale_row=0,
            )
        )
        produced = coupled_rank_output(
            x,
            gate,
            up,
            down,
            gate_up_suh=gate_up_suh,
            down_svh=down_svh,
            intermediate_scales=scales[expert],
            intermediate=intermediate,
        )
        official = official_rank_output(expert, x, slice_begin, slice_end)
        report[str(expert)] = {
            "cosine": cosine(produced.numpy(), official.numpy()),
            "relative_frobenius": relative_frobenius(
                produced.numpy(), official.numpy()
            ),
        }
    return report


@pytest.mark.parametrize(
    "name,slice_begin,slice_end",
    [("ds41-k4-l0-r0", 0, 640), ("ds41-k4-l0-r2", 1280, 1792)],
)
def test_ds41_layer_output_matches_official(
    name: str, slice_begin: int, slice_end: int
) -> None:
    """Layer-output gate: one rank slice's layer output vs the official experts.

    Measured 2026-10-22 11:57: cosine 0.978 / 0.933 (rank 0) and 0.977 / 0.938
    (rank 2), relative Frobenius 0.21 / 0.36-0.39.  The residual is the distance
    between two *different* 4-bit quantizations of the same weights (the release
    is procedural-NPK, the base checkpoint is MXFP4), not a format mismatch: the
    same chain reproduces the release's own GLM kernel at cosine 1.0000
    (``test_glm_reference_matches_release_kernel``).  The bound is 0.90.
    """

    report = _layer_output_gate(name, slice_begin, slice_end, (0, 1), tokens=8, seed=11)
    print(json.dumps({name: report}, indent=1, sort_keys=True))
    for expert, row in report.items():
        assert row["cosine"] > 0.90, (
            f"{name} expert {expert}: cosine {row['cosine']:.4f}"
        )
        assert row["relative_frobenius"] < 0.50


# Single-expert capture: 8 unit-RMS rows, one live route on expert 0 with weight
# 1.0, so the kernel's output is exactly that expert's contribution and the
# comparison needs one decode instead of 288.
@pytest.mark.parametrize(
    "name,slice_begin,slice_end",
    [("ds41-k4-l0-r0", 0, 640), ("ds41-k4-l0-r2", 1280, 1792)],
)
def test_ds41_layer_output_matches_official_reference(
    name: str, slice_begin: int, slice_end: int
) -> None:
    """The layer-output gate through the reference, as specified.

    Measured 2026-10-22 12:47: cosine 0.9771 / 0.9320 (rank 0) and
    0.9765 / 0.9374 (rank 2), the same as the independent chain in
    ``test_ds41_layer_output_matches_official``.  Before the reference's tile-index fix the
    same test gave 0.32, which is what identified the defect.
    """

    from b12x.testing.reference.trellismx_moe import trellismx_moe_reference

    plain, meta = reference_tensors(name)
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(8, meta["hidden_size"], generator=generator).to(torch.bfloat16)
    report = {}
    for expert in (0, 1):
        ids = torch.full((8, meta["top_k"]), -1, dtype=torch.int32)
        ids[:, 0] = expert
        weights = torch.zeros(8, meta["top_k"])
        weights[:, 0] = 1.0
        produced = trellismx_moe_reference(x, ids, weights, plain, meta).float()
        official = official_rank_output(expert, x, slice_begin, slice_end)
        report[str(expert)] = {
            "cosine": cosine(produced.numpy(), official.numpy()),
            "relative_frobenius": relative_frobenius(
                produced.numpy(), official.numpy()
            ),
        }
    print(json.dumps({name: report}, indent=1, sort_keys=True))
    for expert, row in report.items():
        assert row["cosine"] > 0.90, f"{name} expert {expert}: {row['cosine']:.4f}"


GLM_ORACLE = capture("oracle_glm_l3_r0_e0.npz")


def test_glm_reference_matches_release_kernel() -> None:
    """The reference against the release's own P8 kernel on identical inputs.

    The oracle is ``oracle_glm_l3_r0_e0.npz``: 8 unit-RMS bf16 rows with one
    live route on expert 0 through the release's own kernel (GLM layer 3 rank
    0), recorded inside the release's runtime image.  Both this file's
    independent chain (0.999998) and the reference (0.9990, after the tile-index
    fix) reproduce it; the test guards against a regression in either.
    """

    if not os.path.exists(GLM_ORACLE):
        pytest.skip(f"oracle capture absent: {GLM_ORACLE}")
    from b12x.testing.reference.trellismx_moe import trellismx_moe_reference

    data = np.load(GLM_ORACLE, allow_pickle=False)
    tensors, metadata, _ = _load_sidecar("glm-l3-r0")
    meta = dict(
        hidden_size=4096,
        intermediate_size=512,
        num_experts=288,
        top_k=8,
        bits=4,
        rank=0,
        world_size=4,
        sign_draw=0,
    )
    produced = (
        trellismx_moe_reference(
            torch.from_numpy(data["x"]).to(torch.bfloat16),
            torch.from_numpy(data["topk_ids"]).to(torch.int32),
            torch.from_numpy(data["topk_weights"]).float(),
            tensors,
            meta,
        )
        .float()
        .numpy()
    )
    oracle = data["y"]
    score = cosine(produced, oracle)
    print(f"reference vs release kernel: cosine {score:+.4f}")
    assert score > 0.99, f"cosine {score:.4f}"


def test_glm_chain_matches_release_kernel() -> None:
    """The coupled chain in this file against the release's own kernel.

    This is the positive control for the whole format reading: one expert, one
    live route, the same fixed rows.  Measured 2026-10-22 11:57: cosine 1.0000.
    """

    if not os.path.exists(GLM_ORACLE):
        pytest.skip(f"oracle capture absent: {GLM_ORACLE}")
    data = np.load(GLM_ORACLE, allow_pickle=False)
    tensors, metadata, _ = _load_sidecar("glm-l3-r0")
    hidden, intermediate, bits = 4096, 512, 4
    expert = int(np.flatnonzero(data["topk_ids"][0] >= 0)[0])
    live = int(data["topk_ids"][0, expert])
    x = torch.from_numpy(data["x"]).to(torch.bfloat16)
    gate = torch.from_numpy(
        decode_stored_matrix(
            tensors,
            expert=live,
            matrix=0,
            bits=bits,
            hidden=hidden,
            intermediate=intermediate,
            scale_row=intermediate,
        )
    )
    up = torch.from_numpy(
        decode_stored_matrix(
            tensors,
            expert=live,
            matrix=1,
            bits=bits,
            hidden=hidden,
            intermediate=intermediate,
            scale_row=0,
        )
    )
    down = torch.from_numpy(
        decode_stored_matrix(
            tensors,
            expert=live,
            matrix=None,
            bits=bits,
            hidden=hidden,
            intermediate=intermediate,
            scale_row=0,
        )
    )
    produced = coupled_rank_output(
        x,
        gate,
        up,
        down,
        gate_up_suh=torch.from_numpy(tensors["gate_up_suh_fp16"].astype(np.float32)),
        down_svh=torch.from_numpy(tensors["down_svh_fp16"].astype(np.float32)),
        intermediate_scales=torch.from_numpy(
            tensors["intermediate_scales_fp16"].astype(np.float32)[live]
        ),
        intermediate=intermediate,
    ).numpy()
    score = cosine(produced, data["y"])
    print(f"chain vs release kernel (expert {live}): cosine {score:+.6f}")
    assert score > 0.999, f"cosine {score:.6f}"


# --------------------------------------------------------------------------- #
# (b) MoE forward through fused_moe source_format="trellismx"
# --------------------------------------------------------------------------- #
def _gpu() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    # A concrete index: the preparation compares ``tensor.device`` against this
    # value, and ``torch.device("cuda") != torch.device("cuda:0")``.
    return torch.device("cuda", torch.cuda.current_device())


def _sidecar_meta(name: str) -> dict:
    _, layer, rank, bits, hidden, intermediate, top_k = SIDECARS[name]
    return {
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_experts": 288 if name.startswith("glm") else 384,
        "top_k": top_k,
        "bits": bits,
        "layer": layer,
        "rank": rank,
        "world_size": 4 if "k4" in name or name.startswith("glm") else 2,
        "sign_draw": 0,
    }


def _mixed_routing(
    rows: int, experts: int, top_k: int, *, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distinct rows with mixed routing: every expert appears, weights differ."""

    generator = torch.Generator().manual_seed(seed)
    ids = torch.stack(
        [torch.randperm(experts, generator=generator)[:top_k] for _ in range(rows)]
    ).to(device)
    weights = torch.rand(rows, top_k, generator=generator, dtype=torch.float32)
    weights = (weights / weights.sum(-1, keepdim=True)).to(torch.float32)
    return ids.to(torch.int32), weights.to(device)


@pytest.mark.parametrize(
    "name", ["glm-l3-r0", "ds41-k2-l0-r0", "ds41-k4-l0-r0", "ds41-k4-l0-r2"]
)
@pytest.mark.parametrize("rows", [1, 8, 32, 256])
def test_trellismx_moe_matches_reference(name: str, rows: int) -> None:
    """fused_moe ``trellismx`` against the reference, eager and graph replay."""

    from b12x.moe.fused_moe.trellismx import prepare_trellismx_experts
    from b12x.testing.reference.trellismx_moe import trellismx_moe_reference

    device = _gpu()
    tensors, metadata, _ = _load_sidecar(name)
    meta = _sidecar_meta(name)
    experts = meta["num_experts"]
    torch.manual_seed(1234 + rows)
    x = (torch.randn(rows, meta["hidden_size"], device=device) * 0.02).to(
        torch.bfloat16
    )
    ids, weights = _mixed_routing(
        rows, experts, meta["top_k"], seed=7 + rows, device=device
    )

    prepared = prepare_trellismx_experts(
        # The preparation validates that every scale vector is already on the
        # target device, so the caller stages the released carriers there.
        {key: torch.from_numpy(value).to(device) for key, value in tensors.items()},
        metadata,
        num_experts=experts,
        hidden_size=meta["hidden_size"],
        intermediate_size=meta["intermediate_size"],
        top_k=meta["top_k"],
        device=device,
    )
    # Like with like: the release picks its kernel by M — the direct/small-M arm
    # owns m <= 16 and does not quantize the activation, the grouped/materialized
    # arm (m > 16) does (both measured on the recorded forwards).  The
    # reference is matched to the same arm, and it is a CPU
    # reference by design, so it runs on the host while the kernel runs on the GPU.
    plain, _ = reference_tensors(name)
    reference_meta = dict(meta, quantize_activations=rows > 16)
    reference = trellismx_moe_reference(
        x.cpu(), ids.cpu(), weights.cpu(), plain, reference_meta
    ).float()

    from b12x.moe.fused_moe import ExecutionCapacity, bind, plan_execution, run

    execution = plan_execution(
        experts=prepared,
        capacity=ExecutionCapacity(max_tokens=max(rows, 1), top_k=meta["top_k"]),
    )
    spec = execution.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    eager_binding = bind(
        execution, scratch=scratch, a=x, topk_ids=ids, topk_weights=weights
    )
    eager = run(binding=eager_binding)
    # Graph capture requires a caller-owned output buffer.
    output = torch.empty_like(eager)
    graph_binding = bind(
        execution,
        scratch=scratch,
        a=x,
        topk_ids=ids,
        topk_weights=weights,
        output=output,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(binding=graph_binding)
    graph.replay()
    torch.cuda.synchronize()

    def stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
        difference = (a.float() - b.float()).abs()
        scale = b.float().abs().clamp_min(1e-6)
        return float(difference.max()), float((difference / scale).max())

    eager_abs, eager_rel = stats(eager.cpu(), reference)
    graph_abs, graph_rel = stats(captured.cpu(), reference)
    print(
        f"{name} M={rows}: eager max|d|={eager_abs:.3e} max_rel={eager_rel:.3e} "
        f"cosine={cosine(eager.cpu().float().numpy(), reference.float().numpy()):+.5f}; "
        f"graph max|d|={graph_abs:.3e} max_rel={graph_rel:.3e}"
    )
    # BF16 output of a four-term FP32 accumulation chain: one BF16 ulp of the
    # largest output magnitude plus the reference's own FP32 ordering.
    bound = 8 * float(reference.float().abs().max()) * 2**-8
    assert eager_abs <= bound, f"eager {eager_abs} exceeds {bound}"
    # The captured run must still match the reference within the same bound, so replay cannot hide a
    # numerical regression; replay itself may move the result from eager by at most 4 units of one
    # BF16 ulp of the largest output. That spread is the atomic scatter's FP32 addition order, which
    # changes between runs: `deterministic_output=True` would make replay bitwise, but the
    # pre-quantized-A trellis arm refuses it. This is not a loosening of the
    # eager comparison, which keeps its own 8-unit bound above.
    assert graph_abs <= bound, f"graph {graph_abs} exceeds {bound}"
    unit = float(reference.float().abs().max()) * 2**-8
    replay = float((eager.float() - captured.float()).abs().max())
    assert replay <= 4 * unit, (
        f"graph replay moved the result by {replay} ({replay / unit:.2f} units, bound 4)"
    )


def _kernel_vs_oracle(path: str, *, rows: int) -> tuple[float, float]:
    """Run the fused_moe trellismx forward on one capture's own inputs."""

    from b12x.moe.fused_moe import ExecutionCapacity, bind, plan_execution, run
    from b12x.moe.fused_moe.trellismx import prepare_trellismx_experts

    device = _gpu()
    data = np.load(path, allow_pickle=False)
    tensors, metadata, _ = _load_sidecar("glm-l3-r0")
    prepared = prepare_trellismx_experts(
        {key: torch.from_numpy(value).to(device) for key, value in tensors.items()},
        metadata,
        num_experts=288,
        hidden_size=4096,
        intermediate_size=512,
        top_k=8,
        device=device,
    )
    x = torch.from_numpy(data["x"]).to(torch.bfloat16).to(device)
    ids = torch.from_numpy(data["topk_ids"]).to(torch.int32).to(device)
    weights = torch.from_numpy(data["topk_weights"]).float().to(device)
    execution = plan_execution(
        experts=prepared, capacity=ExecutionCapacity(max_tokens=x.shape[0], top_k=8)
    )
    spec = execution.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    binding = bind(execution, scratch=scratch, a=x, topk_ids=ids, topk_weights=weights)
    produced = run(binding=binding).float().cpu().numpy()
    oracle = data["y"]
    return cosine(produced, oracle), float(np.abs(produced - oracle).max())


def test_trellismx_kernel_matches_grouped_oracle() -> None:
    """Kernel vs the release's own kernel on the arm that quantizes activations.

    Capture ``oracle_glm_l3_r0_quant.npz``: 64 distinct unit-RMS rows with mixed
    top-8 routing through the **grouped / materialized** arm (m > 16).  That
    is the arm the
    reference's ``quantize_activations`` models (reference vs the same capture:
    1.000000), so the bound is tight.  Measured 2026-10-22 17:52: 0.999901.
    """

    path = capture("oracle_glm_l3_r0_quant.npz")
    if not os.path.exists(path):
        pytest.skip(f"oracle capture absent: {path}")
    score, worst = _kernel_vs_oracle(path, rows=64)
    print(
        f"kernel vs grouped/materialized oracle (m=64): cosine {score:+.6f} max|d| {worst:.6f}"
    )
    assert score >= 0.9999, f"cosine {score:.6f}"


def test_trellismx_kernel_matches_grouped_oracle_m32() -> None:
    """Same, on the 32-row grouped capture (mixed routing, m = 32).

    Measured 2026-10-22 17:52: 0.999903.  The reference matches this capture at
    1.000000 with ``quantize_activations=True`` and at 0.998957 without, which is
    the size of the declared E4M3 activation quantization on this arm.
    """

    path = capture("oracle_glm_l3_r0.npz")
    if not os.path.exists(path):
        pytest.skip(f"oracle capture absent: {path}")
    score, worst = _kernel_vs_oracle(path, rows=32)
    print(
        f"kernel vs grouped/materialized oracle (m=32): cosine {score:+.6f} max|d| {worst:.6f}"
    )
    assert score >= 0.9999, f"cosine {score:.6f}"


def test_trellismx_kernel_matches_smallm_oracle() -> None:
    """Kernel vs the release's own kernel on the direct / small-M arm.

    Capture ``oracle_glm_l3_r0_e0.npz``: 8 rows, one live route on expert 0, so
    m = 8 <= 16 selects the **direct / small-M (RP2)** arm.  Measured 2026-10-22
    17:52: 0.999997.  The 0.998956 figure measured earlier is the
    *reference-with-quantization* distance to this same capture, i.e. the
    documented cost of the declared E4M3 activation quantization against an arm
    that does not apply it — not a kernel defect, and not this row's bound.
    """

    path = GLM_ORACLE
    if not os.path.exists(path):
        pytest.skip(f"oracle capture absent: {path}")
    score, worst = _kernel_vs_oracle(path, rows=8)
    print(
        f"kernel vs direct/small-M oracle (m=8): cosine {score:+.6f} max|d| {worst:.6f}"
    )
    assert score >= 0.9999, f"cosine {score:.6f}"


@pytest.mark.parametrize("name", sorted(SIDECARS))
def test_trellismx_source_rejects_foreign_metadata(name: str) -> None:
    """Wrong law, bits or boundary raise instead of falling back."""

    from b12x.moe.fused_moe.trellismx import prepare_trellismx_experts

    tensors, metadata, _ = _load_sidecar(name)
    meta = _sidecar_meta(name)
    arguments = dict(
        num_experts=meta["num_experts"],
        hidden_size=meta["hidden_size"],
        intermediate_size=meta["intermediate_size"],
        top_k=meta["top_k"],
        device=torch.device("cpu"),
    )
    # The baseline must prepare at all; otherwise every rejection below is a
    # rejection of the wrong thing, which is how a metadata bug looks.
    prepare_trellismx_experts(
        {key: torch.from_numpy(tensor) for key, tensor in tensors.items()},
        metadata,
        **arguments,
    )
    # Each mutation must be rejected for its own reason: a bitrate change may be
    # reported as the shape it implies (16*bits int16 per tile), everything else
    # must name the field.
    for field, value, expected in (
        ("law", "procedural-mcg", ("law",)),
        ("bits", "3", ("bits", "shape")),
        ("boundary", "identity", ("boundary",)),
    ):
        broken = dict(metadata)
        broken[field] = value
        with pytest.raises((ValueError, KeyError, NotImplementedError)) as caught:
            prepare_trellismx_experts(
                {key: torch.from_numpy(tensor) for key, tensor in tensors.items()},
                broken,
                **arguments,
            )
        message = str(caught.value).lower()
        assert any(word in message for word in expected), (
            f"changing {field!r} raised {caught.value!r}, which does not name it"
        )
