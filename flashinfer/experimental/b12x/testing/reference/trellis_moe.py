"""Torch reference for trellis MoE execution.

Builds native-tile trellis payloads from explicit edge symbols and computes
the decoded weights plus a float32 MoE forward that mirrors the kernel-side
transforms (H128 incoherence rotations at the activation boundary; optional
intermediate-Hadamard interleaved boundary). Inputs and outputs live in the rotated bases
the kernels consume and produce, so comparisons need no host-side residual
transforms. Activation quantization is not emulated: kernel outputs match to
MXFP8 noise, so acceptance uses cosine and relative-L2 bounds.
"""

from __future__ import annotations

import torch

from b12x._lib.quant.lut_e4m3 import lut_e4m3_direct_table_cpu
from b12x.moe._shared.kernels.activations import (
    SITU_DEFAULT_BETA,
    SITU_DEFAULT_LINEAR_BETA,
)
from b12x.moe._shared.trellis_codebooks import (
    LUT_E4M3,
    PROCEDURAL_NPK,
    validate_codebook_bits,
)

_RATE_INDEX = {2: 0, 3: 1, 4: 2}

_W4A8_CODEBOOKS = (LUT_E4M3, PROCEDURAL_NPK)


def _check_w4a8_codebook(codebook: str, bits: int | None = None) -> str:
    """Reject codebook ids the W4A8 E4M3 reference does not decode."""

    text = str(codebook).strip().lower()
    if text not in _W4A8_CODEBOOKS:
        raise ValueError(
            f"unsupported trellis codebook {codebook!r}; the W4A8 E4M3 "
            f"reference decodes {LUT_E4M3!r} or {PROCEDURAL_NPK!r}"
        )
    if bits is not None:
        validate_codebook_bits(text, int(bits))
    return text


def _decode_states(
    states: torch.Tensor, bits: int, codebook: str, device: torch.device
) -> torch.Tensor:
    """Map [W, 256] 16-bit states to float32 weights through one codebook."""

    if codebook == PROCEDURAL_NPK:
        # The procedural law is a CPU reference (pinned on the host, bit for
        # bit); import it lazily so this module stays
        # importable without the kernel DSL.
        from b12x.moe._shared.kernels.w4a8_npk_decode import (
            npk_decode_states_reference,
        )

        bytes_cpu = npk_decode_states_reference(states.detach().cpu())
        return (
            bytes_cpu.to(device=device)
            .view(torch.float8_e4m3fn)
            .float()
            .reshape(states.shape)
        )
    direct = lut_e4m3_direct_table_cpu().to(device)
    rate = _RATE_INDEX[bits] << 16
    return (
        direct[rate + states.reshape(-1)]
        .view(torch.float8_e4m3fn)
        .float()
        .reshape(states.shape)
    )


def _hadamard(order: int, device: torch.device) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < order:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / (order**0.5)).to(device=device, dtype=torch.float32)


def _pack_native_tiles(edges: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack [W, 256] edge symbols into [W, 16*bits] int16 native tiles."""

    values = edges.to(torch.int64) & ((1 << bits) - 1)
    spans = values.reshape(-1, 16, 16)
    symbol_shifts = torch.arange(bits - 1, -1, -1, device=edges.device)
    bitstream = ((spans[..., None] >> symbol_shifts) & 1).reshape(-1, 16, bits * 16)
    word_shifts = torch.arange(15, -1, -1, device=edges.device)
    words = (bitstream.reshape(-1, 16, bits, 16) << word_shifts).sum(dim=-1)
    flat = words.reshape(-1, 16 * bits)
    return (
        flat.reshape(-1, 16 * bits // 2, 2)
        .flip(-1)
        .reshape(-1, 16 * bits)
        .to(torch.int16)
    )


def _cyclic_states(edges: torch.Tensor, bits: int) -> torch.Tensor:
    """[W, 256] edge symbols -> [W, 256] 16-bit states (cyclic history)."""

    states = torch.zeros_like(edges, dtype=torch.int64)
    for lag in range((16 + bits - 1) // bits):
        states |= torch.roll(edges.to(torch.int64), shifts=lag, dims=-1) << (lag * bits)
    return states & 0xFFFF


def _tile_position_map(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """State index (lane*8+j) -> (row n, col k) inside one K16xN16 tile."""

    rows = torch.empty(256, dtype=torch.int64)
    cols = torch.empty(256, dtype=torch.int64)
    for lane in range(32):
        r = lane % 4
        n0 = 2 * (lane // 8) + ((lane >> 2) & 1)
        ks = (2 * r, 2 * r + 1, 2 * r + 8, 2 * r + 9)
        for j in range(8):
            rows[lane * 8 + j] = n0 + (8 if j >= 4 else 0)
            cols[lane * 8 + j] = ks[j % 4]
    return rows.to(device), cols.to(device)


def build_trellis_weight(
    generator: torch.Generator,
    experts: int,
    out_features: int,
    in_features: int,
    bits: int,
    device: torch.device,
    codebook: str = LUT_E4M3,
    scale_plane: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random payload [E, K16, N16, 16*bits] i16 and weights [E, N, K] f32.

    The payload axes follow the kernel layout: K16 tiles over ``in_features``
    and N16 tiles over ``out_features``, each tile owning a 16x16 fragment
    of the weight matrix. ``codebook`` selects the decode law that turns each
    tile's 256 tail-biting 16-bit states into E4M3 weights: ``lut_e4m3``
    (default) or ``procedural_npk``.

    ``scale_plane`` is the UE8M0/K32 weight plane in the *weights'* own row
    order, ``uint8 [E, out_features, in_features // 32]``; each weight is then
    multiplied by ``2 ** (byte - 127)``, which is the block-scaled MMA's
    per-K32 weight scale. ``None`` keeps the unscaled (unit-scale) weights.
    """

    codebook = _check_w4a8_codebook(codebook, bits)
    k16 = in_features // 16
    n16 = out_features // 16
    windows = experts * k16 * n16
    edges = torch.randint(
        0, 1 << bits, (windows, 256), generator=generator, device="cpu"
    ).to(device)
    payload = _pack_native_tiles(edges, bits).reshape(experts, k16, n16, 16 * bits)
    states = _cyclic_states(edges, bits)
    values = _decode_states(states, bits, codebook, device)
    if scale_plane is not None:
        plane = scale_plane.to(device=device)
        expected = (experts, out_features, in_features // 32)
        if tuple(plane.shape) != expected or plane.dtype != torch.uint8:
            raise ValueError(
                f"scale_plane must be uint8 {expected}, got "
                f"{plane.dtype} {tuple(plane.shape)}"
            )
        # ``values`` rows are the windows in (expert, K16 tile, N16 tile)
        # order with the N16 axis fastest, and the 256 states of a window are
        # in lane*8+j order. A window spans 16 output rows and 16 K columns,
        # i.e. exactly one K32 block (k16 // 2), so each of its 256 states
        # takes the plane byte of its own output row in that block.
        window = torch.arange(windows, device=device)
        n16_index = window % n16
        k16_index = (window // n16) % k16
        expert_index = window // (n16 * k16)
        plane_f32 = plane.to(torch.float32)
        rows_map, _ = _tile_position_map(device)
        scale_bytes = plane_f32[
            expert_index[:, None],
            n16_index[:, None] * 16 + rows_map[None, :],
            k16_index[:, None] // 2,
        ]
        values = values * torch.pow(2.0, scale_bytes - 127.0)
    rows, cols = _tile_position_map(device)
    tiles = torch.zeros(windows, 16, 16, dtype=torch.float32, device=device)
    tiles[:, rows, cols] = values
    tiles = tiles.reshape(experts, k16, n16, 16, 16)
    weights = (
        tiles.permute(0, 2, 3, 1, 4)
        .reshape(experts, out_features, in_features)
        .contiguous()
    )
    return payload, weights


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    beta = SITU_DEFAULT_BETA
    linear_beta = SITU_DEFAULT_LINEAR_BETA
    return (
        beta
        * torch.tanh(gate / beta)
        * torch.sigmoid(gate)
        * linear_beta
        * torch.tanh(up / linear_beta)
    )


def _boundary_ordinary(
    gate: torch.Tensor,
    up: torch.Tensor,
    rotations: torch.Tensor,
    experts_idx: torch.Tensor,
    intermediate: int,
    device: torch.device,
) -> torch.Tensor:
    had = _hadamard(128, device)
    rg = rotations[experts_idx, :intermediate].float()
    ru = rotations[experts_idx, intermediate : 2 * intermediate].float()
    sd = rotations[experts_idx, 2 * intermediate : 3 * intermediate].float()
    blocks = intermediate // 128
    g = (gate.reshape(-1, blocks, 128) @ had).reshape(gate.shape) * rg
    u = (up.reshape(-1, blocks, 128) @ had).reshape(up.shape) * ru
    h = _situ(g, u) * sd
    return (h.reshape(-1, blocks, 128) @ had).reshape(h.shape)


def _boundary_intermediate_hadamard(
    gate: torch.Tensor,
    up: torch.Tensor,
    rotations: torch.Tensor,
    experts_idx: torch.Tensor,
    intermediate: int,
    device: torch.device,
) -> torch.Tensor:
    had = _hadamard(128, device)
    rows = gate.shape[0]
    pre_blocks = intermediate // 64
    # Interleaved pre window per 64-neuron block:
    # [g[a:a+32], u[a:a+32], g[a+32:a+64], u[a+32:a+64]].
    pos = torch.arange(128, device=device)
    chunk = pos >> 5
    slot = chunk & 1
    half = chunk >> 1
    local = (half * 32 + (pos & 31)).to(torch.int64)
    pre = torch.empty(rows, pre_blocks, 128, dtype=torch.float32, device=device)
    g_blocks = gate.reshape(rows, pre_blocks, 64)
    u_blocks = up.reshape(rows, pre_blocks, 64)
    src = torch.where(
        (slot == 0).unsqueeze(0).unsqueeze(0),
        g_blocks[:, :, local],
        u_blocks[:, :, local],
    )
    pre = src
    s1 = pre @ had
    coord = torch.arange(pre_blocks, device=device)[:, None] * 64 + local[None, :]
    ua_index = slot[None, :] * intermediate + coord
    ua = (
        rotations[experts_idx][:, ua_index.reshape(-1)]
        .float()
        .reshape(rows, pre_blocks, 128)
    )
    s1 = s1 * ua
    s2 = s1 @ had
    sign_index = (
        3 * intermediate
        + torch.arange(pre_blocks, device=device)[:, None] * 128
        + pos[None, :]
    )
    presign = (
        rotations[experts_idx][:, sign_index.reshape(-1)]
        .float()
        .reshape(rows, pre_blocks, 128)
    )
    s2 = s2 * presign
    h_pairs = _situ(s2[..., 0::2], s2[..., 1::2])
    h = h_pairs.reshape(rows, intermediate)
    post_blocks = intermediate // 128
    ub = rotations[experts_idx, 5 * intermediate : 6 * intermediate].float()
    dn = rotations[experts_idx, 2 * intermediate : 3 * intermediate].float()
    v = (h * ub).reshape(rows, post_blocks, 128) @ had
    v = (v.reshape(rows, intermediate) * dn).reshape(rows, post_blocks, 128)
    v = v @ had
    return v.reshape(rows, intermediate)


def trellis_moe_reference(
    x_rot: torch.Tensor,
    w13_weights: torch.Tensor,
    w2_weights: torch.Tensor,
    rotations: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    intermediate_hadamard: bool,
    codebook: str = LUT_E4M3,
) -> torch.Tensor:
    """Float32 MoE forward in the kernels' rotated input/output bases.

    ``x_rot`` is the already-rotated kernel input; the result is the raw
    scatter-basis output (pre any residual unrotation). ``w13_weights`` is
    [2, E, I, K] (gate, up); ``w2_weights`` is [E, K, I].

    ``codebook`` names the decode law that produced the weight tensors
    (``build_trellis_weight``'s argument). The forward itself is
    codebook-independent -- decoding happened when the weights were built --
    so the argument exists to keep the scaffold's parametrization symmetric
    and to reject a call that names an arm this reference does not model.
    """

    _check_w4a8_codebook(codebook)
    device = x_rot.device
    m, hidden = x_rot.shape
    intermediate = w13_weights.shape[2]
    out = torch.zeros(m, hidden, dtype=torch.float32, device=device)
    flat_ids = topk_ids.reshape(-1).long()
    flat_w = topk_weights.reshape(-1).float()
    token_of = (
        torch.arange(m, device=device).repeat_interleave(topk_ids.shape[1]).long()
    )
    xin = x_rot.float()[token_of]
    gate = torch.einsum("rk,rik->ri", xin, w13_weights[0, flat_ids].float())
    up = torch.einsum("rk,rik->ri", xin, w13_weights[1, flat_ids].float())
    if intermediate_hadamard:
        h = _boundary_intermediate_hadamard(
            gate, up, rotations, flat_ids, intermediate, device
        )
    else:
        h = _boundary_ordinary(gate, up, rotations, flat_ids, intermediate, device)
    y = torch.einsum("ri,rki->rk", h, w2_weights[flat_ids].float())
    out.index_add_(0, token_of, y * flat_w[:, None])
    return out
