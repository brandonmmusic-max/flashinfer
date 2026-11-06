"""Shared machinery for the unowned-cache-byte tests.

A request owns the K/V rows of its own pages at key positions ``[0, L)``.  Every
other byte of the pool (tail rows of a partly filled page, pages the request
never received, page-table padding, kernel scratch) may hold anything a previous
tenant left behind.  These helpers fill exactly those bytes with a poison
pattern and let a test compare the request's outputs bitwise against a run whose
unowned bytes are zero.

Patterns (byte fills of the whole plane):

``ZERO``   all-zero bytes; the comparison baseline.
``FF``     0xFF: NaN in E4M3, BF16 and FP32.
``7F``     0x7F: NaN in E4M3, a huge finite value in BF16/FP32.
``RANDn``  random bytes from seed ``n``.

Planes name the unowned bytes poisoned in one run: ``k``, ``v``, ``scratch``
and ``all``.  A plane that an attention variant does not have is skipped.
"""

from __future__ import annotations

import dataclasses
import math

import torch

# Byte templates, little endian.  See the module docstring for the meaning of
# each value in E4M3, BF16 and FP32.
_TEMPLATES = {
    "ZERO": (0x00,),
    "FF": (0xFF,),
    "7F": (0x7F,),
}


def pattern_bytes(pattern: str, nbytes: int, device) -> torch.Tensor:
    """``nbytes`` bytes of ``pattern`` (``RANDn`` uses seed ``n``)."""
    if pattern.startswith("RAND"):
        seed = int(pattern[4:])
        gen = torch.Generator(device=device).manual_seed(seed)
        return torch.randint(
            0, 256, (nbytes,), dtype=torch.uint8, device=device, generator=gen
        )
    template = torch.tensor(_TEMPLATES[pattern], dtype=torch.uint8, device=device)
    reps = (nbytes + template.numel() - 1) // template.numel()
    return template.repeat(reps)[:nbytes].contiguous()


def fill_like(tensor: torch.Tensor, pattern: str) -> torch.Tensor:
    """An output tensor shaped like ``tensor`` whose bytes are ``pattern``."""
    flat = pattern_bytes(pattern, tensor.numel() * tensor.element_size(), tensor.device)
    return flat.view(tensor.dtype).view(tensor.shape)


def overlay(
    clean: torch.Tensor, owned_rows: torch.Tensor, pattern: str
) -> torch.Tensor:
    """``clean`` where ``owned_rows[page, row]`` is true, ``pattern`` elsewhere."""
    out = fill_like(clean, pattern)
    out_b = out.view(torch.uint8).reshape(
        clean.shape[0], clean.shape[1], clean.shape[2], -1
    )
    clean_b = clean.view(torch.uint8).reshape_as(out_b)
    mask = owned_rows[:, :, None, None].expand_as(out_b)
    out_b[mask] = clean_b[mask]
    return out


@dataclasses.dataclass
class Comparison:
    equal: bool
    differing_bytes: int
    differing_elements: int
    max_abs_diff: float
    base_nonfinite: int
    other_nonfinite: int
    first_diff: tuple[int, ...] | None

    def describe(self) -> str:
        if self.equal:
            return "bitwise identical"
        return (
            f"{self.differing_elements} elements differ "
            f"(max |diff| {self.max_abs_diff:.6g}, "
            f"base non-finite {self.base_nonfinite}, "
            f"poison non-finite {self.other_nonfinite}, "
            f"first at {self.first_diff})"
        )


def compare(
    base: torch.Tensor, other: torch.Tensor, rows: torch.Tensor | None = None
) -> Comparison:
    """Bitwise comparison of two outputs, optionally restricted to ``rows``."""
    if rows is not None:
        base = base.index_select(0, rows)
        other = other.index_select(0, rows)
    base = base.contiguous()
    other = other.contiguous()
    if base.shape != other.shape or base.dtype != other.dtype:
        raise ValueError("outputs must match in shape and dtype")
    b = base.view(torch.uint8).reshape(-1)
    o = other.view(torch.uint8).reshape(-1)
    differing_bytes = int((b != o).sum().item())
    base_f = base.float()
    other_f = other.float()
    element_diff = (base_f.view(-1) != other_f.view(-1)).view(base.shape)
    differing_elements = int(element_diff.sum().item())
    first = None
    if differing_elements:
        flat = int(element_diff.view(-1).nonzero()[0].item())
        first = tuple(
            int(v) for v in torch.unravel_index(torch.tensor(flat), base.shape)
        )
    diff = (base_f - other_f).abs()
    finite = torch.isfinite(diff)
    max_abs = float(diff[finite].max().item()) if bool(finite.any()) else float("nan")
    return Comparison(
        equal=differing_bytes == 0,
        differing_bytes=differing_bytes,
        differing_elements=differing_elements,
        max_abs_diff=max_abs,
        base_nonfinite=int((~torch.isfinite(base_f)).sum().item()),
        other_nonfinite=int((~torch.isfinite(other_f)).sum().item()),
        first_diff=first,
    )


@dataclasses.dataclass
class PagedPool:
    """A paged K/V pool with per-request ownership.

    Page 0 is reserved and belongs to no request, so a page-table padding entry
    of ``0`` addresses unowned bytes.
    """

    k: torch.Tensor
    v: torch.Tensor
    owned: torch.Tensor
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    poison_pages: tuple[int, ...]
    k_descale: torch.Tensor | None
    v_descale: torch.Tensor | None

    def page_table_for(self, plane: str, pattern: str) -> torch.Tensor:
        """Page table for one plane.

        ``all`` repoints every padding entry at a poisoned page, so a read of a
        padding entry lands in poison instead of the real table.
        """
        if plane != "all" or pattern == "ZERO":
            return self.page_table
        table = self.page_table.clone()
        width = table.shape[1]
        padding = table == 0
        # Keep the real ids; only padding entries move.  Point them at the
        # poisoned pages so a read of a padding entry lands in poison.
        if self.poison_pages:
            ids = torch.tensor(
                self.poison_pages, dtype=torch.int32, device=table.device
            )
            count = int(padding.sum().item())
            if count:
                table[padding] = (
                    ids[:count]
                    if ids.numel() >= count
                    else ids.repeat((count + ids.numel() - 1) // ids.numel())[:count]
                )
        assert table.shape == (self.page_table.shape[0], width)
        return table


def build_paged_pool(
    *,
    lens: tuple[int, ...],
    page_size: int,
    kv_heads: int,
    dim_k: int,
    dim_v: int,
    kv_dtype: torch.dtype,
    q_lens: tuple[int, ...] | None = None,
    window: int = -1,
    extra_pages: int = 2,
    seed: int = 0,
) -> PagedPool:
    """Build a pool whose owned rows hold finite random K/V and whose unowned
    bytes are zero (the ``ZERO`` baseline).

    With a sliding window, rows before the oldest key any query row of the
    request attends are unowned too: the kernel must mask them exactly as it
    masks rows past the request's length.
    """
    device = "cuda"
    batch = len(lens)
    gen = torch.Generator(device=device).manual_seed(seed)
    pages_per_req = [math.ceil(length / page_size) for length in lens]
    total = sum(pages_per_req)
    num_pages = 1 + total + extra_pages
    width = max(1, max(pages_per_req))

    clean = torch.randn(
        (num_pages, page_size, kv_heads, dim_k + dim_v),
        generator=gen,
        device=device,
        dtype=torch.float32,
    )
    owned = torch.zeros((num_pages, page_size), dtype=torch.bool, device=device)

    k_descale = v_descale = None
    if kv_dtype == torch.float8_e4m3fn:
        k_scale = float(clean[..., :dim_k].abs().amax()) / 448.0
        v_scale = float(clean[..., dim_k:].abs().amax()) / 448.0
        clean[..., :dim_k] /= k_scale
        clean[..., dim_k:] /= v_scale
        clean = clean.to(torch.float8_e4m3fn)
        k_descale = torch.tensor([k_scale], dtype=torch.float32, device=device)
        v_descale = torch.tensor([v_scale], dtype=torch.float32, device=device)
    else:
        clean = clean.to(kv_dtype)

    table = torch.zeros((batch, width), dtype=torch.int32, device=device)
    cursor = 1
    for index, (length, count) in enumerate(zip(lens, pages_per_req, strict=True)):
        if count == 0:
            continue
        ids = torch.arange(cursor, cursor + count, dtype=torch.int32, device=device)
        table[index, :count] = ids
        q_len = 1 if q_lens is None else int(q_lens[index])
        first = 0 if window < 0 else max(0, length - q_len - window)
        for local, page in enumerate(range(cursor, cursor + count)):
            start = max(first, local * page_size)
            end = min(length, (local + 1) * page_size)
            if end > start:
                owned[page, start - local * page_size : end - local * page_size] = True
        cursor += count

    k = clean[..., :dim_k].contiguous()
    v = clean[..., dim_k:].contiguous()
    if q_lens is not None:
        for length, q_len in zip(lens, q_lens, strict=True):
            if q_len > length:
                raise ValueError("q_len exceeds cache length")
    return PagedPool(
        k=k,
        v=v,
        owned=owned,
        page_table=table,
        cache_seqlens=torch.tensor(lens, dtype=torch.int32, device=device),
        poison_pages=tuple(range(cursor, num_pages)),
        k_descale=k_descale,
        v_descale=v_descale,
    )


def active_rows(q_lens: tuple[int, ...]) -> torch.Tensor:
    """Row indices of a packed ragged batch that belong to a non-empty request."""
    rows: list[int] = []
    start = 0
    for length in q_lens:
        rows.extend(range(start, start + length))
        start += length
    return torch.tensor(rows, dtype=torch.long, device="cuda")


def set_owned_nan(
    tensor: torch.Tensor, page: int, row: int, columns: slice | None = None
) -> None:
    """Write the ``FF`` poison bytes into one owned cache row (positive control).

    ``0xFF`` is NaN in BF16 and the E4M3 NaN encoding in FP8.  Some FP8 kernels
    widen E4M3 NaN to a finite value, so the control asserts that the owned
    bytes are *observed*, not that the output is non-finite.
    """
    if tensor.dtype == torch.float8_e4m3fn:
        view = tensor.view(torch.uint8)
        view[page, row, :, columns if columns is not None else slice(None)] = 0xFF
    elif tensor.dtype == torch.bfloat16:
        view = tensor.view(torch.int16)
        view[page, row, :, columns if columns is not None else slice(None)] = -1
    else:
        raise TypeError(f"unsupported dtype {tensor.dtype}")
