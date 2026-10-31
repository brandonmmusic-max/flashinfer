"""Shared caller-owned scratch plan helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, kw_only=True)
class ScratchBufferSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    #: Byte alignment of the buffer's start. A kernel that hands the buffer to a
    #: vector load or a tensor-core descriptor with an assumed alignment needs
    #: the caller's pointer to carry it; 1 accepts any address.
    alignment: int = 1

    @property
    def nbytes(self) -> int:
        numel = 1
        for dim in self.shape:
            numel *= int(dim)
        return numel * self.dtype.itemsize


def scratch_buffer_spec(
    name: str,
    *,
    nbytes: int,
    device: torch.device,
    alignment: int = 1,
) -> ScratchBufferSpec:
    if alignment < 1 or alignment & (alignment - 1):
        raise ValueError(
            f"{name} scratch alignment must be a positive power of two; got {alignment}"
        )
    return ScratchBufferSpec(
        name=name,
        shape=(max(int(nbytes), 1),),
        dtype=torch.uint8,
        device=device,
        alignment=int(alignment),
    )


def scratch_tensor(
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    specs: tuple[ScratchBufferSpec, ...],
    *,
    owner: str,
) -> torch.Tensor:
    if len(specs) != 1:
        raise RuntimeError(
            f"{owner} scratch plans currently expect exactly one scratch buffer"
        )
    spec = specs[0]
    if isinstance(scratch, torch.Tensor):
        tensor = scratch
    elif isinstance(scratch, Mapping):
        if spec.name not in scratch:
            raise KeyError(f"scratch mapping is missing {spec.name!r}")
        tensor = scratch[spec.name]
    else:
        if len(scratch) != 1:
            raise ValueError(
                f"scratch sequence must contain exactly one tensor, got {len(scratch)}"
            )
        tensor = scratch[0]
    if tensor.dtype != spec.dtype:
        raise TypeError(
            f"{spec.name} scratch must have dtype {spec.dtype}, got {tensor.dtype}"
        )
    if tensor.device != spec.device:
        raise ValueError(
            f"{spec.name} scratch device {tensor.device} does not match {spec.device}"
        )
    if int(tensor.numel()) < int(spec.shape[0]):
        raise ValueError(
            f"{spec.name} scratch has {int(tensor.numel())} bytes, requires {int(spec.shape[0])}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{spec.name} scratch must be contiguous")
    if spec.alignment > 1 and tensor.data_ptr() % spec.alignment:
        raise ValueError(
            f"{spec.name} scratch must start at a {spec.alignment}-byte aligned "
            f"address; got {tensor.data_ptr()} (offset {tensor.data_ptr() % spec.alignment})"
        )
    return tensor.reshape(-1)


def require_disjoint(output: torch.Tensor, inputs, *, owner: str) -> None:
    """Reject overlapping byte ranges while allowing disjoint views of one pool."""
    if not output.numel():
        return

    def interval(value):
        begin = value.data_ptr()
        span = 1 + sum(
            (size - 1) * stride for size, stride in zip(value.shape, value.stride())
        )
        return begin, begin + span * value.element_size()

    begin, end = interval(output)
    for value in inputs:
        if value is None or not value.numel() or value.device != output.device:
            continue
        other_begin, other_end = interval(value)
        if begin < other_end and other_begin < end:
            raise ValueError(f"{owner} must not overlap input or workspace storage")


__all__ = [
    "ScratchBufferSpec",
    "scratch_buffer_spec",
    "scratch_tensor",
    "require_disjoint",
]
