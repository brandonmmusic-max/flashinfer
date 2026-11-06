"""Fixtures that the consecutive startup races of one device share."""

from math import prod
from types import SimpleNamespace
import weakref

import torch

_ALIGN = 256


def _aligned(nbytes):
    return (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN


class StartupFixtures:
    """Fixtures shared by serial races in one caller-owned storage instance.

    Races replay one at a time and drain before the next starts, so their
    scratch (source, output and mismatch mask) aliases one arena sized for the
    largest race. References are generated row by row and independently of the
    row count, so a race reads the leading rows of the largest race with the
    same row layout. Growth rebinds every live race until one is captured;
    afterwards a larger race gets fixtures of its own.
    """

    def __init__(self, device):
        self.device = device
        self.scratch = torch.empty(0, dtype=torch.uint8, device=device)
        self.banks = {}
        self.users = weakref.WeakSet()
        self.frozen = False

    def attach(self, user):
        key, rows, build = user.bank_request()
        need = user.scratch_bytes()
        bank = self.banks.get(key)
        grow_scratch = need > self.scratch.numel()
        grow_bank = bank is None or bank.rows < rows
        if self.frozen and (grow_scratch or grow_bank):
            # Captured races keep reading their storage.
            user.bind(
                torch.empty(need, dtype=torch.uint8, device=self.device), build(rows)
            )
            return
        if grow_scratch:
            self.scratch = torch.empty(need, dtype=torch.uint8, device=self.device)
        if grow_bank:
            self.banks[key] = build(rows)
        self.users.add(user)
        for live in list(self.users):
            live.bind(self.scratch, self.banks[live.bank_request()[0]])


def _reduction_bank(world, rank, rows, row_shape, dtype, device):
    # Exactly representable partial sums isolate protocol and binding errors.
    values = torch.stack(
        [
            torch.randint(
                -16,
                17,
                (rows, *row_shape),
                generator=torch.Generator().manual_seed(81721 + peer),
            ).float()
            / 32
            for peer in range(world)
        ]
    )
    cases = (values, torch.zeros_like(values), -values)
    return SimpleNamespace(
        rows=rows,
        seeds=tuple(case[rank].to(device=device, dtype=dtype) for case in cases),
        expected=tuple(
            case.sum(dim=0).to(device=device, dtype=dtype) for case in cases
        ),
    )


def _gather_bank(world, rank, rows, row_bytes, device):
    # Byte fixtures exercise FP8 payloads and FP32 bit views without arithmetic conversion.
    offsets = torch.arange(rows * row_bytes, device=device)
    values = torch.stack(
        [((offsets + peer * 7919) % 251).to(torch.uint8) for peer in range(world)]
    ).view(world, rows, row_bytes)
    cases = (values, torch.zeros_like(values), 255 - values)
    return SimpleNamespace(
        rows=rows,
        seeds=tuple(case[rank].reshape(-1).clone() for case in cases),
        expected=tuple(
            case.permute(1, 0, 2).contiguous().reshape(-1) for case in cases
        ),
    )


class StartupReductionInputs:
    """Plain all-reduce fixtures bound to a device's shared startup storage."""

    def __init__(self, fixtures, state, *, inplace, dtype, repetitions):
        query = state.query
        self.shape = tuple(query.setup["shape"])
        self.dtype, self.inplace, self.repetitions = dtype, inplace, repetitions
        self.world, self.rank, self.device = (
            query.world_size,
            query.rank,
            state.runtime.device,
        )
        self.residual = None
        fixtures.attach(self)

    def bank_request(self):
        world, rank, rows, row_shape = (
            self.world,
            self.rank,
            self.shape[0],
            self.shape[1:],
        )
        dtype, device = self.dtype, self.device
        return (
            ("reduction", world, rank, dtype, row_shape),
            rows,
            lambda capacity: _reduction_bank(
                world, rank, capacity, row_shape, dtype, device
            ),
        )

    def scratch_bytes(self):
        nbytes = prod(self.shape) * self.dtype.itemsize
        return _aligned(nbytes) * (1 if self.inplace else 2) + _aligned(
            prod(self.shape)
        )

    def bind(self, scratch, bank):
        elements = prod(self.shape)
        nbytes = elements * self.dtype.itemsize
        self.source = scratch[:nbytes].view(self.dtype).view(self.shape)
        offset = _aligned(nbytes)
        if self.inplace:
            self.out = self.source
        else:
            self.out = (
                scratch[offset : offset + nbytes].view(self.dtype).view(self.shape)
            )
            offset += _aligned(nbytes)
        self.checked = self.out
        self.mismatch = (
            scratch[offset : offset + elements].view(torch.bool).view(self.shape)
        )
        rows = self.shape[0]
        self.reference = SimpleNamespace(
            seeds=tuple(seed[:rows] for seed in bank.seeds),
            expected=tuple(expected[:rows] for expected in bank.expected),
        )

    def produce(self, index):
        self.source.copy_(self.reference.seeds[index])


class StartupGatherInputs:
    """All-gather fixtures bound to a device's shared startup storage."""

    def __init__(self, fixtures, state):
        q = state.query
        self.shape, self.out_shape = tuple(q.call["shape"]), tuple(q.call["out_shape"])
        self.dtype = getattr(torch, q.call["dtype"].removeprefix("torch."))
        self.world, self.rank, self.device = q.world_size, q.rank, state.runtime.device
        self.row_bytes = prod(self.shape[1:]) * self.dtype.itemsize
        self.repetitions = 1
        fixtures.attach(self)

    def bank_request(self):
        world, rank, row_bytes, device = (
            self.world,
            self.rank,
            self.row_bytes,
            self.device,
        )
        return (
            ("gather", world, rank, row_bytes),
            self.shape[0],
            lambda capacity: _gather_bank(world, rank, capacity, row_bytes, device),
        )

    def scratch_bytes(self):
        source = self.shape[0] * self.row_bytes
        return _aligned(source) + 2 * _aligned(self.world * source)

    def bind(self, scratch, bank):
        source = self.shape[0] * self.row_bytes
        out = self.world * source
        self.source = scratch[:source].view(self.dtype).view(self.shape)
        offset = _aligned(source)
        self.out = scratch[offset : offset + out].view(self.dtype).view(self.out_shape)
        offset += _aligned(out)
        self.checked = self.out.view(torch.uint8).reshape(-1)
        self.mismatch = scratch[offset : offset + out].view(torch.bool)
        self.reference = SimpleNamespace(
            seeds=tuple(
                seed[:source].view(self.dtype).view(self.shape) for seed in bank.seeds
            ),
            expected=tuple(expected[:out] for expected in bank.expected),
        )

    def produce(self, index):
        self.source.copy_(self.reference.seeds[index])
