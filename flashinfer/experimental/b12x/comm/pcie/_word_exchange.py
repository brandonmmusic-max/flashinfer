# SPDX-License-Identifier: Apache-2.0
"""Exact peer copies with independent sequence numbers for fixed word chunks.

Each chunk owns a counter and two receive banks. Chunk ownership does not
change when a launch groups more warps in a CTA or caps the grid. A packet
publishes its sequence and payload in the same aligned 64-bit store.

Calls sharing a channel execute serially on each rank. Two banks protect a
receiver from a peer starting the following call. Completing that call needs
a packet from every rank, so a peer cannot reuse a bank before its receiver
has consumed the preceding contents.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint8, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from ._dcp_cute_common import _membar_sys

CHUNK_WORDS = 64
MAX_CHUNKS = 64


@dsl_user_op
def _counter(address: Int64, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _publish_counter(address: Int64, value: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _peer_counter(address: Int64, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.relaxed.sys.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _publish_clear(address: Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip)],
        "st.relaxed.sys.global.u32 [$0], 1;",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _packet(address: Int64, *, loc=None, ip=None):
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u64 $0, [$1];",
            "=l,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _publish_packet(address: Int64, value: Uint64, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.relaxed.sys.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class WordExchange:
    def __init__(self, rank, world, row_words, threads, chunks):
        if (
            world not in (2, 4)
            or not 0 <= rank < world
            or row_words <= 0
            or threads not in (32, 64, 128, 256)
            or not 1 <= chunks <= MAX_CHUNKS
        ):
            raise ValueError(
                "word exchange requires TP2/TP4 and a bounded warp/chunk layout"
            )
        self.rank, self.world, self.row_words = rank, world, row_words
        self.warps, self.chunks = threads // 32, chunks

    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        output: cute.Pointer,
        p0: cute.Pointer,
        p1: cute.Pointer,
        p2: cute.Pointer,
        p3: cute.Pointer,
        stride: Int64,
        offset: Int64,
        counter_offset: Int64,
        rows: Int32,
        grid_blocks: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            source, output, p0, p1, p2, p3, stride, offset, counter_offset, rows
        ).launch(grid=(grid_blocks, 1, 1), block=(self.warps * 32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self, source, output, p0, p1, p2, p3, stride, offset, counter_offset, rows
    ):
        grid_blocks, _, _ = cute.arch.grid_dim()
        self.exchange(
            source,
            output,
            p0,
            p1,
            p2,
            p3,
            stride,
            offset,
            counter_offset,
            rows,
            Int32(grid_blocks),
        )

    @cute.jit
    def exchange(
        self,
        source,
        output,
        p0,
        p1,
        p2,
        p3,
        stride,
        offset,
        counter_offset,
        rows,
        grid_blocks,
        prefetched=None,
    ):
        data = (p0, p1, p2, p3)
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        tx, bx = Int32(tx), Int32(bx)
        warp, lane = tx // Int32(32), tx % Int32(32)
        total = rows * Int32(self.row_words)
        count = (total + Int32(CHUNK_WORDS - 1)) // Int32(CHUNK_WORDS)
        local = data[self.rank].toint()
        first = bx * Int32(self.warps)
        if cutlass.const_expr(prefetched is not None):
            next_previous, next_first, next_second = prefetched
        while first < count:
            chunk = first + warp
            previous = Uint32(0)
            counter = counter_offset + Int64(chunk) * Int64(4)
            if chunk < count:
                if cutlass.const_expr(prefetched is not None):
                    previous = next_previous
                else:
                    previous = _counter(local + counter)
            epoch = previous + Uint32(1)
            if previous == Uint32(0xFFFFFFFF):
                for bank in cutlass.range_constexpr(2):
                    index = lane
                    while index < Int32(self.world * CHUNK_WORDS):
                        address = local + offset + Int64(bank) * stride
                        address += Int64(
                            chunk * Int32(self.world * CHUNK_WORDS) + index
                        ) * Int64(8)
                        _publish_packet(address, Uint64(0))
                        index += Int32(32)
                _membar_sys()
                cute.arch.sync_warp()
                if lane == Int32(0):
                    _publish_clear(local + counter)
                    # No peer can leave sequence one until this warp sends its
                    # sequence-two packets. Seeing one therefore acknowledges
                    # cleared banks without racing the next exchange.
                    for peer in cutlass.range_constexpr(self.world):
                        if cutlass.const_expr(peer != self.rank):
                            ready = _peer_counter(data[peer].toint() + counter)
                            while ready != Uint32(1):
                                ready = _peer_counter(data[peer].toint() + counter)
                cute.arch.sync_warp()
                epoch = Uint32(2)
            bank = Int64(epoch & Uint32(1)) * stride + offset
            for item in cutlass.range_constexpr(2):
                within = lane + Int32(item * 32)
                word = chunk * Int32(CHUNK_WORDS) + within
                if word < total:
                    if cutlass.const_expr(prefetched is not None):
                        value = next_first if item == 0 else next_second
                    else:
                        value = Uint32(cute.arch.load(source + Int64(word), Uint32))
                    packet = (Uint64(epoch) << Uint64(32)) | Uint64(value)
                    address_offset = Int64(
                        (chunk * Int32(self.world) + Int32(self.rank))
                        * Int32(CHUNK_WORDS)
                        + within
                    ) * Int64(8)
                    for peer in cutlass.range_constexpr(self.world):
                        if cutlass.const_expr(peer != self.rank):
                            _publish_packet(
                                data[peer].toint() + bank + address_offset, packet
                            )
                    destination = (
                        word // Int32(self.row_words) * Int32(self.world)
                        + Int32(self.rank)
                    ) * Int32(self.row_words)
                    destination += word % Int32(self.row_words)
                    cute.arch.store(output + Int64(destination), value)
            for peer in cutlass.range_constexpr(self.world):
                if cutlass.const_expr(peer != self.rank):
                    for item in cutlass.range_constexpr(2):
                        within = lane + Int32(item * 32)
                        word = chunk * Int32(CHUNK_WORDS) + within
                        if word < total:
                            address = (
                                local
                                + bank
                                + Int64(
                                    (chunk * Int32(self.world) + Int32(peer))
                                    * Int32(CHUNK_WORDS)
                                    + within
                                )
                                * Int64(8)
                            )
                            packet = _packet(address)
                            while Uint32(packet >> Uint64(32)) != epoch:
                                packet = _packet(address)
                            destination = (
                                word // Int32(self.row_words) * Int32(self.world)
                                + Int32(peer)
                            ) * Int32(self.row_words)
                            destination += word % Int32(self.row_words)
                            cute.arch.store(output + Int64(destination), Uint32(packet))
            cute.arch.sync_warp()
            if lane == Int32(0) and chunk < count:
                _publish_counter(
                    local + counter_offset + Int64(chunk) * Int64(4), epoch
                )
            first += Int32(grid_blocks) * Int32(self.warps)
            if cutlass.const_expr(prefetched is not None):
                chunk = first + warp
                word = chunk * Int32(CHUNK_WORDS) + lane
                if chunk < count:
                    next_previous = _counter(
                        local + counter_offset + Int64(chunk) * Int64(4)
                    )
                if word < total:
                    next_first = Uint32(cute.arch.load(source + Int64(word), Uint32))
                if word + Int32(32) < total:
                    next_second = Uint32(
                        cute.arch.load(source + Int64(word + Int32(32)), Uint32)
                    )


@program_cache
def compile_word_exchange(rank, world, row_words, threads, blocks, chunks):
    if not 1 <= blocks <= MAX_CHUNKS:
        raise ValueError("word exchange grid limit must be between one and 64")
    operation = WordExchange(rank, world, row_words, threads, chunks)

    def pointer(dtype, address=16, alignment=16):
        return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=alignment)

    executable = compile_kernel(
        operation,
        pointer(Uint32, alignment=4),
        pointer(Uint32, alignment=4),
        *(pointer(Uint8) for _ in range(4)),
        Int64(256),
        Int64(256),
        Int64(256),
        Int32(1),
        Int32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.word_exchange", 1, (rank, world, row_words, threads, chunks)
        ),
    )

    def launch(source, output, data, stride, offset, counter_offset, rows):
        if rows <= 0 or rows * row_words > chunks * CHUNK_WORDS:
            raise ValueError("word exchange exceeds its declared chunk capacity")
        if source % 4 or output % 4:
            raise ValueError("word exchange requires aligned 32-bit payload words")
        padded_data = (*data, *data)[:4]
        grid = min(
            blocks,
            (rows * row_words + CHUNK_WORDS * (threads // 32) - 1)
            // (CHUNK_WORDS * (threads // 32)),
        )
        executable(
            pointer(Uint32, source, 4),
            pointer(Uint32, output, 4),
            *(pointer(Uint8, value) for value in padded_data),
            Int64(stride),
            Int64(offset),
            Int64(counter_offset),
            Int32(rows),
            Int32(grid),
            current_cuda_stream(),
        )

    return attach_programs(launch, executable)
