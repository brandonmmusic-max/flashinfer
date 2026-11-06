"""Column gather with device-selected protocol and active CTA count.

The low 16 selector bits hold the active CTA count; bit 16 selects word
exchange instead of barrier gather. All ranks execute the same profile and
serialize calls sharing the DCP channel. Protocols retain independent epochs
and disjoint receive storage, including when captured calls alternate modes.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from ._dcp_a2a_cute import _AllGatherHeadsLaunch
from ._word_exchange import WordExchange, _counter


@dsl_user_op
def _prefetch(selector, counter, source, word, total, tid, *, loc=None, ip=None):
    """Keep bounds checks predicated before protocol dispatch in wide CTAs."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [
            value.ir_value(loc=loc, ip=ip)
            for value in (selector, counter, source, word, total, tid)
        ],
        """{
        .reg .pred warp, valid, fetch;
        .reg .b32 base, second;
        mov.b32 $1, 0;
        mov.b32 $2, 0;
        mov.b32 $3, 0;
        setp.lt.u32 warp, $9, 32;
        and.b32 base, $7, -64;
        setp.lt.s32 valid, base, $8;
        and.pred fetch, warp, valid;
        @fetch ld.volatile.global.u32 $1, [$5];
        ld.volatile.global.u32 $0, [$4];
        setp.lt.s32 valid, $7, $8;
        and.pred fetch, warp, valid;
        @fetch ld.global.u32 $2, [$6];
        add.s32 second, $7, 32;
        setp.lt.s32 valid, second, $8;
        and.pred fetch, warp, valid;
        @fetch ld.global.u32 $3, [$6+128];
        }""",
        "=r,=r,=r,=r,l,l,l,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip))
        for index in range(4)
    )


class SelectedGather(_AllGatherHeadsLaunch):
    def __init__(self, world, rank, row_words, threads, blocks, chunks, transport):
        if world not in (2, 4, 8, 9, 10, 12, 16) or not 0 <= rank < world:
            raise ValueError("gather requires a supported TP size and rank")
        if threads not in (32, 64, 128, 256, 512):
            raise ValueError("gather requires 32, 64, 128, 256, or 512 threads")
        if transport not in ("barrier", "word_exchange", "joint"):
            raise ValueError(
                "gather transport must be barrier, word_exchange, or joint"
            )
        if row_words <= 0 or row_words % 4 or not 1 <= blocks <= 64:
            raise ValueError("gather requires aligned rows and one to 64 CTAs")
        super().__init__(world, rank, threads, True)
        self.row_words, self.blocks, self.transport = row_words, blocks, transport
        if transport != "barrier":
            # Packet synchronization is warp-local; its grouping is independent
            # of the barrier protocol's CTA width.
            self.words = WordExchange(
                rank, world, row_words, 32 if transport == "joint" else threads, chunks
            )

    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        output: cute.Pointer,
        selector: cute.Pointer,
        stages: tuple,
        signals: tuple,
        rows: Int32,
        stride: Int64,
        offset: Int64,
        counter: Int64,
        stream: cuda.CUstream,
    ):
        self.selected(
            source, output, selector, stages, signals, rows, stride, offset, counter
        ).launch(grid=(self.blocks, 1, 1), block=(self._threads, 1, 1), stream=stream)

    @cute.kernel
    def selected(
        self, source, output, selector, stages, signals, rows, stride, offset, counter
    ):
        block, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        previous, first_word, second_word = Uint32(0), Uint32(0), Uint32(0)
        profile = Uint32(0)
        if cutlass.const_expr(self.transport == "joint"):
            if cutlass.const_expr(self._threads == 32):
                if Int32(tid) < Int32(32):
                    word = Int32(block) * Int32(64) + Int32(tid)
                    total = rows * Int32(self.row_words)
                    if Int32(block) * Int32(64) < total:
                        previous = _counter(
                            stages[self._rank].toint()
                            + counter
                            + Int64(block) * Int64(4)
                        )
                    profile = _counter(selector.toint())
                    if word < total:
                        first_word = Uint32(
                            cute.arch.load(source + Int64(word), Uint32)
                        )
                    if word + Int32(32) < total:
                        second_word = Uint32(
                            cute.arch.load(source + Int64(word + Int32(32)), Uint32)
                        )
                else:
                    profile = _counter(selector.toint())
            else:
                word = Int32(block) * Int32(64) + Int32(tid)
                profile, previous, first_word, second_word = _prefetch(
                    selector.toint(),
                    stages[self._rank].toint() + counter + Int64(block) * Int64(4),
                    source.toint() + Int64(word) * Int64(4),
                    word,
                    rows * Int32(self.row_words),
                    Int32(tid),
                )
        else:
            profile = Uint32(cute.arch.load(selector, Uint32))
        active = Int32(profile & Uint32(0xFFFF))
        if Int32(block) < active:
            if cutlass.const_expr(self.transport == "word_exchange"):
                self.words.exchange(
                    source,
                    output,
                    stages[0],
                    stages[1],
                    stages[2],
                    stages[3],
                    stride,
                    offset,
                    counter,
                    rows,
                    active,
                )
            elif cutlass.const_expr(self.transport == "joint"):
                if profile & Uint32(0x10000):
                    tid, _, _ = cute.arch.thread_idx()
                    if Int32(tid) < Int32(32):
                        self.words.exchange(
                            source,
                            output,
                            stages[0],
                            stages[1],
                            stages[2],
                            stages[3],
                            stride,
                            offset,
                            counter,
                            rows,
                            active,
                            (previous, first_word, second_word),
                        )
                else:
                    self.gather(
                        source,
                        output,
                        stages,
                        signals,
                        rows,
                        Int32(1),
                        Int32(self.row_words // 4),
                        Int32(stride // Int64(256)),
                        active,
                    )
            else:
                self.gather(
                    source,
                    output,
                    stages,
                    signals,
                    rows,
                    Int32(1),
                    Int32(self.row_words // 4),
                    Int32(stride // Int64(256)),
                    active,
                )


@program_cache
def compile_selected_gather(world, rank, row_words, threads, blocks, chunks, transport):
    operation = SelectedGather(
        world, rank, row_words, threads, blocks, chunks, transport
    )

    def pointer(dtype, address=16, alignment=16):
        return make_ptr(
            dtype, address, mem_space=cute.AddressSpace.gmem, assumed_align=alignment
        )

    raw = compile_kernel(
        operation,
        pointer(Uint32),
        pointer(Uint32),
        pointer(Uint32, alignment=4),
        tuple(pointer(Uint8) for _ in range(16)),
        tuple(pointer(Uint32, alignment=4) for _ in range(16)),
        Int32(1),
        Int64(256),
        Int64(0),
        Int64(512),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.selected_gather",
            11,
            (world, rank, row_words, threads, blocks, chunks, transport),
        ),
    )

    def run(source, output, selector, stages, signals, rows, stride, offset, counter):
        if len(stages) != world or len(signals) != world:
            raise ValueError("gather requires one staging and signal address per rank")
        if rows <= 0 or (transport != "barrier" and rows * row_words > chunks * 64):
            raise ValueError("gather exceeds its declared word exchange capacity")
        if source % 16 or output % 16:
            raise ValueError("gather requires 16-byte aligned input and output")
        if (
            source < output + rows * row_words * world * 4
            and output < source + rows * row_words * 4
        ):
            raise ValueError("gather requires disjoint input and output")
        raw(
            pointer(Uint32, source),
            pointer(Uint32, output),
            pointer(Uint32, selector, 4),
            tuple(pointer(Uint8, address) for address in (tuple(stages) * 16)[:16]),
            tuple(
                pointer(Uint32, address, 4) for address in (tuple(signals) * 16)[:16]
            ),
            Int32(rows),
            Int64(stride),
            Int64(offset),
            Int64(counter),
            current_cuda_stream(),
        )

    return attach_programs(run, raw)
