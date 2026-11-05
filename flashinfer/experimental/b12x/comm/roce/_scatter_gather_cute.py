"""CuTe DSL kernels for the RoCE pipelined reduce-scatter and shard gather.

The two halves of the pipelined two-shot all-reduce (``_twoshot_cute``) as
separate operations over the same ``pipe_send``/``pipe_recv`` buffers, chunk
flags and counters.  A message of ``P`` packs is split into shards exactly as the
two-shot splits it: rank ``j`` owns packs ``[j P / world, (j + 1) P / world)``,
and each shard is split into the same number of chunks.

``reduce_scatter`` (ctrl kind 3) takes the whole message as input and leaves
this rank's reduced shard in an output of shard size:

1. doorbell: block 0 records the op and rings ``seq``;
2. stage: blocks copy chunk ``c`` of every peer's shard into ``pipe_send``; the
   last block to finish a chunk publishes ``ready_rs[c] = seq`` and the proxy
   writes that chunk to its owner;
3. reduce: per chunk of our shard, wait for every peer's stripe flags and sum
   the local input and the peers' chunks in fixed rank order into the output;
4. epoch: the last block advances the epoch once the proxy released ``pipe_send``.

``gather_shards`` (ctrl kind 4) takes this rank's shard as input and leaves the
whole message in an output of message size:

1. doorbell;
2. stage: per chunk of our shard, copy it into ``pipe_send`` and into our shard
   of the output; the last block to finish publishes ``ready_ag[c] = seq`` and
   the proxy sends the chunk to every peer;
3. gather: per chunk, wait for every peer's chunk and copy it into the output;
4. epoch, as above.

A reduce-scatter followed by a shard gather of its output moves the same bytes
as one two-shot all-reduce and yields the same bits on every rank, since each
shard is reduced by its owner alone with FP32 accumulation in rank order.  The
grid is the runtime's fixed two-shot grid, so the free-running chunk counters
stay consistent across every pipelined kind.
"""

from __future__ import annotations

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
import torch

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr

from ._cute_intrinsics import (
    atomic_add_relaxed_gpu_u32,
    fence_sc_gpu,
    fence_sc_sys,
    ld_global_v4_u32,
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_u32,
    ld_relaxed_sys_v4_u32,
    spin_until_eq_acquire_sys,
    st_global_v4_u32,
    st_release_gpu_u32,
    st_relaxed_sys_u32,
)
from ._oneshot_cute import _DTYPE_NAMES, PACK_BYTES
from ._twoshot_cute import (
    _CTRL_PIPE_DONE,
    _CTRL_READY_AG,
    _CTRL_READY_RS,
    _CTRL_SLOT_CHUNK_PACKS,
    _CTRL_SLOT_CHUNKS,
    _CTRL_SLOT_KIND,
    _RoceTwoshotLaunch,
)

# Control-record op kinds (see _roce_proxy.c).
KIND_REDUCE_SCATTER = 3
KIND_GATHER_SHARDS = 4
OPERATIONS = ("reduce_scatter", "gather_shards")
_PREPARED_LAUNCHERS: set[tuple[object, ...]] = set()


class _RoceScatterGatherLaunch(_RoceTwoshotLaunch):
    def __init__(
        self,
        operation: str,
        dtype_name: str,
        world_size: int,
        rank: int,
        threads: int,
        flag_stride: int,
        hca_count: int,
        max_chunks: int,
    ) -> None:
        """Bind one specialization: operation, dtype, world size, rank, and layout constants."""
        if operation not in OPERATIONS:
            raise ValueError(f"unsupported RoCE pipelined operation {operation!r}")
        super().__init__(
            dtype_name, world_size, rank, threads, flag_stride, hca_count, max_chunks
        )
        self._operation = operation

    @cute.jit
    def _copy_pack(self, source: Int64, destination: Int64) -> None:
        words = ld_global_v4_u32(source)
        st_global_v4_u32(destination, words[0], words[1], words[2], words[3])

    @cute.kernel
    def kernel(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        size_packs: Int32,
        nbytes: Int32,
        chunk_packs: Int32,
        chunks: Int32,
        send_base: Int64,
        recv_base: Int64,
        flag_base: Int64,
        shard_cap: Int64,
        ctrl_base: Int64,
        epoch_ptr: Int64,
        counter_base: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
    ) -> None:
        """Device kernel: doorbell, then the stage/reduce or stage/gather phases, then the epoch.

        ``size_packs`` and ``nbytes`` describe the whole message; the shard-sized
        operand (the reduce-scatter's output, the gather's input) holds packs
        ``[own_first, own_end)`` of it at offset zero.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        input_base = Int64(input_ptr.toint())
        output_base = Int64(output_ptr.toint())
        epoch = ld_relaxed_gpu_u32(epoch_ptr)
        seq = epoch + Uint32(1)
        slot = Int64(seq & Uint32(1))
        index = Int32(bidx) * Int32(self._threads) + Int32(tidx)
        stride = Int32(gdim) * Int32(self._threads)
        stage_counters = counter_base
        reduce_counters = counter_base + Int64(4 * self._max_chunks)
        tail_counter = counter_base + Int64(8 * self._max_chunks)
        own_first = self._shard_start(Int32(self._rank), size_packs)
        own_end = self._shard_start(Int32(self._rank + 1), size_packs)

        poisoned = ld_relaxed_gpu_u32(poison_ptr)
        if poisoned == Uint32(0):
            # 1. the op record, then the doorbell; the proxy then follows the ready words
            if Int32(bidx) == Int32(0):
                if Int32(tidx) == Int32(0):
                    st_relaxed_sys_u32(ctrl_base + Int64(4), Uint32(nbytes))
                    st_relaxed_sys_u32(
                        ctrl_base + Int64(16) + slot * Int64(4), Uint32(nbytes)
                    )
                    if cutlass.const_expr(self._operation == "reduce_scatter"):
                        st_relaxed_sys_u32(
                            ctrl_base + Int64(4 * _CTRL_SLOT_KIND) + slot * Int64(4),
                            Uint32(KIND_REDUCE_SCATTER),
                        )
                    else:
                        st_relaxed_sys_u32(
                            ctrl_base + Int64(4 * _CTRL_SLOT_KIND) + slot * Int64(4),
                            Uint32(KIND_GATHER_SHARDS),
                        )
                    st_relaxed_sys_u32(
                        ctrl_base + Int64(4 * _CTRL_SLOT_CHUNK_PACKS) + slot * Int64(4),
                        Uint32(chunk_packs),
                    )
                    st_relaxed_sys_u32(
                        ctrl_base + Int64(4 * _CTRL_SLOT_CHUNKS) + slot * Int64(4),
                        Uint32(chunks),
                    )
                    fence_sc_sys()
                    st_relaxed_sys_u32(ctrl_base, seq)

            if cutlass.const_expr(self._operation == "reduce_scatter"):
                # 2. stage every peer's shard, chunk by chunk
                for chunk in cutlass.range(chunks, unroll=1):
                    for shard in cutlass.range_constexpr(self._world_size):
                        if cutlass.const_expr(shard != self._rank):
                            first = self._shard_start(Int32(shard), size_packs)
                            end = self._shard_start(Int32(shard + 1), size_packs)
                            lo = first + chunk * chunk_packs
                            hi = cutlass.min(end, lo + chunk_packs)
                            stage_index = lo + index
                            while stage_index < hi:
                                offset = Int64(stage_index) * Int64(PACK_BYTES)
                                self._copy_pack(input_base + offset, send_base + offset)
                                stage_index += stride
                    cute.arch.sync_threads()
                    self._arrive(
                        stage_counters + Int64(4) * Int64(chunk),
                        ctrl_base + Int64(4 * _CTRL_READY_RS) + Int64(4) * Int64(chunk),
                        seq,
                        Int32(gdim),
                        ctrl_base,
                    )

                # 3. reduce our shard in fixed rank order as peers' chunks arrive
                for chunk in cutlass.range(chunks, unroll=1):
                    if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                        self._wait_chunk(
                            flag_base, 0, chunk, seq, ctrl_base, poison_ptr, spin_limit
                        )
                    cute.arch.sync_threads()
                    if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                        lo = own_first + chunk * chunk_packs
                        hi = cutlass.min(own_end, lo + chunk_packs)
                        reduce_index = lo + index
                        while reduce_index < hi:
                            accumulator = cute.make_rmem_tensor(
                                (self._pack_elems,), cutlass.Float32
                            )
                            offset = Int64(reduce_index) * Int64(PACK_BYTES)
                            shard_offset = Int64(reduce_index - own_first) * Int64(
                                PACK_BYTES
                            )
                            for source in cutlass.range_constexpr(self._world_size):
                                if cutlass.const_expr(source == self._rank):
                                    words = ld_global_v4_u32(input_base + offset)
                                else:
                                    words = ld_relaxed_sys_v4_u32(
                                        recv_base
                                        + Int64(source * 2) * shard_cap
                                        + shard_offset
                                    )
                                self._accumulate_words(accumulator, words, source == 0)
                            self._store_accumulator(
                                output_base + shard_offset, accumulator
                            )
                            reduce_index += stride
                    cute.arch.sync_threads()
            else:
                # 2. stage our shard into pipe_send and the output, chunk by chunk
                for chunk in cutlass.range(chunks, unroll=1):
                    lo = own_first + chunk * chunk_packs
                    hi = cutlass.min(own_end, lo + chunk_packs)
                    stage_index = lo + index
                    while stage_index < hi:
                        offset = Int64(stage_index) * Int64(PACK_BYTES)
                        shard_offset = Int64(stage_index - own_first) * Int64(
                            PACK_BYTES
                        )
                        words = ld_global_v4_u32(input_base + shard_offset)
                        st_global_v4_u32(
                            send_base + offset, words[0], words[1], words[2], words[3]
                        )
                        st_global_v4_u32(
                            output_base + offset, words[0], words[1], words[2], words[3]
                        )
                        stage_index += stride
                    cute.arch.sync_threads()
                    self._arrive(
                        reduce_counters + Int64(4) * Int64(chunk),
                        ctrl_base + Int64(4 * _CTRL_READY_AG) + Int64(4) * Int64(chunk),
                        seq,
                        Int32(gdim),
                        ctrl_base,
                    )

                # 3. gather every peer's shard
                for chunk in cutlass.range(chunks, unroll=1):
                    if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                        self._wait_chunk(
                            flag_base, 1, chunk, seq, ctrl_base, poison_ptr, spin_limit
                        )
                    cute.arch.sync_threads()
                    if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                        for shard in cutlass.range_constexpr(self._world_size):
                            if cutlass.const_expr(shard != self._rank):
                                first = self._shard_start(Int32(shard), size_packs)
                                end = self._shard_start(Int32(shard + 1), size_packs)
                                lo = first + chunk * chunk_packs
                                hi = cutlass.min(end, lo + chunk_packs)
                                gather_index = lo + index
                                while gather_index < hi:
                                    words = ld_relaxed_sys_v4_u32(
                                        recv_base
                                        + Int64(shard * 2 + 1) * shard_cap
                                        + Int64(gather_index - first)
                                        * Int64(PACK_BYTES)
                                    )
                                    st_global_v4_u32(
                                        output_base
                                        + Int64(gather_index) * Int64(PACK_BYTES),
                                        words[0],
                                        words[1],
                                        words[2],
                                        words[3],
                                    )
                                    gather_index += stride

        # 4. the last block to finish publishes the next epoch, once the proxy
        # released pipe_send: the next pipelined op stages over it.
        fence_sc_gpu()
        cute.arch.sync_threads()
        if Int32(tidx) == Int32(0):
            prior = atomic_add_relaxed_gpu_u32(tail_counter, Uint32(1))
            if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                fence_sc_gpu()
                if ld_relaxed_sys_u32(ctrl_base + Int64(8)) == Uint32(0):
                    released = spin_until_eq_acquire_sys(
                        ctrl_base + Int64(4 * _CTRL_PIPE_DONE), seq, spin_limit
                    )
                    if released != Uint32(0):
                        st_relaxed_sys_u32(ctrl_base + Int64(12), Uint32(self._rank))
                        st_relaxed_sys_u32(ctrl_base + Int64(8), seq)
                        st_release_gpu_u32(poison_ptr, seq)
                    else:
                        st_release_gpu_u32(epoch_ptr, seq)


def _dummy(dtype, alignment: int):
    """A CUDA pointer of ``dtype`` used to trace launcher argument types."""
    return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=alignment)


def _process_key(*key) -> tuple[object, ...]:
    return tuple(key)


def is_launcher_prepared(*key) -> bool:
    """Return whether this exact process-local launcher is already loaded."""
    return _process_key(*key) in _PREPARED_LAUNCHERS


@program_cache
def get_launcher(
    operation: str,
    dtype_name: str | torch.dtype,
    world_size: int,
    rank: int,
    threads: int,
    flag_stride: int,
    hca_count: int,
    max_chunks: int,
    device_index: int,
) -> Callable[..., None]:
    """Compile the reduce-scatter or shard-gather launcher for ``key`` once and return it."""
    if isinstance(dtype_name, torch.dtype):
        try:
            dtype_name = _DTYPE_NAMES[dtype_name]
        except KeyError:
            raise ValueError(
                f"unsupported RoCE pipelined dtype {dtype_name!r}"
            ) from None
    process_key = _process_key(
        str(operation),
        str(dtype_name),
        int(world_size),
        int(rank),
        int(threads),
        int(flag_stride),
        int(hca_count),
        int(max_chunks),
        int(device_index),
    )
    launch = _RoceScatterGatherLaunch(
        operation,
        dtype_name,
        world_size,
        rank,
        threads,
        flag_stride,
        hca_count,
        max_chunks,
    )
    cache_key = process_key[:-1]
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    raw = b12x_compile(
        launch,
        _dummy(cutlass.Uint32, 16),
        _dummy(cutlass.Uint32, 16),
        1,
        16,
        1,
        1,
        16,
        16,
        16,
        4096,
        16,
        16,
        16,
        16,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.roce.scatter_gather", 1, cache_key
        ),
    )

    def run(
        input_address: int,
        output_address: int,
        size_packs: int,
        nbytes: int,
        chunk_packs: int,
        chunks: int,
        send_base: int,
        recv_base: int,
        flag_base: int,
        shard_cap: int,
        ctrl_base: int,
        epoch_address: int,
        counter_base: int,
        poison_address: int,
        spin_limit: int,
        grid_x: int,
    ) -> None:
        """Launch the compiled kernel with runtime scalar arguments."""
        raw(
            make_ptr(
                cutlass.Uint32, input_address, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Uint32, output_address, cute.AddressSpace.gmem, assumed_align=16
            ),
            int(size_packs),
            int(nbytes),
            int(chunk_packs),
            int(chunks),
            int(send_base),
            int(recv_base),
            int(flag_base),
            int(shard_cap),
            int(ctrl_base),
            int(epoch_address),
            int(counter_base),
            int(poison_address),
            int(spin_limit),
            int(grid_x),
            current_cuda_stream(),
        )

    _PREPARED_LAUNCHERS.add(process_key)
    return attach_programs(run, raw)


__all__ = [
    "KIND_GATHER_SHARDS",
    "KIND_REDUCE_SCATTER",
    "OPERATIONS",
    "get_launcher",
    "is_launcher_prepared",
]
