"""Cluster-scoped phased transfers alongside the ordinary joint protocols.

A fixed 64-CTA grid consists of eight portable clusters. Phased profiles admit
only cluster zero; ordinary profiles use their selected CTA count without a
cluster barrier. The signal and payload regions retain the ordinary layout.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import program_cache
from b12x._lib.intrinsics.memory import ld_global_v4_u32, st_global_v4_u32
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from ._cute_intrinsics import (
    graph_epoch_arrive_serialized,
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_global_v4_u32,
    pack_f32x2_to_bf16x2,
    st_relaxed_sys_global_v4_u32,
)
from ._joint_cute import _JointLaunch, _EPOCH_OFFSET
from ._joint_profile import validate_cluster_geometry
from ._oneshot_cute import _dummy


@dsl_user_op
def _threadfence_sys(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "fence.sc.sys;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class _JointClusterLaunch(_JointLaunch):
    @cute.jit
    def __call__(
        self,
        peers: cute.Pointer,
        signals: cute.Pointer,
        local_signal: cute.Pointer,
        selector: cute.Pointer,
        payload: cute.Pointer,
        output: cute.Pointer,
        size: Int64,
        capacity: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            peers, signals, local_signal, selector, payload, output, size, capacity
        ).launch(grid=(64, 1, 1), block=(512, 1, 1), cluster=(8, 1, 1), stream=stream)

    @cute.jit
    def _finish_write_phase(self):
        # Every thread in the active cluster participates, including empty tails.
        _threadfence_sys()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    @cute.jit
    def _ready_phased(
        self,
        peers: cute.Pointer,
        payload: Int64,
        output: Int64,
        size: Int64,
        capacity: Int64,
        flat: Int64,
        stride: Int64,
        mode: Uint32,
    ):
        local = Int64(peers[self._rank]) + capacity * Int64(16)
        # The final peer owns the remainder, including every pack when part is zero.
        part = size // Int64(self._world_size)
        for scatter_phase in cutlass.range_constexpr(self._world_size):
            target = (self._rank + scatter_phase) % self._world_size
            begin, end = Int64(0), size
            if mode == Uint32(1):
                begin = Int64(target) * part
                end = begin + part
                if cutlass.const_expr(target == self._world_size - 1):
                    end = size
            index = begin + flat
            while index < end:
                words = ld_global_v4_u32(payload + index * Int64(16))
                dest = Int64(peers[target]) + (
                    capacity + Int64(self._rank) * capacity + index
                ) * Int64(16)
                self._publish_bf16_words(dest, words)
                index += stride
            if cutlass.const_expr(scatter_phase != 0):
                self._finish_write_phase()
        begin, end = Int64(0), size
        if mode == Uint32(1):
            begin = Int64(self._rank) * part
            end = begin + part
            if cutlass.const_expr(self._rank == self._world_size - 1):
                end = size
        index = begin + flat
        while index < end:
            acc = cute.make_rmem_tensor((8,), Float32)
            self._receive_bf16_sum(local, capacity, index, acc)
            result = cute.make_rmem_tensor((4,), Uint32)
            for word in cutlass.range_constexpr(4):
                result[word] = pack_f32x2_to_bf16x2(acc[word * 2], acc[word * 2 + 1])
            st_global_v4_u32(
                output + index * Int64(16),
                result[0],
                result[1],
                result[2],
                result[3],
            )
            index += stride
        if mode == Uint32(1):
            for gather_phase in cutlass.range_constexpr(1, self._world_size):
                target = (self._rank + gather_phase) % self._world_size
                # Match the reduction's thread map so each thread reads its own
                # output packs. Empty intervals still enter every phase barrier.
                index = begin + flat
                while index < end:
                    words = ld_global_v4_u32(output + index * Int64(16))
                    dest = Int64(peers[target]) + (
                        capacity + Int64(self._rank) * capacity + index
                    ) * Int64(16)
                    self._publish_bf16_words(dest, words)
                    index += stride
                self._finish_write_phase()
            for owner in cutlass.range_constexpr(self._world_size):
                if cutlass.const_expr(owner != self._rank):
                    remote_begin = Int64(owner) * part
                    remote_end = remote_begin + part
                    if cutlass.const_expr(owner == self._world_size - 1):
                        remote_end = size
                    index = remote_begin + flat
                    while index < remote_end:
                        address = local + (Int64(owner) * capacity + index) * Int64(16)
                        gathered = cute.make_rmem_tensor((4,), Uint32)
                        waiting = cutlass.Boolean(True)
                        while waiting:
                            words = ld_relaxed_sys_global_v4_u32(address)
                            waiting = cutlass.Boolean(False)
                            for word in cutlass.range_constexpr(4):
                                gathered[word] = words[word]
                                waiting = (
                                    waiting
                                    | ((words[word] & Uint32(0xFFFF)) == Uint32(0))
                                    | ((words[word] & Uint32(0xFFFF0000)) == Uint32(0))
                                )
                        st_global_v4_u32(
                            output + index * Int64(16),
                            gathered[0],
                            gathered[1],
                            gathered[2],
                            gathered[3],
                        )
                        st_relaxed_sys_global_v4_u32(
                            address, Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        )
                        index += stride

    @cute.kernel
    def kernel(
        self,
        peers: cute.Pointer,
        signals: cute.Pointer,
        local_signal: cute.Pointer,
        selector: cute.Pointer,
        payload: cute.Pointer,
        output: cute.Pointer,
        size: Int64,
        capacity: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        profile = selector[0]
        mode = profile >> Uint32(16)
        active = profile & Uint32(0xFFFF)
        if (mode == Uint32(5)) | (mode == Uint32(6)):
            # A partial cluster cannot enter a cluster barrier, even if a caller
            # corrupts the selector after host-side profile validation.
            if active != Uint32(8):
                active = Uint32(0)
            if cutlass.const_expr(self._family == "bandwidth"):
                if mode == Uint32(5):
                    active = Uint32(0)
        if Uint32(bid) < active:
            # Inactive clusters return before any IPC state read or epoch arrival.
            self_signal = Int64(local_signal.toint())
            generation = ld_relaxed_gpu_u32(self_signal + Int64(_EPOCH_OFFSET))
            slot_peers = peers + Int64(generation % Uint32(2)) * Int64(16)
            cute.arch.sync_threads()
            if tid == Int32(self._threads - 1):
                graph_epoch_arrive_serialized(
                    self_signal + Int64(_EPOCH_OFFSET),
                    self_signal + Int64(_EPOCH_OFFSET + 4),
                    active,
                )
            flat = Int64(bid) * Int64(self._threads) + Int64(tid)
            stride = Int64(active) * Int64(self._threads)
            src, dst = Int64(payload.toint()), Int64(output.toint())
            if mode == Uint32(6):
                self._ready_phased(
                    slot_peers,
                    src,
                    dst,
                    size,
                    capacity,
                    flat,
                    stride,
                    Uint32(1),
                )
            elif mode == Uint32(4):
                self._pull(slot_peers, signals, src, dst, size, flat, stride)
            elif mode == Uint32(1):
                self._ready(slot_peers, src, dst, size, capacity, flat, stride, mode)
            else:
                if cutlass.const_expr(self._family == "latency"):
                    if mode == Uint32(5):
                        self._ready_phased(
                            slot_peers,
                            src,
                            dst,
                            size,
                            capacity,
                            flat,
                            stride,
                            Uint32(0),
                        )
                    else:
                        self._ready(
                            slot_peers, src, dst, size, capacity, flat, stride, mode
                        )
                else:
                    if mode == Uint32(2):
                        self._bulk(
                            slot_peers,
                            signals,
                            src,
                            dst,
                            size,
                            capacity,
                            flat,
                            stride,
                            True,
                        )
                    else:
                        self._bulk(
                            slot_peers,
                            signals,
                            src,
                            dst,
                            size,
                            capacity,
                            flat,
                            stride,
                            False,
                        )


@program_cache
def get_joint_cluster_launcher(
    family, world_size, rank, threads, max_blocks, cluster_size, device_index
):
    validate_cluster_geometry(max_blocks, threads, cluster_size)
    if cluster_size != 8 or family not in ("latency", "bandwidth"):
        raise ValueError(
            "clustered joint launch requires cluster eight and a joint family"
        )
    if world_size not in (2, 3, 4, 6, 8, 10) or not 0 <= rank < world_size:
        raise ValueError("unsupported joint collective rank group")
    key = (family, world_size, rank, threads, max_blocks, cluster_size, device_index)
    launch = _JointClusterLaunch(family, world_size, rank, threads, max_blocks)
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    raw = b12x_compile(
        launch,
        _dummy(cutlass.Uint64, 8),
        _dummy(cutlass.Uint64, 8),
        _dummy(Uint32, 4),
        _dummy(Uint32, 4),
        _dummy(Uint32, 16),
        _dummy(Uint32, 16),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("comm.pcie.joint_cluster", 2, key[:-1]),
    )

    def run(
        peer_table_address,
        signal_table_address,
        local_signal_address,
        selector_address,
        input_address,
        output_address,
        size_packs,
        capacity_packs,
    ):
        if not 0 < size_packs <= capacity_packs:
            raise ValueError("joint payload must fit its prepared capacity")
        raw(
            make_ptr(
                cutlass.Uint64,
                peer_table_address,
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            make_ptr(
                cutlass.Uint64,
                signal_table_address,
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            make_ptr(
                Uint32, local_signal_address, cute.AddressSpace.gmem, assumed_align=4
            ),
            make_ptr(Uint32, selector_address, cute.AddressSpace.gmem, assumed_align=4),
            make_ptr(Uint32, input_address, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Uint32, output_address, cute.AddressSpace.gmem, assumed_align=16),
            int(size_packs),
            int(capacity_packs),
            current_cuda_stream(),
        )

    return attach_programs(run, raw)


def validate_cluster_residency(launcher, blocks, threads, cluster_size):
    """Reject unsupported compiled geometry without changing the captured grid."""
    from b12x.preparation.types import CandidateResourceIneligibleError

    (raw,) = launcher.__b12x_dependencies__
    executable = raw._executable
    if len(executable.kernel_info) != 1:
        raise RuntimeError("clustered joint launcher must retain one CUDA kernel")
    (name,) = executable.kernel_info
    status, kernel = cuda.cuLibraryGetKernel(
        cuda.CUlibrary(int(executable.library)), name.encode()
    )
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError("cannot resolve clustered joint kernel")
    status, function = cuda.cuKernelGetFunction(kernel)
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError("cannot query clustered joint kernel")
    attribute = cuda.CUlaunchAttribute()
    attribute.id = cuda.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    attribute.value.clusterDim.x = cluster_size
    attribute.value.clusterDim.y = attribute.value.clusterDim.z = 1
    config = cuda.CUlaunchConfig()
    config.gridDimX, config.gridDimY, config.gridDimZ = blocks, 1, 1
    config.blockDimX, config.blockDimY, config.blockDimZ = threads, 1, 1
    config.sharedMemBytes = 0
    config.attrs, config.numAttrs = [attribute], 1
    status, potential = cuda.cuOccupancyMaxPotentialClusterSize(function, config)
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise CandidateResourceIneligibleError(
            "cannot support the prepared joint cluster geometry"
        )
    status, resident = cuda.cuOccupancyMaxActiveClusters(function, config)
    if (
        status != cuda.CUresult.CUDA_SUCCESS
        or potential < cluster_size
        or resident * cluster_size < blocks
    ):
        raise CandidateResourceIneligibleError(
            "joint cluster grid exceeds compiled kernel residency"
        )
