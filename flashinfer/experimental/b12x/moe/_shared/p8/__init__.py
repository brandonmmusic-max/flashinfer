"""In-tree P8 runtime for TrellisMX (procedural NPK) routed experts.

The P8 kernels are Brandon M. Music's, from local-inference-lab/b12x#342
(brandonmmusic-max/b12x, branch review/trellismx-p8-native-20260907): a K3,
K4 or K5 trellis stream decoded to E4M3 inside the MMA, UE8M0/32 weight scales
consumed by the tensor core, and the coupled H512/H128 boundary with suh/svh
scales. This package carries them unchanged in their numerics, tile policy and
arms; ``p8_native_kernel.P8NativeTPMoE`` is the runtime, configured by
``options.P8RuntimeOptions`` and one per-call ``launch.P8LaunchConfig``.

The declaration side of the runtime is import-light: ``plan.P8ProgramSpec`` and
``plan.launch_spec`` say which arms a capacity reaches and how a launch
configuration specializes them, and ``scratch.p8_launch_layout`` is the arena a
call carves its buffers out of. ``p8_native_kernel`` builds the programs from
those declarations, so a plan can size and freeze a runtime without loading the
kernels.

``kernels.w4a8_phase1``, ``kernels.w4a8_phase2``, ``kernels.w4a8_launch`` and
``kernels.route_hoist_dynamic`` are thin subclasses of the shared
``moe._shared.kernels.w4a8_phase1/2``, ``moe._shared.kernels.dynamic.kernel``
and ``fused_moe._impl._DynamicMoEW4A8Launch``: they select the P8 arms' decode
law, input boundary, phase owners and intermediate extent, and the P8
differences those kernels need live in the shared modules behind parameters
whose defaults keep every other recipe's default specialization unchanged. No
P8 module carries a second copy of a shared kernel. Consumers use
``b12x.moe.fused_moe.trellismx_p8_runtime``.
"""
