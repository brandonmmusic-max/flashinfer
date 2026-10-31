"""P8 specialization of b12x's shared W4A8 launch wrapper.

The launch itself is the shared ``_DynamicMoEW4A8Launch`` of
``b12x.moe.fused_moe._impl``; the P8 runtime only narrows the intermediate
scratch extent, because its phase kernels write exactly ``n`` intermediate
columns instead of the 128-column materialized tiles the shared phase kernels
write.
"""

from __future__ import annotations

from b12x.moe.fused_moe._impl import _DynamicMoEW4A8Launch

# The compile cache fingerprints a launch by its class's import closure: name
# the kernel module it launches so that editing the P8 kernels invalidates it.
__b12x_source_dependencies__ = ("b12x.moe._shared.p8.kernels.route_hoist_dynamic",)


class P8DynamicMoEW4A8Launch(_DynamicMoEW4A8Launch):
    """The shared W4A8 launch over the P8 kernels' unpadded intermediate."""

    def __init__(self, kernel, k, n, w1_n, num_topk):
        super().__init__(kernel, k, n, w1_n, num_topk, materialize_intermediate_n=False)
