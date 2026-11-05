"""D-x2: two-term (hi + lo) E4M3/UE8M0-K32 carrier at the P8 down (FC2) input only."""

from __future__ import annotations

import importlib

# The variants below are looked up in each phase kernel's own module. These are
# the modules that define them, declared for the compile cache's source closure.
__b12x_source_dependencies__ = (
    "b12x.moe._shared.p8.kernels.direct_m32_fc2",
    "b12x.moe._shared.p8.kernels.fc2_k5_funnel",
    "b12x.moe._shared.p8.kernels.grouped_fc2_grid376",
    "b12x.moe._shared.p8.kernels.policy_m32_fc1",
    "b12x.moe._shared.p8.kernels.policy_m32_fc2",
    "b12x.moe._shared.p8.kernels.route_hoist_direct32",
    "b12x.moe._shared.p8.kernels.route_hoist_k5",
    "b12x.moe._shared.p8.kernels.route_hoist_prefill",
)


def enable_down_remainder(backend, phases: str = "both") -> None:
    """Upgrade a constructed backend's phase kernels to their statically defined DX2 classes.

    `phases` is a diagnostic split: "fc1" only writes the lo plane (FC2 ignores it), "fc2" only
    stages the (zero) lo plane and issues the extra MMA. Both must be bit-identical to prod.
    """
    if phases not in ("both", "fc1", "fc2"):
        raise ValueError(f"unknown D-x2 phase split: {phases!r}")
    attrs = {
        "both": ("materialized_phase1_kernel", "materialized_phase2_kernel"),
        "fc1": ("materialized_phase1_kernel",),
        "fc2": ("materialized_phase2_kernel",),
    }[phases]
    for attr in attrs:
        obj = getattr(backend, attr)
        cls = type(obj)
        dx2 = getattr(
            importlib.import_module(cls.__module__), cls.__name__ + "DX2", None
        )
        if dx2 is None or not issubclass(dx2, cls):
            raise RuntimeError(f"no D-x2 variant for {cls.__module__}.{cls.__name__}")
        obj.__class__ = dx2
    if phases == "fc1":
        return
    fc2 = backend.materialized_phase2_kernel
    # FC2 double-buffers the lo A tile + lo scales after everything it already owns.
    fc2.dx2_a_offset = (int(fc2.shared_bytes) + 1023) // 1024 * 1024
    fc2.shared_bytes = fc2.dx2_a_offset + 2 * int(fc2.a_stage_bytes)
    fc2.shared_words = (fc2.shared_bytes + 3) // 4
    fc2.trellis_lut_offset = fc2.shared_bytes


def enable_rowpack(backend, input_hop: bool = False) -> None:
    """Direct (M<=16) rowpack variant: the direct phase kernels become their static RP classes;
    shared memory is unchanged.

    input_hop (the RP2 variant): FC1 becomes the RP2 class and the backend's input prologue also writes x_lo.
    """
    if input_hop:
        backend.p8_input_rowpack = True
    for attr in ("materialized_phase1_kernel", "materialized_phase2_kernel"):
        obj = getattr(backend, attr)
        cls = type(obj)
        suffix = "RP2" if (input_hop and attr == "materialized_phase1_kernel") else "RP"
        rp = getattr(
            importlib.import_module(cls.__module__), cls.__name__ + suffix, None
        )
        if rp is None or not issubclass(rp, cls):
            raise RuntimeError(
                f"no D-x2-RP variant for {cls.__module__}.{cls.__name__}"
            )
        obj.__class__ = rp
