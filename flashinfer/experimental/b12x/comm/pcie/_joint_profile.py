"""Prepared profile geometry for joint BF16 PCIe all-reduce kernels."""

JOINT_MODES = {
    "full": 0,
    "payload_rsag": 1,
    "bulk_push": 2,
    "bulk_pull": 3,
    "pull": 4,
    "full_phased": 5,
    "payload_rsag_phased": 6,
}
JOINT_FAMILIES = {
    "joint_latency": ("full", "payload_rsag", "pull"),
    "joint_bandwidth": ("payload_rsag", "bulk_push", "bulk_pull", "pull"),
}
JOINT_CLUSTER_PHASES = {
    "joint_latency": ("full_phased", "payload_rsag_phased"),
    "joint_bandwidth": ("payload_rsag_phased",),
}


def validate_cluster_geometry(max_blocks, threads, cluster_size, device=None):
    """Admit one portable active cluster inside the fixed joint launch grid."""
    if type(cluster_size) is not int or cluster_size not in (1, 8):
        raise ValueError("joint cluster size must be 1 or 8")
    if cluster_size == 1:
        return
    if max_blocks != 64 or threads != 512:
        raise ValueError("clustered joint kernels require 64 CTAs and 512 threads")
    if device is not None and (
        device.compute_capability not in ((12, 0), (12, 1)) or device.sm_count < 64
    ):
        raise ValueError(
            "clustered joint kernels require SM120/SM121 with at least 64 SMs"
        )


def joint_modes(implementation, *, cluster_size=1):
    if type(cluster_size) is not int or cluster_size not in (1, 8):
        raise ValueError("joint cluster size must be 1 or 8")
    modes = JOINT_FAMILIES.get(implementation, ())
    return modes + (
        JOINT_CLUSTER_PHASES.get(implementation, ()) if cluster_size == 8 else ()
    )


def profile_records(implementation, max_blocks, *, cluster_size=1):
    if (
        implementation not in JOINT_FAMILIES
        or type(max_blocks) is not int
        or not 1 <= max_blocks <= 64
    ):
        raise ValueError(
            "joint profiles require a latency/bandwidth family and 1–64 CTAs"
        )
    modes = joint_modes(implementation, cluster_size=cluster_size)
    if cluster_size == 8 and max_blocks != 64:
        raise ValueError("clustered joint profiles require a 64-CTA physical grid")
    return tuple(
        (JOINT_MODES[mode], blocks)
        for mode in modes
        for blocks in (
            (8,)
            if mode in JOINT_CLUSTER_PHASES[implementation]
            else range(1, max_blocks + 1)
        )
    )


def profile_index(
    implementation, max_blocks, submode, active_blocks, *, cluster_size=1
):
    if type(active_blocks) is not int or not 1 <= active_blocks <= max_blocks:
        raise ValueError("active CTA count must fit the prepared maximum grid")
    if submode not in joint_modes(implementation, cluster_size=cluster_size):
        raise ValueError("submode does not belong to the prepared joint family")
    if submode in JOINT_CLUSTER_PHASES.get(implementation, ()) and active_blocks != 8:
        raise ValueError(
            "phased joint profiles require exactly one active eight-CTA cluster"
        )
    return profile_records(implementation, max_blocks, cluster_size=cluster_size).index(
        (JOINT_MODES[submode], active_blocks)
    )


def selected_profile(
    implementation, max_blocks, submode, active_blocks, *, cluster_size=1
):
    """Encode an admitted profile for one device load without changing its identity."""
    index = profile_index(
        implementation, max_blocks, submode, active_blocks, cluster_size=cluster_size
    )
    mode, blocks = profile_records(
        implementation, max_blocks, cluster_size=cluster_size
    )[index]
    return (mode << 16) | blocks


JOINT_MAX_BLOCKS = 64
JOINT_MAX_RANKS = 16
JOINT_SIGNAL_BYTES = 266248


def joint_slot_packs(capacity, world):
    """Reserve staging, payload-ready, bulk scatter, and bulk gather regions."""
    if type(capacity) is not int or capacity <= 0 or world not in (2, 3, 4, 6, 8, 10):
        raise ValueError(
            "joint storage requires positive capacity and a supported peer count"
        )
    return (world + 1) * capacity + 2 * world * ((capacity + world - 1) // world)
