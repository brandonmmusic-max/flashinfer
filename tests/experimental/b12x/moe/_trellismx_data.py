"""Where the TrellisMX real-data tests find their data.

Every location is optional. A test that needs one skips when its variable is
unset or the file it names is absent.

``B12X_TRELLISMX_GLM_RELEASE``
    Root of the GLM-5.3 Flash TrellisMX release
    (``sidecars/p8-layer-LLL-tp4-rank-R.safetensors``).
``B12X_TRELLISMX_DS41_TESTDATA``
    DeepSeek-V4.1 Flash TrellisMX test sidecars
    (``K2/trellismx/tp2/...`` and ``K4/trellismx/tp4/layer-000-rank-R.safetensors``).
``B12X_TRELLISMX_DS41_BASE``
    The DeepSeek-V4.1 Flash base checkpoint in CSF form (``tensors/``, ``receipts/``).
``B12X_TRELLISMX_CAPTURES``
    Directories, separated by ``os.pathsep``, holding forwards recorded from the
    release's own runtime (``oracle_glm_l3_r0*.npz``, ``oracle_smallm_m*.npz``).
"""

from __future__ import annotations

import os


def _root(name: str) -> str:
    # An unset root still yields a path, one that cannot exist, so a missing
    # dataset is an ordinary skip whose message names the variable.
    return os.environ.get(name) or f"<{name} unset>"


GLM_RELEASE = _root("B12X_TRELLISMX_GLM_RELEASE")
DS41_TESTDATA = _root("B12X_TRELLISMX_DS41_TESTDATA")
DS41_BASE = _root("B12X_TRELLISMX_DS41_BASE")


def glm_sidecar(layer: int = 3, rank: int = 0) -> str:
    """One released GLM sidecar file."""

    return os.path.join(
        GLM_RELEASE, "sidecars", f"p8-layer-{layer:03d}-tp4-rank-{rank}.safetensors"
    )


def ds41_sidecar(bits: int, rank: int = 0) -> str:
    """One DeepSeek-V4.1 layer-0 TrellisMX test sidecar (K2 at TP2, K4 at TP4)."""

    world = {2: 2, 4: 4}[bits]
    return os.path.join(
        DS41_TESTDATA,
        f"K{bits}",
        "trellismx",
        f"tp{world}",
        f"layer-000-rank-{rank}.safetensors",
    )


def capture(name: str) -> str:
    """The first capture directory holding ``name``, or a path that does not exist."""

    for directory in os.environ.get("B12X_TRELLISMX_CAPTURES", "").split(os.pathsep):
        if directory and os.path.exists(os.path.join(directory, name)):
            return os.path.join(directory, name)
    return os.path.join("<B12X_TRELLISMX_CAPTURES unset or without it>", name)
