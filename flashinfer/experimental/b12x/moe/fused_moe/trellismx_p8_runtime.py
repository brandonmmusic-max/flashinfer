"""The in-tree P8 runtime for TrellisMX experts and its backend declaration.

The P8 kernels are Brandon M. Music's, from local-inference-lab/b12x#342
(brandonmmusic-max/b12x, branch review/trellismx-p8-native-20260907). They
live in ``b12x.moe._shared.p8``; this module is their consumer entry point.

Build one runtime per layer and tensor-parallel rank from the same device
tensors the prepared bundle holds (``P8TensorSource``), compile and install
the programs for its largest token count, and attach it with the in-tree
declaration::

    runtime = P8NativeTPMoE(
        P8TensorSource(sidecar_tensors, sidecar_metadata), device=device,
        tp_rank=rank, layer=layer, intermediate=512,
        expected_design_sha256=design_sha256,
        expected_transform_sha256=transform_sha256)
    programs = compile_p8_programs(
        runtime.program_spec, max_tokens, worker_ordinal)
    runtime.install_compiled_programs(programs, max_tokens)
    experts = attach_trellismx_p8_runtime(
        prepared, runtime, backend=in_tree_p8_backend())

The attachment records the runtime's ``program_declaration``, so every plan
made from the attached experts declares the programs a call launches through
:func:`compile_p8_programs` and b12x preparation accepts them, and sizes each
capacity's caller-owned scratch from the same metadata. A call takes that arena
from its caller: ``P8NativeTPMoE.scratch_specs(device, tokens, config)`` reports
the bytes, and the runtime carves every buffer out of it instead of allocating.
The launch configuration (``P8LaunchConfig``) is what a prepared plan races;
passing one pins it.
``examples/experimental/b12x_trellismx_p8_runtime.py`` does this end to end on
a released sidecar and checks the result against the reference.
``P8RuntimeOptions()`` (the default) is the GLM-5.3 Flash serving
configuration; nothing in the P8 package reads the process environment.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from b12x.moe._shared.p8.options import (
    DOWN_REMAINDER_MODES,
    GLM53_SERVING_OPTIONS,
    P8RuntimeOptions,
)
from b12x.moe._shared.p8.p8_native_kernel import (
    P8_PROGRAM_FACTORY,
    P8NativeTPMoE,
    P8ProgramSpec,
    P8TensorSource,
    compile_p8_programs,
)

__all__ = [
    "DOWN_REMAINDER_MODES",
    "IN_TREE_P8_PACKAGE",
    "IN_TREE_P8_UPSTREAM",
    "P8_PROGRAM_FACTORY",
    "P8NativeTPMoE",
    "P8ProgramSpec",
    "P8RuntimeOptions",
    "P8TensorSource",
    "GLM53_SERVING_OPTIONS",
    "compile_p8_programs",
    "in_tree_p8_backend",
]

#: Import root of the in-tree backend, as ``TrellisMxP8Backend.package``.
IN_TREE_P8_PACKAGE = "b12x"
#: Recorded verbatim in the declaration.
IN_TREE_P8_UPSTREAM = (
    "b12x.moe._shared.p8: in-tree port of local-inference-lab/b12x#342 "
    "(Brandon M. Music; brandonmmusic-max/b12x review/trellismx-p8-native-20260907)"
)


@lru_cache(maxsize=1)
def in_tree_p8_backend():
    """The ``TrellisMxP8Backend`` declaring this package's P8 runtime.

    It pins every file of ``b12x.moe._shared.p8`` by content, so its identity
    changes whenever the P8 kernels or runtime do. The shared b12x helpers the
    kernels import are part of the same installed package and are covered by
    the compile cache's source closure. Computed once per process (b12x source
    is process-immutable); ``attach_trellismx_p8_runtime`` hashes the files
    again on every attach.
    """

    from b12x.moe.fused_moe.trellismx_p8 import TrellisMxP8Backend

    import b12x.moe._shared.p8 as p8

    package_dir = Path(p8.__file__).resolve().parent
    root = package_dir.parents[2]
    files = {}
    for path in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        files[path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    backend = TrellisMxP8Backend(
        package=IN_TREE_P8_PACKAGE, files=files, upstream=IN_TREE_P8_UPSTREAM
    )
    if backend.root() != root:
        raise RuntimeError(
            f"b12x resolves to {backend.root()}, not to the tree holding its P8 "
            f"package ({root})"
        )
    return backend
