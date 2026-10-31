# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve one TrellisMX expert layer through b12x's in-tree P8 runtime.

The P8 kernels are Brandon M. Music's (local-inference-lab/b12x#342). This
example prepares one layer and tensor-parallel rank of the GLM-5.3 Flash
TrellisMX release with b12x, builds the in-tree P8 runtime on the same device
tensors, compiles and installs its arms, attaches it with its in-tree backend
declaration, prepares one fused-MoE execution plan per token count, runs each,
and compares the output with the CPU reference (cosine) and with a direct call
of the runtime (bytes).

From the repository root:

    PYTHONPATH="$PWD:$PWD/flashinfer/experimental/b12x/_compat" \\
      python examples/experimental/b12x_trellismx_p8_runtime.py \\
      --release /path/to/glm-5.3-flash-tr3-fp8 --layer 3 --rank 0

The release may also be named by ``B12X_TRELLISMX_GLM_RELEASE``. It needs an
SM120 or SM121 GPU. Exit status: 0 when every check passes, 1 when one fails,
2 when there is no such GPU or no release data.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

# GLM-5.3 Flash routed experts at tensor-parallel degree 4, the geometry the
# P8 runtime serves.
HIDDEN, LOCAL_INTERMEDIATE, EXPERTS, TOP_K, WORLD = 4096, 512, 288, 8, 4
SIDECAR_TENSORS = (
    "w13_trellis",
    "w2_trellis",
    "w13_scale_ue8m0",
    "w2_scale_ue8m0",
    "intermediate_scales_fp16",
    "coupled_sign_draw_u8",
    "gate_up_suh_fp16",
    "down_svh_fp16",
)
# The runtime validates its coupled scale component on the CPU.
SCALE_TENSORS = ("gate_up_suh_fp16", "intermediate_scales_fp16", "down_svh_fp16")


def stop(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(2)


def routed_inputs(rows: int, seed: int):
    """Distinct BF16 rows, distinct top-8 experts per row, normalized weights."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=generator).to(torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(EXPERTS, generator=generator)[:TOP_K] for _ in range(rows)]
    )
    weights = torch.rand(rows, TOP_K, generator=generator, dtype=torch.float32)
    return x, ids, weights / weights.sum(-1, keepdim=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--release",
        default=os.environ.get("B12X_TRELLISMX_GLM_RELEASE"),
        help="root of the TrellisMX release (default: $B12X_TRELLISMX_GLM_RELEASE)",
    )
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--tokens", default="1,2,17,129", help="comma-separated token counts to run"
    )
    parser.add_argument(
        "--check-rows",
        type=int,
        default=32,
        help="rows per call compared with the CPU reference",
    )
    args = parser.parse_args()
    token_counts = [int(value) for value in args.tokens.split(",")]

    if not torch.cuda.is_available():
        stop("No CUDA device: the P8 runtime needs an SM120 or SM121 GPU.")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) not in ((12, 0), (12, 1)):
        stop(f"The P8 runtime needs an SM120 or SM121 GPU; found sm_{major}{minor}.")
    if not args.release:
        stop("No release: pass --release or set B12X_TRELLISMX_GLM_RELEASE.")
    release = Path(args.release)
    sidecar = (
        release
        / "sidecars"
        / f"p8-layer-{args.layer:03d}-tp4-rank-{args.rank}.safetensors"
    )
    manifest_path = release / "trellismx-manifest.json"
    transform_path = release / "design" / "transform.json"
    missing = [
        str(p) for p in (sidecar, manifest_path, transform_path) if not p.is_file()
    ]
    if missing:
        stop(f"Release data absent: {', '.join(missing)}")

    from safetensors import safe_open

    from b12x.moe import fused_moe
    from b12x.moe.fused_moe.trellismx import prepare_trellismx_experts
    from b12x.moe.fused_moe.trellismx_p8 import (
        attach_trellismx_p8_runtime,
        trellismx_p8_attachment,
        verify_trellismx_p8_attachment,
    )
    from b12x.moe._shared.p8.launch import P8LaunchConfig
    from b12x.moe.fused_moe.trellismx_p8_runtime import (
        P8NativeTPMoE,
        P8RuntimeOptions,
        P8TensorSource,
        compile_p8_programs,
        in_tree_p8_backend,
    )
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.testing.reference.trellismx_moe import trellismx_moe_reference

    device = torch.device("cuda", torch.cuda.current_device())
    manifest = json.loads(manifest_path.read_text())
    record = next(
        (
            entry
            for entry in manifest["files"]
            if entry["layer"] == args.layer and entry["rank"] == args.rank
        ),
        None,
    )
    if record is None:
        stop(f"The release manifest has no layer {args.layer} rank {args.rank}.")
    with safe_open(str(sidecar), framework="pt") as handle:
        host = {name: handle.get_tensor(name) for name in SIDECAR_TENSORS}
        metadata = dict(handle.metadata())
    payload = {name: tensor.to(device) for name, tensor in host.items()}

    # 1. b12x prepares the expert bundle from the payload on the device.
    prepared = prepare_trellismx_experts(
        payload,
        metadata,
        num_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=LOCAL_INTERMEDIATE,
        top_k=TOP_K,
        device=device,
    )

    # 2. The P8 runtime reads the same device tensors in place. Its options
    #    default to the GLM-5.3 Flash serving configuration.
    tensors = {**payload, **{name: payload[name].cpu() for name in SCALE_TENSORS}}
    runtime = P8NativeTPMoE(
        P8TensorSource(tensors, metadata),
        device=device,
        tp_rank=args.rank,
        world_size=WORLD,
        layer=args.layer,
        topk=TOP_K,
        hidden=HIDDEN,
        intermediate=LOCAL_INTERMEDIATE,
        expected_design_sha256=record["source_design_sha256"],
        expected_transform_sha256=hashlib.sha256(
            transform_path.read_bytes()
        ).hexdigest(),
        options=P8RuntimeOptions(),
    )

    # 3. Compile and install every arm before capture or freeze.
    with torch.cuda.device(device):
        runtime.install_compiled_programs(
            compile_p8_programs(
                runtime.program_spec,
                max(token_counts),
                device.index or torch.cuda.current_device(),
            ),
            max(token_counts),
        )

    # 4. Attach it: the returned bundle owns the runtime and runs every call
    #    through it.
    backend = in_tree_p8_backend()
    experts = attach_trellismx_p8_runtime(prepared, runtime, backend=backend)
    verify_trellismx_p8_attachment(experts)
    attachment = trellismx_p8_attachment(experts)
    print(
        f"layer {args.layer} rank {args.rank}: K{runtime.trellis_bits} attached; "
        f"backend {backend.identity[:16]} ({len(backend.files)} files), "
        f"attachment {attachment.identity[:16]}",
        flush=True,
    )

    # 5. One execution plan per token count, prepared by a session.
    cases = {}
    requests = []
    for rows in token_counts:
        x, ids, weights = routed_inputs(rows, seed=1000 * args.layer + rows)
        case = {
            "host": (x, ids, weights),
            "fixture": x.to(device),
            "x": torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device),
            "ids": ids.to(device, torch.int32),
            "weights": weights.to(device),
            "out": torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device),
        }
        case["x"].copy_(case["fixture"])
        case["plan"] = fused_moe.plan_execution(
            experts=experts,
            capacity=fused_moe.ExecutionCapacity(
                max_tokens=rows, top_k=TOP_K, route_num_experts=0
            ),
        )

        def make_call(state, case=case):
            # Every buffer comes from the caller: the arena the plan sized, and
            # the caller's own output tensor.
            scratch = tuple(
                torch.empty(spec.shape, dtype=spec.dtype, device=device)
                for spec in state.scratch.scratch_specs()
            )
            binding = state.bind(
                a=case["x"],
                topk_ids=case["ids"],
                topk_weights=case["weights"],
                output=case["out"],
                scratch=scratch,
                input_scales_static=True,
            )

            def produce():
                case["x"].copy_(case["fixture"])

            return PreparedCall(
                run=binding.run,
                output=case["out"],
                produce=produce,
                benchmark_producers=(produce,),
                owners=(binding, *scratch),
            )

        requests.append(
            case["plan"].request(
                name=f"layer{args.layer}.moe.m{rows}",
                prepare_call=make_call,
                benchmark_call=make_call,
            )
        )
        cases[rows] = case
    # The session races the plan's launch configurations (tile, carrier
    # stride, FC1 schedule, resident-grid cap) and freezes the winner.
    session = PreparationSession(device=device, autotune=True, compile_workers=0)
    failed = False
    try:
        result = session.prepare(requests)
        session.freeze()
        print(
            f"prepared {len(requests)} capacities: "
            f"benchmarked_candidates={result.benchmarked_candidates} "
            f"restored_candidates={result.restored_candidates} "
            f"cache_hits={result.cache_hits}",
            flush=True,
        )

        # 6. Run through fused_moe and check each call.
        meta = {
            "hidden_size": HIDDEN,
            "intermediate_size": LOCAL_INTERMEDIATE,
            "num_experts": EXPERTS,
            "top_k": TOP_K,
            "bits": runtime.trellis_bits,
            "rank": args.rank,
            "world_size": WORLD,
            "sign_draw": 0,
        }
        for rows, case in cases.items():
            plan = case["plan"]
            scratch = tuple(
                torch.empty(spec.shape, dtype=spec.dtype, device=device)
                for spec in plan.scratch_specs()
            )
            binding = fused_moe.bind(
                plan,
                a=case["x"],
                topk_ids=case["ids"],
                topk_weights=case["weights"],
                scratch=scratch,
                output=case["out"],
                input_scales_static=True,
            )
            fused_moe.run(binding=binding)
            selected = (
                plan.selection.config
                if plan.selection is not None
                else P8LaunchConfig.serving()
            )
            direct = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)
            runtime(
                case["x"],
                case["weights"],
                case["ids"],
                direct,
                scratch=tuple(
                    torch.empty(spec.shape, dtype=spec.dtype, device=device)
                    for spec in runtime.scratch_specs(device, rows, selected)
                ),
                config=selected,
            )
            torch.cuda.synchronize()
            same = torch.equal(case["out"].view(torch.int16), direct.view(torch.int16))
            # Direct rows (m <= 16) carry a two-term input, grouped rows a
            # quantized one; each is compared in its own activation regime.
            direct_arm = rows <= runtime.rp2_direct_max_tokens
            count = min(rows, args.check_rows)
            x, ids, weights = case["host"]
            reference = trellismx_moe_reference(
                x[:count],
                ids[:count],
                weights[:count],
                host,
                dict(meta, quantize_activations=not direct_arm),
            ).double()
            cosine = torch.nn.functional.cosine_similarity(
                case["out"][:count].double().cpu().flatten(),
                reference.flatten(),
                dim=0,
            ).item()
            good = same and cosine >= 0.9999
            failed |= not good
            print(
                f"m={rows:5d} {'direct ' if direct_arm else 'grouped'} "
                f"launch {selected.compile_identity()} "
                f"fused_moe==runtime: {same}  cosine vs reference "
                f"({count} rows): {cosine:.7f}  {'OK' if good else 'FAIL'}",
                flush=True,
            )
    finally:
        session.close()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
