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

# Run one TrellisMX (procedural NPK) MoE layer through b12x fused MoE and compare
# it with the CPU reference. A small synthetic layer is built in memory unless
# --sidecar (or B12X_TRELLISMX_SIDECAR) names a released rank sidecar, such as
# sidecars/p8-layer-003-tp4-rank-0.safetensors of the GLM-5.3 Flash TrellisMX
# release. Needs an SM120 or SM121 GPU. From the repository root:
#
#   PYTHONPATH="$PWD:$PWD/flashinfer/experimental/b12x/_compat" \
#     python examples/experimental/b12x_trellismx_moe.py

import argparse
import os
import sys

import torch


def synthetic_layer(experts=32, hidden=512, intermediate=256, bits=4, top_k=4):
    """Random procedural-NPK carriers with unit UE8M0 scales, laid out as a release."""

    gen = torch.Generator().manual_seed(0)
    tiles_h, tiles_i, words = hidden // 16, intermediate // 16, 16 * bits
    u8 = torch.uint8
    tensors = {
        "w13_trellis": torch.randint(
            0, 1 << 15, (2, experts, tiles_h, tiles_i, words), generator=gen
        ).short(),
        "w2_trellis": torch.randint(
            0, 1 << 15, (experts, tiles_i, tiles_h, words), generator=gen
        ).short(),
        "w13_scale_ue8m0": torch.full(
            (experts, 2 * intermediate, hidden // 32), 127, dtype=u8
        ),
        "w2_scale_ue8m0": torch.full(
            (experts, hidden, intermediate // 32), 127, dtype=u8
        ),
        "intermediate_scales_fp16": (
            torch.randint(0, 2, (experts, 3 * intermediate), generator=gen) * 2 - 1
        ).half(),
        "coupled_sign_draw_u8": torch.zeros(experts, dtype=u8),
        "gate_up_suh_fp16": torch.ones(hidden, dtype=torch.float16),
        "down_svh_fp16": torch.ones(hidden, dtype=torch.float16),
    }
    metadata = {
        "law": "procedural-npk",
        "compander_scale": "2.0",
        "alphabet": "e4m3",
        "scale": "ue8m0-k32",
        "boundary": "coupled-h512-h128-suh-svh-v1",
        "activation": "silu-cap10",
        "bits": str(bits),
        "hidden": str(hidden),
        "experts_start": "0",
        "experts_end": str(experts),
        "intermediate_per_rank": str(intermediate),
        "topk": str(top_k),
        "world_size": "1",
        "slice_begin": "0",
        "slice_end": str(intermediate),
    }
    return tensors, metadata, top_k


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a TrellisMX MoE layer through b12x fused MoE against the reference."
    )
    parser.add_argument("--sidecar", default=os.environ.get("B12X_TRELLISMX_SIDECAR"))
    parser.add_argument("--top-k", type=int, default=8, help="for --sidecar")
    parser.add_argument("--tokens", type=int, nargs="+", default=[8, 32])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("No CUDA device: this example needs an SM120 or SM121 GPU.")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) not in ((12, 0), (12, 1)):
        sys.exit(f"b12x needs an SM120 or SM121 GPU; found sm_{major}{minor}.")
    try:
        from b12x.moe import fused_moe
        from b12x.moe.fused_moe.trellismx import prepare_trellismx_experts
        from b12x.testing.reference.trellismx_moe import trellismx_moe_reference
    except ImportError as error:
        sys.exit(
            f"b12x with TrellisMX support is not importable ({error}); see the header."
        )

    if args.sidecar:
        if not os.path.exists(args.sidecar):
            sys.exit(f"No data: sidecar {args.sidecar} does not exist.")
        from safetensors import safe_open

        with safe_open(args.sidecar, framework="pt") as handle:
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
            metadata = dict(handle.metadata() or {})
        top_k = args.top_k
    else:
        tensors, metadata, top_k = synthetic_layer()
    _, experts, tiles_h, tiles_i, _ = tensors["w13_trellis"].shape
    hidden, intermediate = 16 * tiles_h, 16 * tiles_i
    device = torch.device("cuda", torch.cuda.current_device())
    experts_on_device = prepare_trellismx_experts(
        {name: value.to(device) for name, value in tensors.items()},
        metadata,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=top_k,
        device=device,
    )
    reference_meta = {
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_experts": experts,
        "top_k": top_k,
        "bits": int(metadata["bits"]),
        "rank": int(metadata.get("rank", 0)),
        "world_size": int(metadata.get("world_size", 1)),
        "sign_draw": int(metadata.get("sign_draw", 0)),
    }

    print(
        f"{'released sidecar ' + args.sidecar if args.sidecar else 'synthetic layer'}: "
        f"{experts} experts, hidden {hidden}, intermediate {intermediate}, top-k {top_k}"
    )
    failures = 0
    for tokens in args.tokens:
        gen = torch.Generator().manual_seed(tokens)
        x = torch.randn(tokens, hidden, generator=gen).to(torch.bfloat16)
        ids = torch.stack(
            [torch.randperm(experts, generator=gen)[:top_k] for _ in range(tokens)]
        )
        weights = torch.rand(tokens, top_k, generator=gen)
        weights = weights / weights.sum(-1, keepdim=True)

        plan = fused_moe.plan_execution(
            experts=experts_on_device,
            capacity=fused_moe.ExecutionCapacity(max_tokens=tokens, top_k=top_k),
        )
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in plan.scratch_specs()
        )
        output = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device)
        binding = fused_moe.bind(
            plan,
            a=x.to(device),
            topk_ids=ids.to(device, torch.int32),
            topk_weights=weights.to(device),
            scratch=scratch,
            output=output,
        )
        fused_moe.run(binding=binding)
        torch.cuda.synchronize()

        # m up to the small-M limit keeps 16-bit activations; the grouped arm
        # quantizes them to E4M3 with UE8M0 scales, and so does its reference.
        small_m = tokens <= getattr(binding, "trellismx_smallm_tokens", 0)
        reference = trellismx_moe_reference(
            x,
            ids,
            weights,
            tensors,
            dict(
                reference_meta,
                quantize_activations=not small_m,
                expert_subset=sorted(set(ids.flatten().tolist())),
            ),
        )
        cosine = torch.nn.functional.cosine_similarity(
            output.double().cpu().flatten(), reference.double().flatten(), dim=0
        ).item()
        bound = 0.99995 if small_m else 0.9998
        good = cosine >= bound and bool(torch.isfinite(output.float()).all())
        failures += not good
        print(
            f"m={tokens:4d} arm={'small-M' if small_m else 'grouped':7s} "
            f"cosine vs reference {cosine:.7f} (bound {bound}) {'OK' if good else 'FAIL'}"
        )
    if failures:
        sys.exit(f"{failures} token count(s) did not match the reference.")


if __name__ == "__main__":
    main()
