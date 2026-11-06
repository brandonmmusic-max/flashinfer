# W4A16 with NVFP4 activation prefill

Expert-presence marking, NVFP4-CSF scale expansion, and deterministic dynamic
route reduction have registered mutating operators. ATen tracing preserves
these launches and their caller-owned outputs for subsequent compilation and
CUDA graph capture. Scale expansion retains its supplied prepared CuTe program;
route reduction uses the current capture stream.

This experimental SM120 path reuses ModelOpt NVFP4 W4A16 weights for
NVFP4 activation GEMMs during prefill. Activation quantization changes model
outputs and requires model quality validation before deployment. The default
remains W4A16.

Use `ActivationSpec(mode="a16", a4_prefill=True)` to retain calibrated A4
prefill on the same packed weight owner. Explicit enablement requires finite,
positive input and intermediate calibration scales and compatible geometry;
invalid preparation or an unavailable requested A4 binding raises an error.
It uses one activation plane and eight warps independently of the A4 environment
controls. `a4_prefill=False` disables the capability.

The default `a4_prefill=None` preserves environment-based preparation:
`B12X_W4A16_A4_PREFILL=1` enables the capability, while absent calibration or a
pair of all-one placeholder scales retains W4A16. The shared input global scale
is the minimum across experts. Intermediate scales remain per expert. Ordinary
A4 weight preparation retains its per-expert scales.

Supported weights use packed ModelOpt NVFP4 with ordinary or compressed
NVFP4-CSF block scales, BF16 input/output, SiLU gating, hidden size divisible
by 256, and intermediate size divisible by 64. FC1 masks the unused half of
a final 128-channel intermediate tile and reads gate/up scale halves from
native packed scale storage. A positive `swiglu_limit` uses
the W4A16 activation contract: clamp the BF16 GEMM gate above the limit and the
up input to the symmetric interval, round SiLU and up to BF16, multiply, then
round the intermediate to BF16 before quantization. Expert maps, input router
weighting, activation-amax collection, and unsupported layouts cannot use A4.

The execution binding accepts `a4_prefill`:

- `True` selects prepared A4 launches.
- `False` or `None` selects W4A16.

Session preparation can fix this choice with
`plan_execution(invocation=FrozenMapping({"a4_prefill": False}))` for A16 or
`True` for A4 prefill. The declared choice participates in the tuning query
and compilation identity. Binding inherits it when the argument is omitted;
an explicit conflicting choice is rejected. An A16 declaration can tune both
inline CSF and scratch expansion even when the weights retain A4 capability.
Inline is selected only when it wins the workload's tuning race. A4 declarations
retain expanded scales. Omitting the declaration preserves per-binding precision
selection and its conservative inline eligibility.

The caller selects precision from request semantics: prefill may use A4;
decode and speculative verification use A16. Token count does not select
hybrid precision, and `ActivationSpec.a16_max_tokens` does not override an
explicit `a4_prefill=True`. Each call must fit its prepared capacity and
caller-owned scratch buffers. When an exact decode variant lacks the padded
A4 workspace, a composite plan reuses a larger fitting prepared variant in
the same scratch arena. A plan with no fitting A4 variant retains W4A16.

Compressed NVFP4 and MXFP4 scales use `csf_32x2_v1` resident blocks. MoE
preparation exposes `MoeDecodeConfig.csf_inline_scales` as a tuning choice for
compatible FP4 kernels. `False` expands scales into caller-owned native buffers
before compute; small calls expand routed experts and long prefills expand all
experts. `True` reconstructs scales inside the compute kernel's shared-memory
pipeline stages. Both choices read the same resident
payload and preserve native scale bytes, normalization, and arithmetic.

The autotuner measures the complete invocation, including expansion, route
planning, synchronization resets, and reduction. Token capacity is part of the
tuning query. Preparation compiles the candidates and fixes the selected
configuration and programs before CUDA graph capture. Explicit
`MoeDecodeConfig` overrides pin supported choices; incompatible geometry is
rejected during planning. On SM120,
untuned preparation prefers inline scales for moderate routing density and
expansion for tiny calls, compact N64 W4A8, and dense prefill. Inline readers
cover M16/M32 NVFP4 W4A4 and MXFP4 W4A8 tiles, native compact N64 MXFP4 W4A8
micro kernels, and packed W4A16 kernels. Scalar kernels,
M64/M128 activation-quantized tiles, materialized NVFP4 W4A4,
source-native W4A16, dense GEMM, and the optional
W4A16 activation-prefill path consume expanded scales. Unsupported inline
overrides are rejected during planning.

The SM120 defaults select inline from 16 tokens through 16 routed rows per
expert for NVFP4 W4A4 and packed W4A16, and from 8 tokens through 32 routed
rows per expert for noncompact MXFP4 W4A8. These density limits use total routed
rows divided by the declared expert count. NVFP4 switches to M32 at 12 rows
per expert and eligible materialized M128 at 48; noncompact MXFP4 uses M32
from 16 rows per expert and retains its dense M64 crossover. These are untuned
defaults; autotuning retains both supported scale choices, including compact
N64 inline candidates. SM121 retains its device-specific defaults.

M16/M32 inline candidates at routing densities above one tile per expert are
searched only at `B12X_AUTOTUNE_EFFORT=audit`; the default effort omits them.
Resident-grid caps race a power-of-two ladder plus half, three-quarter,
seven-eighth and full rungs up to the cooperative residency bound. The
default effort omits caps below a quarter of the SMs, and on the compact W4A8
path it races the micro kernel only up to 16 tokens and the dynamic kernel only
from 8 tokens; the audit races every rung and both kernels. `off` selects the
heuristic configuration. Effort is captured in the tuning query and does not affect
compiled program identities. Compact MXFP4 W4A8
retains reconstructed K128 scale words across both K64 pipeline steps.

Packed FP4 W4A16 execution searches FC1/FC2 tile pairs, route-block sizes,
and pipeline depths. Explicit launch geometries use packed routing; the
specialized direct kernels remain separate candidates. Mixed-precision
queries retain the launch search of each constituent precision, including
W4A8 resident-grid caps. Benchmark callbacks must time the complete routed
operation, including activation conversion and output reduction.

Eligible compact N64 MXFP4 shards offer direct-route micro kernels through
16 tokens and grouped M16 or M64 compute. M64 FC1 uses M32 arithmetic when a
source tile has at most 32 live rows, while retaining its prepared M64 storage
and launch capacity. Four-lane groups quantize each activated K32 block with
the same BF16 boundary and MXFP8 scale calculation.

Native compact N64 MXFP4 W4A8 uses `fp4_e8m0_k32` weights with intermediate
width congruent to 64 modulo 128. Its micro FC1 tuning controls are
`micro_fc1_tile_n` (64 or 128), `micro_fc1_max_copy_groups` (1 or 2 pending
copy groups), and `micro_fc1_pipeline_stages` (1 or 2). One stage requires
one pending copy group. N128 tiles pair gate/up projections when the prepared
token capacity is 4 through 8. For these paired projections,
`compact_fuse_input_quantization=True` quantizes the input inside FC1; it
requires BF16 I/O, SiLU gating, and hidden size divisible by 256. `False`
uses a separate input-quantization launch. Both support inline or expanded
scales when the corresponding scale layout is available.

`materialized_fc1_cta_budget` sets the compiler's resident-CTA target per SM
and the FC1 launch-grid multiplier to 2 (default) or 3. Three requires dynamic
M16 native compact N64 MXFP4 W4A8 with SiLU gating, hidden size divisible by
256, and inline CSF scales. These controls participate in complete-invocation
tuning and remain fixed during captured replay;
live route counts do not retune or replace the prepared geometry.

Expansion reconstructs each selected expert's scales once per invocation.
Inline reconstruction repeats for every compute tile that consumes those
scales. Inline can reduce memory traffic when many experts are active and each
has few token tiles. With few active experts, expanded scales can remain in L2;
with many token tiles per expert, repeated inline decoding competes with compute
while expansion is amortized. NVFP4 has one scale byte per 16 weights, twice
MXFP4's scale density. Kernel staging, routing distribution, and cache residency
determine whether the saved traffic pays for decoding and synchronization.

The benchmark option `--scale-compression csf_32x2_v1` uses these resident
blocks. `--moe-config '{"backend":"w4a16","route_planner":"internal",
"max_active_clusters":null,"w4a16_route_mode":"packed",
"csf_inline_scales":true}'` forces inline W4A16. Omit `--moe-config` to tune
both choices. `--time-graph-replay` measures the captured invocation, including
scale expansion when selected.

Pre-routed single-op benchmarks default to `--routing-workload mixture`.
The synthetic scenarios cover diffuse expert popularity, concentrated popularity,
and correlated preferences within groups of four tokens. Expert IDs are sampled
without replacement within each token. `--tuning-samples` selects independent
seeds per scenario, and `--routing-samples` selects held-out evaluation seeds.
No recorded model inputs are required. `--routing-workload default` uses the
profile's input generator; individual workload names allow controlled comparisons.

Candidate races minimize the arithmetic mean of invocation latencies across the
synthetic scenarios. Equal scenario weights provide distribution coverage;
they are not measured production traffic frequencies. Invocation-frequency
weights, when known, belong in this arithmetic mean. Geometric mean speedup can
favor a configuration that consumes more total GPU time. Evaluation reports
individual scenarios and their arithmetic mean. Each benchmark process uses a
fresh selection cache so a previous routing distribution cannot supply its winner.

`expand_scales(experts)` expands every NVFP4 expert on the current stream.
A later `bind(..., scales_expanded=True)` skips per-call expansion, except for
micro kernels whose synchronization reset is fused with expansion. Callers
must order expansion after the previous use of shared scale scratch and before
compute. Concurrent execution streams require disjoint scratch or explicit
stream dependencies.

`uses_expanded_nvfp4_scales(plan, num_tokens=..., a4_prefill=...,
route_ids_dtype=..., scales_expanded=...)` reports whether a prepared call has
an expansion consumer. It returns `False` for an inline plan, allowing callers
to skip scale prefetch. The query performs no GPU work.

With `ActivationSpec.a4_prefill=None`, `B12X_W4A16_A4_PREFILL_TERMS=1`
uses one NVFP4 activation plane. A value of `2`
adds an independently quantized residual plane. `B12X_W4A16_A4_PREFILL_WARPS`
accepts `8` (default) or `16`. These controls and the capability flag are immutable
execution-plan inputs and are part of the tuning query. Changing the
environment requires preparing another execution plan. Kernel compilation
retains device-owned callables; replay performs no compilation or tensor
allocation. A4 always applies router weights in the FP32 final sum. Set
`B12X_W4A16_FP32_TOPK_WEIGHTS=1` before importing B12X when comparing with an
A16 baseline that also uses FP32 router weighting.

Run the synthetic quantization-contract and changed-input graph checks on an
SM120 GPU from the FlashInfer repository:

```sh
PYTHONPATH="$PWD:$PWD/flashinfer/experimental/b12x/_compat" \
  .venv/bin/python -m pytest \
  tests/experimental/b12x/moe/test_w4a16_a4_prefill.py -v
```

The two-device ownership check requires two visible SM120 GPUs. These checks
verify the specified lossy arithmetic and replay behavior; they do not measure
model accuracy or serving throughput.

# TrellisMX weights and the external P8 runtime

`source_format="trellismx"` reads a released procedural-NPK TrellisMX sidecar
(`fused_moe.trellismx.prepare_trellismx_experts`) onto the W4A8 trellis recipe.
Preparation accepts only what every TrellisMX path executes: BF16 activations
and GLM-5.3's `silu-cap10`, `silu(min(gate, 10)) * clamp(up, -10, 10)` with the
clamp before SiLU. Other dtypes, limits or clamp placements, router weighting on
the input, activation-amax collection, expert maps and A4 prefill are rejected
while planning. `plan_execution(..., smallm_max_tokens=N)` declares how much of
the small-M arm's 1..16 token range the plan serves: at most `N` tokens, the
default 16, or `0` to send every token count to the grouped arm. The value is
validated (an integer in 0..16), recorded in the plan's controls and tuning
query, and compiled for; no environment variable selects it. Calls with at most
`N` tokens take a small-M arm with 16-bit activations when every UE8M0
weight-scale byte lies in 112..139, the range in which its fp16 weight-scale
product is exact, and the call allows fast math; otherwise the grouped arm
serves every token count. The plan resolves that choice once, keeps it in its
identity, compiles for it, and stamps it on every binding it makes; dispatch
follows the binding. Preparation also builds the arm's plan and programs for
every planned token count the arm serves, so plan, bind and capture need no
eager call, and capture allocates and compiles nothing.

The arm's activation, route and output buffers are caller-owned scratch. Its
plan declares them (`smallm_scratch_layout`: one 16-byte-aligned region per
buffer), the variant's arena reserves them after the grouped arm's regions, and
`bind` carves one set per binding out of the buffer it was given. A call with a
bound caller-owned output writes only its binding's views and allocates nothing,
which is the only form capture accepts; an eager call whose binding carries no
output allocates the BF16 tensor it returns. A missing or short arena is
rejected at bind, and dispatch refuses a binding that carries no carved
buffers. Two bindings of one prepared layer therefore hold distinct storage and
may be captured, replayed and run without serializing against each other; only
the compiled programs are shared, and those are read-only.

`fused_moe.trellismx_p8.attach_trellismx_p8_runtime(experts, runtime,
backend=TrellisMxP8Backend(...), options=None)` returns a copy of the prepared
experts that owns an external P8 runtime and runs every call through it. The
backend declaration pins the runtime's source files by SHA-256, hashed again on
every attach. Attachment verifies them; the geometry, bitrate, SwiGLU limit and
reducer configuration against the prepared record and the experts' plan; that
every tensor the runtime passes its kernels as a raw pointer is contiguous,
16-byte aligned and exactly as long as expected; that the runtime reads the
experts' own carriers and scale planes in place; and that its repacked scales
and coupled-boundary vectors are byte-equal to the experts'. The runtime's
resolved behavior settings (`P8_OPTION_ATTRIBUTES`: RP2 row packing, direct
kernel limit and tile, FC1 schedule, grid policy and the rest) are recorded and
are part of the attachment identity and of the plan query; file hashes alone do
not identify them. `options` may name settings the runtime must have resolved.
`plan_execution` on attached experts races the runtime's launch configuration
per capacity, as it does for any b12x component. b12x preparation rejects a
call that launches a b12x-compiled program its plan did not declare, so the
runtime states a `program_declaration` (a compile-job factory from its pinned
backend and the metadata it compiles from); each capacity of the plan carries
one compile job per raced configuration, and each call launches only declared
programs. The plan also sizes every capacity's caller-owned scratch from that
metadata, so a runtime without a declaration cannot be planned: preparation
would not know how much arena its calls need. Keep the attached experts, or a
binding or plan made from them, alive while a captured graph that uses them can
replay.

While attached, nothing the attachment verified may change. Replacing a
verified tensor, record field or plan field with `dataclasses.replace` drops
nothing silently: it raises. Planning, binding and dispatch reject a runtime
whose tensor handles or settings changed (host-side checks, no device
synchronization). In-place writes to tensor contents cannot be seen without
reading device memory; `verify_trellismx_p8_attachment(experts)` re-checks
them on request, outside capture. To change weights, scales or boundary
vectors, detach the runtime, prepare the experts again, attach again, and
recapture every graph made from the old experts: replays of graphs captured
earlier bypass every check.

b12x ships one such runtime: `fused_moe.trellismx_p8_runtime.P8NativeTPMoE`,
Brandon M. Music's P8 kernels from local-inference-lab/b12x#342, in
`moe/_shared/p8`. Its behavior settings are one `P8RuntimeOptions` passed at
construction; the defaults are the GLM-5.3 Flash serving configuration and
nothing is read from the environment. `in_tree_p8_backend()` is its
`TrellisMxP8Backend` declaration, pinning every file of that package. It
compiles through b12x's compiler and declares its programs with
`compile_p8_programs`, which builds each arm and the reducer from the
runtime's `P8ProgramSpec` with the runtime's own code.

A call takes its whole arena from the caller and writes a caller-owned output.
`P8NativeTPMoE.scratch_specs` reports the bytes a capacity's calls need, the
plan declares exactly that much in `MemoryRequirements.scratch`, and the runtime
carves every buffer (route carrier, activation scales, task queues and the FP32
per-route partial sum) out of the bound buffer. Nothing is allocated during a
call, so a captured graph holds no graph-private pool memory and a missing,
too-small or 16-byte misaligned buffer raises instead of allocating. The result
is not part of the arena: `P8NativeTPMoE.__call__` requires `out=`, so a call
never hands back scratch that the next call would overwrite. A fused-MoE
binding without an output gets one allocated on the eager path and must carry
one under capture.

The launch geometry is not an option. One `P8LaunchConfig` (the direct and
grouped tile M, the direct carrier stride, the direct arm's FC1 pipeline depth
and warp count, and the resident-grid cap) is a candidate of the component's
tuning contract: preparation races the configurations that can change the
capacity's own call, compiles one set of programs per configuration, and fixes
the winner in the prepared state. The serving configuration is one of the
candidates, and it is also what a runtime uses when a call names no
configuration. A `P8LaunchConfig` passed to `plan_execution` pins the choice
and skips the race. The resident-grid cap has one owner per value: a cap
configuration `0` selects the runtime's `grid_policy`, and a runtime built with
`mac_override` pins its cap, which wins over any raced cap (the race then keeps
the cap axis at `0` and rejects a configuration that names another).

Each capacity declares the arms its calls reach: three for the serving tile
policy, because 1 and 2 tokens take the same direct arm (`select_tile(1)[0]`
tiles every row count up to the direct limit) and 17 and 129 take the grouped
arms of their tile bands. Compile every declared arm before capture; a compile
after kernel resolution is frozen raises.
`examples/experimental/b12x_trellismx_p8_runtime.py` prepares a released layer,
attaches the runtime and checks it against the reference.

The runtime never leaves its process. Pickling or deep-copying attached
experts, or a binding or prepared plan state that holds them, produces a
`TrellisMxP8Descriptor` in the attachment's place. It keeps the backend
declaration, contract, options, identity and program declaration, owns no
runtime and no programs, plans the same fixed-backend plan with the same
compile jobs, binds, and raises
`TrellisMxP8Unavailable` on any launch instead of falling back to this package's
arms. Attaching a runtime to experts that carry a descriptor rebinds them and
requires the same identity.

CPU contract checks:

```sh
PYTHONPATH="$PWD:$PWD/flashinfer/experimental/b12x/_compat" \
  .venv/bin/python -m pytest \
  tests/experimental/b12x/moe/test_trellismx_p8_attachment.py \
  tests/experimental/b12x/moe/test_trellismx_p8_runtime.py \
  tests/experimental/b12x/moe/test_trellismx_source.py -v
```
