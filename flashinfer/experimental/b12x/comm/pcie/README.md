# PCIe collective preparation

`AllReduce.plan` declares the eligible transports for an established channel.
`PreparationSession` races them jointly across the communicator and retains the
selected native execution state before graph capture. Each callsite submits its
actual shape, dtype, layout, arithmetic options, and aliasing contract. Equivalent
declarations coalesce; request names and layer numbers do not create separate
tuning keys. Every caller still binds its own serving operands.

## Candidate operations

| Declaration | Candidates |
| --- | --- |
| Staged `AllReduce.all_reduce`, direct peer groups | Native pull; LL and Simple peer push; BF16 payload-ready full reduction and reduce-scatter/all-gather; provisioned BF16 two-shot |
| Staged fused residual addition and RMSNorm | Native staged transports compatible with declared input, residual, normalization, grouping, and output arithmetic |
| DCP `all_gather_heads` and `lse_reduce_scatter`, TP2/4/8 | Thread count and block limit; equivalent effective grids are measured once |

Payload-ready transports require their reserved incoming slots, BF16 support,
and sufficient SMs to keep their grid resident. Two-shot retains its native
capacity and alignment constraints.
Unsupported alternatives are absent, rather than failing over during execution.
Plain tuning admits FP16/BF16 tensors with positive 16-byte-packed payloads
within the prepared staging capacity. LL requires two capacity shards per remote
source; Simple additionally keeps payload bytes disjoint from its 36 CTA flag
records. Full reduction and reduce-scatter/all-gather use separate BF16 payload
shards. Widths, row counts, and divisibility by rank count do not restrict this
measured candidate catalog.
The peer-push transports require the prepared alternating graph slots. Thread counts
32, 64, 128, 256, and 512 are measured over grids up to 36 CTAs, including the
payload-covering grid. Pull remains a measured alternative. Optimized TP2 LL and
Simple implementations remain separate entrants alongside generic peer push.
Two-shot measures warp-aligned thread counts through 512 and block limits through
64, coalescing limits that produce the same effective grid.

Native peer-push kernels support rank counts 2, 3, 4, 6, 8, and 10 with fully
mapped peers. Single-channel TP10 auto preparation provisions the actual direct
pool and retains it on success. Only a coordinated peer-mapping rejection with
complete rollback selects hierarchy; unrelated setup or cleanup failures remain
fatal. Explicit hierarchical choices and multichannel TP10 declarations retain
hierarchical routing. Pairwise reachability alone does not establish simultaneous
full-peer mapping capacity. Inputs and outputs must be 16-byte aligned, use
compatible physical traversal, and be either exact aliases or disjoint.

Tuning races the native alternatives over their launch geometry; explicitly
disabled tuning retains the static dispatch policy. Explicit
`CollectiveConfig(implementation=..., threads=..., blocks=...)` pins a legal
configuration. Native low-level plans still use `PcieConfig`.

Default-effort plain all-reduce tuning calibrates transfer-size buckets before
restricting exact-shape races to two families. Bucket upper bounds are
`1, 2, 4, 8, 16, 32, 48, 64, 96, 128` units of 8 KiB, then powers of two.
Within a preparation batch, the largest declared payload in each bucket races
all eligible families first. The winning family and the fastest other family
remain eligible for every subsequent request in that bucket; each request
still races all their eligible thread counts and block grids at its actual shape.
Across preparation batches in one session, the first completed calibration for
a bucket supplies its shortlist. Buckets are independent for each ordered GPU
group, dtype, aliasing contract, transport catalog, and staging-storage geometry.

Calibration shortlists are saved with their representative selections. Exact
selections include the calibration's program-aware cache identity and retained
families, so a changed calibration cannot reuse an incompatible selection.
Audit effort measures the unrestricted catalog. Explicit pins and disabled
tuning bypass calibration. Caller-supplied benchmark contexts and dependent
requests do not share built-in calibration. Requests retaining measured finalists
preserve their complete family catalogs. Fused arithmetic and DCP operations
retain their shape-specific searches. A bucket shortlist is empirical: performance
within a bucket should be checked against audit races when changing transport
kernels or bucket boundaries.

Hierarchical collectives, DMA rings, FP8 sequence-parallel transports, paired gathers,
and owner/top-k exchanges retain their native fixed declarations. Transport
codecs with different arithmetic are not interchangeable tuning candidates.

`PCIeTwoShotBF16.all_reduce` accepts a disjoint output or the exact input
storage span. Reduce-scatter consumes the input into a fixed private shard
before same-stream all-gather writes the output; peers consume IPC copies.
Partial overlaps remain rejected. Standalone reduce-scatter and all-gather
require disjoint outputs. All operations and graph replays sharing one runtime
must remain serialized.

## Benchmark and synchronization contract

The `_benchmark.py` fixtures allocate private operands, generate and reset inputs,
capture candidate calls, and check arithmetic before timing. They preserve
declared tensor layouts and residual aliasing. A caller can supply an explicit
`benchmark_call` for a different representative execution context.

All ranks compile and bind a candidate before any rank primes it. Each rank
queues the same captured calls behind a stream gate; the coordinator releases
the sample block after every rank reports readiness. An untimed collective
aligns GPU entry. Corresponding event samples reduce with **maximum across
ranks** before repeat sizing, elimination, and winner selection. The tournament
retains a champion and one challenger, bounding candidate operand residency to
two fixtures plus graph/event and cache-eviction storage.

`PreparationProgress.ready_collective_tuning` carries
`CollectiveTuningRequirement(key, ranks, step, payload)`. An embedding coordinator
exchanges these contributions in communicator rank order and resumes
`PreparationJob.advance(collective_tuning=contributions)`. All members must report
the same key, ranks, and step. Query signatures reject inconsistent shapes or
arithmetic before candidate entry. Cancellation is agreed at drained group
boundaries. Coordinators must propagate preparation exceptions to every member.

Collective candidates execute on the whole group; they are not divided among
the individual-kernel tuner's candidate shards. Independent TP replicas tune
their own communication group. The selection key includes the ordered GPU UUIDs,
rank-local query/program identities, and measurement policy. A valid cached
record from any member is adopted by the group. Cache-only mode fails if none
has a record. Interrupted collective races restart; completed selections reuse
the ordinary selection cache.

## Validation

Run the standalone group test with two or four visible peer-accessible SM120
GPUs:

```sh
B12X_RUN_COLLECTIVE_TUNING_TEST=1 python -m pytest -q \
  tests/experimental/b12x/comm/test_collective_tuning_gpu.py
```

The test exercises duplicate declarations, cache reuse with a missing member
cache, one-rank cancellation, inconsistent shapes, cache-only misses, and graph
replay. `benchmarks/experimental/b12x/benchmark_pcie_allreduce_transports.py`
compares explicit transports and the static dispatch policy with CUDA graphs.

## Joint BF16 profiles

Explicit joint-only plans for staged BF16 reductions with payload-ready peer
storage race two joint entrypoints. `joint_latency` supports `full`, `payload_rsag`, and
`pull`; `joint_bandwidth` supports `payload_rsag`, `bulk_push`, `bulk_pull`, and
`pull`. Ordinary plans retain standalone transport competition. Each entrypoint
uses one compiled program per thread count and maximum grid; its submode and
active CTA count are device data and do not specialize compilation. The maximum
grid is capped by the device SM count and 64 CTAs. The tuning
record includes both values. For example,
`CollectiveConfig(implementation="joint_latency", threads=256, blocks=64,
submode="full", active_blocks=16)` pins a joint configuration.

Joint workspace requires `joint=True` on `PCIeAllReduce.from_exchange_group`,
`from_process_group`, or the corresponding oneshot channel/pool factory. The
default `joint=False` allocates only static transport storage. Ranks agree on
this option before shared allocation. The public direct constructor
accepting externally supplied pointers retains standalone transports; BF16
payload-ready support alone does not admit joint candidates. Joint alternatives
require an existing workspace and never allocate one during declaration.

Coordinated channel setup allocates every admitted submode's IPC storage before
asynchronous compilation and tuning begin. Materialization reuses that storage
and allocates a device selector without entering communicator collectives.
Immutable host records enumerate admitted `(mode, active CTA count)` profiles;
the selector encodes the selected record in one uint32. These resources remain
owned by the established channel and execution state. The captured launch
retains fixed addresses, thread count, and maximum grid. The state method
`select_profile(submode, active_blocks)` collectively drains pending CUDA work,
checks rank agreement, and updates that selector. Every rank must stop submitting
work before entering this method with identical arguments. A profile change
cannot occur during capture or overlap graph replay. No topology is inferred
from the profile; callers choose the prepared profile explicitly.

### Cluster-scoped phased profiles

Experimental cluster profiles opt in through
`AllReduce.plan(..., joint_only=True, joint_cluster_size=8)`. The default is
`joint_cluster_size=1`, which preserves ordinary queries and launch policy.
Cluster plans require SM120/SM121, at least 64 SMs, 512 threads per CTA, and a
fixed 64-CTA grid. Preparation checks that the compiled kernel can keep this
grid resident with eight CTAs per cluster; unsupported geometry is rejected.

`joint_latency` additionally admits `full_phased` and `payload_rsag_phased`;
`joint_bandwidth` additionally admits `payload_rsag_phased`. These modes require
exactly eight active CTAs: one complete cluster. All other clusters return
before accessing IPC state. Each writer issues a system fence before the
active cluster synchronizes and advances to its next peer destination.
Ordinary modes retain active counts 1–64 and execute without cluster barriers.
They still use the cluster launch geometry when selected on a cluster plan;
this geometry can affect scheduling and does not guarantee zero overhead.
Both paths use the same workspace layout, BF16 arithmetic, and aliasing contract.

For an established joint channel, declare the geometry with the existing
configuration type and submit the plan through the normal preparation batch:

```python
plan = runtime.plan(
    inp,
    out=out,
    joint_only=True,
    joint_cluster_size=8,
    override=CollectiveConfig(
        implementation="joint_bandwidth",
        threads=512,
        blocks=64,
        submode="payload_rsag_phased",
        active_blocks=8,
    ),
)
```

After preparation, a startup specification can retain ordinary alternatives
alongside the phased profile:

```python
spec = joint_startup_spec(
    plan,
    profiles=(("payload_rsag_phased", 8), ("bulk_pull", 11), ("payload_rsag", 42)),
)
```

The cluster dimension is immutable plan metadata, while the selected mode and
active count remain device data. Startup selection and serving use the same
compiled launcher. Cluster launches use ordinary CUDA graph kernel attributes,
without cooperative launch or additional protocol counters.

### Captured startup mode selection

`AllReduce.plan(...)` excludes joint kernels by default and retains static transport
competition. `AllReduce.plan(..., joint_only=True)` limits an eligible staged BF16 collective
plan to the joint families. Attach `joint_finalist_policy()` to its preparation
request to retain the best measured active CTA count for every mode and compiled
geometry. The preparation cache restores these finalists with the selected plan.
`joint_startup_spec(plan)` then retains only the modes of the winning family,
thread count, and physical grid; it rejects incomplete finalist sets.

For offline recommendations, pin the geometry with `AllReduce.plan(...,
joint_only=True, override=CollectiveConfig(...))` and supply explicit profiles
to `joint_startup_spec(plan, profiles=(("bulk_push", 32), ("bulk_pull", 8)))`.
This path needs no measured finalists or preparation-time geometry race. Modes
must belong to the prepared family and active CTA counts must fit its physical
grid. Multiple counts per mode are allowed; duplicate pairs are rejected.

Collectively call `prepare_joint_startup_tuner(specs, cache_policy="cold")` before
freezing kernel resolution. All specs in a producer must have the same semantic
key, rank, and device. Each spec exposes its actual serving `selector` tensor.
The producer retains its specs, fixtures, timing storage, and compiled programs.
Its IPC storage remains owned by the collective runtime.

Capture `producer.run()` into a separate startup graph. It times the same
compiled collective launcher used by serving, with the measured CTA count for
each mode. Seven interleaved rounds use the ordinary reduction benchmark's
three fixtures and `ceil(100 / rows)` invocation chain. A device rank gate
precedes each timer; each sample is the maximum rank elapsed time. The smallest
median of 21 samples wins, with family mode order breaking ties. The producer
publishes the packed profile to every spec's selector. No host selection or
recapture is needed after startup replay.

Before replay, set the producer's persistent `int32[1]` status to `-1`. Successful
completion writes `0` after all numerical checks and positive timing samples.
Serving graphs must wait for successful startup completion, and all ranks must
replay equivalent startup graphs in the same order. Startup and serving must
not execute concurrently on their shared collective workspace.

Numerical checks run once for every mode and fixture before timing begins.
By default, each runtime owns the fixture and cold-cache eviction storage
shared by its startup producers. Share those allocations across runtimes in
one preparation session by passing an explicit owner:

```python
from b12x.comm.pcie import StartupTuningStorage, prepare_joint_startup_tuner

storage = StartupTuningStorage(runtime.device)
producer = prepare_joint_startup_tuner(specs, cache_policy="cold", storage=storage)
```

An owner shares one scratch arena and row-prefix reference banks. Preparing a
larger race rebinds existing fixtures until the first capture. After capture,
larger races get private fixtures so recorded addresses remain valid. Races
sharing an owner must replay serially, even when they use different runtimes.
Races using one runtime also serialize access to its collective and timing IPC
storage. Retain producers until replay drains; releasing the runtime, producers,
and their owner releases the fixture and eviction tensors. Create a separate
owner for each preparation session; no process-wide tensor pool retains model
storage.

Cold races can read existing resident model storage instead of allocating an
eviction buffer. Before preparing or capturing those races, use:

```python
if resident_weights.numel() * resident_weights.element_size() >= storage.eviction_bytes:
    storage.use_eviction(resident_weights)
```

The source must be a contiguous CUDA tensor on the owner's device, aligned to
four bytes. The minimum size is twice that device's L2 cache or 64 MiB,
whichever is larger. Registration creates tensor views without copying or
allocating fallback eviction storage. A smaller model can retain the default
eviction buffer by omitting registration. Registration after capture fails,
including replacement by the same source.

`producer.memory_requirements()` reports resident fixtures, timing, status, and
owned eviction storage through the preparation memory interface, with shared
storage keys suitable for deduplication. Registered external storage is
excluded because its caller already accounts for it; the reduction scalar is
still reported.

### Retained startup memory check

`benchmarks/pcie_startup_storage.py` measures live startup tensors and total
CUDA memory after capture, validated replay, synchronization, and allocator
cache release. It keeps the graphs alive and allocates the same 512 MiB resident
weight tensor before each measurement. Four BF16 message sizes use one runtime;
`--arm candidate` shares an explicit owner and reads the resident tensor for
L2 eviction. `--arm base` uses the installed revision's default preparation API.

Run from the desired FlashInfer checkout with its preparation environment and
four unused GPUs:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/pcie_startup_storage.py \
  --arm candidate --source . --output /tmp/pcie-startup-memory
```

To measure a baseline, run the same script with that checkout imported, pass its
path as `--source`, and select `--arm base`. The script rejects an import from a
different source tree. Recorded SM120 TP4 measurements are in
`benchmarks/results/pcie_startup_storage_tp4_sm120_{base,shared}.json`. These
measure retained startup memory, not inference latency.

### Standalone transfer-size search

`b12x.tools.collective_ladder` measures joint BF16 all-reduce configurations under
`torchrun`. Defaults are disjoint buffers, cold caches, and 209 transfer sizes
from 1 KiB through 8 MiB at 16 points per doubling. Each size recommends one
confirmed full configuration, including compiled geometry and runtime profile.
This tool does not change serving dispatch or preparation defaults.

For column all-gather, pass `--operation all_gather --row-bytes BYTES`. Sizes
count input bytes per rank and must contain complete rows. Automatic size
ladders align their points to that row width. The collector supports TP2/TP4/TP8
and uses exact row-major, rank-interleaved output checks.

`--gather-kernel runtime` measures device-selected CTA counts in fixed-protocol
kernels and a joint barrier/word-exchange kernel. The joint kernel supports
TP2/TP4 payloads through 16 KiB per rank; larger payloads use barrier gather.
It launches a fixed physical grid and thread count, with protocol and active
CTA count read from one uint32 selector. Word exchange uses one warp per CTA;
barrier gather uses the full CTA. Wide CTAs use predicated first-packet loads
to overlap selector and payload fetches without reconverging before dispatch.
Both protocols use the established channel's
separate receive regions and counters. All ranks must execute the same profile
and serialize calls sharing that channel.

`--gather-kernel static` measures the separate kernels with host-selected grids.
`--gather-kernel compare` includes both static and runtime candidates under the
same timing protocol. These flags affect only all-gather collection. DCP plans
with `gather_transport` set to `joint`, `selected_barrier`, or
`selected_word_exchange` compile a fixed geometry and retain a device selector.
Explicit profiles passed to `joint_startup_spec` use the shared captured startup
tuner. Ordinary DCP plans retain their preparation-time selection policy.

```sh
torchrun --standalone --nproc-per-node=4 -m b12x.tools.collective_ladder \
  --operation all_gather --row-bytes 128 --gather-kernel runtime \
  --min-bytes 128 --max-bytes 8MiB --budget-seconds 600 \
  --output gather-switched.jsonl --fresh

python -m b12x.tools.collective_ladder recommend-gather \
  --switched gather-switched.jsonl --host gather-host.jsonl \
  --output gather-recommendations.json
```

`--gather-proposal PATH` restricts collection to the geometries and profiles in a
recommendation JSON. Each bucket can also contain a `references` list of full
configuration objects (`implementation`, `threads`, `blocks`, `submode`, and
`active_blocks`) for balanced comparisons. Configurations must satisfy the
kernel's protocol and grid limits; a supplied control may include idle CTAs.
Use `--gather-capacity-bytes BYTES` when measuring a subset
of sizes to preserve the full sweep's channel capacity. `--rounds 17` enforces
at least 102 individual rank-maximum timings per measured candidate.

Run collection separately for each topology and row width. Each input report must have complete results. Reports from the same devices,
row width, channel capacity, and timing conditions are pooled before matching
payload coverage between topology cohorts. Different machines remain separate
cohorts. The merger produces `min_bytes` and one `buckets` list shared across
all measured row widths. It chooses one compiled geometry
per byte size, minimizing worst host cost among choices within 2% of the fastest
runtime kernel on every switched report. `--switched-margin` sets that margin
through 5%. If no shared geometry meets it, switched cost takes priority. The
result retains the smallest measured profile menu that stays within the
switched margin and within 5% of that geometry's best profile on each host
cohort. If the compiled geometry cannot meet the switched margin, its fastest
measured switched profiles are retained. Reported regret includes static baselines when supplied
through comparison reports. The output describes measured recommendations;
it does not install a serving ladder.

Search covers every compiled geometry and submode at every size, then refines
runtime profiles using neighboring measurements. Screening rounds scale inversely
with payload bytes, up to eight rounds by default. Up to eight candidates within
10% of the screening best enter balanced confirmation targeting 1 ms of measured
GPU time per candidate. The two fastest then compete with a 4 ms target; only
these fresh final samples determine the recommendation. Counts use measured
latency estimates and are capped at 512 rounds. `--rounds` raises the minimum.

Each round uses three exact input fixtures. An untimed invocation precedes two
individually timed invocations per fixture, with fresh input and the requested
cache preparation before each timed invocation. The pair covers both shared-buffer
slots. Eight-round captured graphs are reused to collect large sample sets with
fixed timing storage. Complete reports require all size/group coverage and both
confirmation stages; exhaustive reports additionally require the entire legal
catalog.

```sh
torchrun --standalone --nproc-per-node=8 -m b12x.tools.collective_ladder \
  --fresh --output ladder-short.jsonl

torchrun --standalone --nproc-per-node=8 -m b12x.tools.collective_ladder \
  --resume ladder-short.jsonl --budget-seconds 600

torchrun --standalone --nproc-per-node=8 -m b12x.tools.collective_ladder \
  --exhaustive --budget-seconds 3600 --output ladder-reference.jsonl

torchrun --standalone --nproc-per-node=8 -m b12x.tools.collective_ladder \
  --replay ladder-short.jsonl ladder-reference.jsonl --output ladder-comparison.jsonl
```

The default charged budget is 600 seconds, or 3600 seconds with `--exhaustive`.
Only compilation time is excluded; fixture construction, correctness checks,
capture, measurements, and report writing count. An executing measurement batch
drains before stopping. Final stdout timings include report closure; report
summary timings identify their checkpoint before the final summary write.

Use `--sizes 1KiB 64KiB 8MiB` for explicit points, or `--min-bytes`, `--max-bytes`,
and `--points-per-doubling` for another range. Sizes are 16-byte aligned.
`--cache-policy warm` and `--alias` select cache and output aliasing semantics.
`--fresh` requires an unused output path. `--resume` validates hardware, catalogs,
versions, and measurement semantics before reusing actual samples. Reports retain
all legal configurations, measured samples in microseconds, correctness results,
GPU topology with physical indices, UUIDs and PCI bus IDs, recommendations, and
missing configurations as incremental JSONL.

`--replay` independently measures the union of input reports' recommended full
configurations at every size, using the same 4 ms sampling target as final selection.
Each measurement identifies the source reports recommending its configuration.
Input reports must match sizes, hardware, catalogs, software versions, dtype,
cache policy and aliasing. Sampling protocols may differ: replay imports only
configurations and latency estimates for sample counts, and collects all timings
afresh. Resume requires an identical sampling protocol. Replay reports
identify unfinished sizes rather than treating an interrupted comparison as
complete. Compare every size against an independently measured exhaustive
reference; an aggregate alone does not establish a per-size regret bound.

### Choice and bucket plots

Install the optional Matplotlib dependency (`python -m pip install matplotlib`)
and add `--plot` to a search command to render its recommendations after GPU
resources are released. Rendering time counts toward the reported charged time.
To render a saved report without initializing CUDA or collecting measurements:

```sh
python -m b12x.tools.collective_ladder plot ladder-short.jsonl
```

Each report produces two adjacent files: `ladder-short.choices.png` shows kernel
family, runtime mode, threads, physical CTAs, and active CTAs across payloads;
`ladder-short.buckets.png` lists every exact byte size and selected configuration.
Each image contains only that report's topology and final summary. Missing and
unconfirmed recommendations are marked explicitly. Columns represent exact
sampled sizes, not inferred dispatch intervals. Regenerating plots overwrites
the PNGs without modifying the JSONL report. Replay reports contain no selected
recommendations and cannot be plotted.
