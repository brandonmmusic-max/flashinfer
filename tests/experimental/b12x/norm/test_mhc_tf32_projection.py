"""Every selectable mHC projection route must keep the FP32 coefficient contract.

The projection contracts a BF16 four-stream state with FP32 mixing weights.
BF16 values are exact in TF32, so the only precision question is the weight
operand: rounding it to a single TF32 term leaves a systematic truncation bias
of up to 2^-11 of each weight, while the operator is defined by an FP32 linear
projection. The native route reaches FP32 precision, so a TF32 route that does
not compensate its weights computes a different operator at 1e-4, three orders
of magnitude above the reference, and the autotuner's choice would change the
model's numerics.
"""

from __future__ import annotations

import torch

from b12x.norm import mhc
from b12x.norm.mhc import _impl
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall
from b12x.testing.mhc import make_inputs as _make_inputs
from ..conftest import require_b12x as require_sm120
from ._mhc import prepare, bind, mhc_session

# FP32 accumulation order and the Sinkhorn iterations bound the native route at
# ~2.4e-7; a single-TF32 weight term lands at 1e-4 and above.
COEFFICIENT_ATOL = 1e-6
SINKHORN_ITERS = 20
EPS = 1e-6


def _coefficient_reference(streams, fn, scale, bias):
    """FP64 pre/post/comb for the operator's own definition."""
    flat = streams.flatten(1).double()
    mixes = (flat @ fn.double().T) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + EPS
    )
    pre = torch.sigmoid(mixes[:, :4] * scale.double()[0] + bias.double()[:4]) + EPS
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale.double()[1] + bias.double()[4:8])
    comb = mixes[:, 8:24] * scale.double()[2] + bias.double()[8:24]
    comb = torch.softmax(comb.view(-1, 4, 4), dim=-1) + EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + EPS)
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + EPS)
    return pre, post, comb


def _case(tokens, hidden, seed):
    device = require_sm120()
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens, hidden_size=hidden, seed=seed, device=device
    )
    _, prev_post, prev_comb = _coefficient_reference(residual, fn, scale, bias)
    options = dict(
        rms_eps=EPS,
        hc_eps=EPS,
        sinkhorn_iters=SINKHORN_ITERS,
        norm_weight=torch.ones(hidden, dtype=torch.bfloat16, device=device),
        norm_eps=EPS,
    )
    args = (
        x,
        residual,
        prev_post.float().contiguous(),
        prev_comb.float().contiguous(),
        fn,
        scale,
        bias,
    )
    return args, options


def _assert_contract(backend, actual, args):
    # The non-lagged post_pre returns (residual, post, comb, y); pre is consumed
    # inside the operation, so post and comb are the projection's own outputs.
    _, post, comb, _ = actual
    _, expected_post, expected_comb = _coefficient_reference(
        actual[0], args[4], args[5], args[6]
    )
    for name, got, want in (
        ("post", post, expected_post),
        ("comb", comb, expected_comb),
    ):
        torch.testing.assert_close(
            got.double(),
            want,
            rtol=0,
            atol=COEFFICIENT_ATOL,
            msg=lambda message: (
                f"{backend} {name} leaves the FP32 coefficient contract: {message}"
            ),
        )


def test_post_pre_projection_routes_keep_fp32_coefficients(mhc_session):
    """Native and TF32 post-pre both reproduce the FP64 coefficient contract."""
    for backend in ("native", "tf32_tma"):
        args, options = _case(512, 4096, 7311)
        plan = prepare(
            mhc_session, "post_pre", args, options, capacity=4096, backend=backend
        )
        assert plan.prepared.selection.config.backend == backend
        actual = mhc.run_post_pre(*args, binding=bind(plan, tokens=512), **options)
        torch.cuda.synchronize()
        _assert_contract(backend, actual, args)


def test_post_pre_projection_routes_agree_at_fp32_precision(mhc_session):
    """A route change may not move the coefficients beyond FP32 accumulation."""
    args, options = _case(4096, 4096, 7312)
    outputs = {}
    for backend in ("native", "tf32_tma"):
        plan = prepare(
            mhc_session, "post_pre", args, options, capacity=4096, backend=backend
        )
        outputs[backend] = mhc.run_post_pre(
            *args, binding=bind(plan, tokens=4096), **options
        )
    torch.cuda.synchronize()
    for name, left, right in zip(
        ("post", "comb"), outputs["native"][1:3], outputs["tf32_tma"][1:3]
    ):
        torch.testing.assert_close(
            left.double(),
            right.double(),
            rtol=0,
            atol=COEFFICIENT_ATOL,
            msg=lambda message: f"{name} differs between routes: {message}",
        )


def test_autotuned_post_pre_selection_keeps_fp32_coefficients():
    """Whatever route the race selects must meet the same contract."""
    device = require_sm120()
    tokens, hidden = 4096, 4096
    args, options = _case(tokens, hidden, 7313)
    seeds = tuple(tensor.clone() for tensor in args)
    work = tuple(torch.empty_like(tensor) for tensor in seeds)

    def produce():
        for target, seed in zip(work, seeds, strict=True):
            target.copy_(seed)

    def prepare_call(state):
        (spec,) = state.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        bound = state.bind(
            scratch=scratch,
            tokens=tokens,
            out=torch.empty((tokens, 4, hidden), dtype=torch.bfloat16, device=device),
            y=torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device),
            post=torch.empty((tokens, 4), device=device),
            comb=torch.empty((tokens, 4, 4), device=device),
        )
        return PreparedCall(
            run=lambda: _impl._b12x_mhc_post_pre_impl(
                *work, binding=bound, _state=state, **options
            ),
            produce=produce,
            owners=(scratch, bound),
        )

    plan = mhc.plan(
        mhc.Caps(device=device, max_tokens=tokens, hidden_size=hidden),
        invocation=FrozenMapping(
            dict(
                operation="post_pre",
                output_mode="provided",
                has_norm_weight=True,
                norm_weight_dtype="bfloat16",
                rms_eps=EPS,
                hc_eps=EPS,
                sinkhorn_iters=SINKHORN_ITERS,
                norm_eps=EPS,
            )
        ),
    )
    with PreparationSession(device=device, autotune=True, compile_workers=2) as session:
        session.prepare(
            (
                plan.request(
                    name="mhc-post_pre",
                    prepare_call=prepare_call,
                    benchmark_call=prepare_call,
                ),
            )
        )
        actual = mhc.run_post_pre(*args, binding=bind(plan, tokens=tokens), **options)
        torch.cuda.synchronize()
        # Read the raced selection while the session still owns the plan.
        backend = plan.prepared.selection.config.backend
    assert backend in {"native", "tf32_tma"}
    _assert_contract(backend, actual, args)
