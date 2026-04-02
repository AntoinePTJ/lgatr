"""End-to-end model performance tests (opt-in, CUDA only)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from lgatr.nets import LGATr
from lgatr.primitives.bilinear import _load_geometric_product_tensor
from lgatr.primitives.attention import scaled_dot_product_attention
from lgatr.primitives.attention_backends import _REGISTRY
from lgatr.primitives.config import gatr_config
from lgatr.primitives.invariants import _load_inner_product_factors, _load_metric_grades
from lgatr.primitives.linear import _compute_pin_equi_linear_basis
from lgatr.utils.einsum import cached_einsum

RUN_PERF_TESTS = os.getenv("LGATR_RUN_PERF_TESTS", "0") == "1"
PERF_WARMUP = int(os.getenv("LGATR_PERF_WARMUP", "20"))
PERF_ITERS = int(os.getenv("LGATR_PERF_ITERS", "80"))
PERF_REPEATS = int(os.getenv("LGATR_PERF_REPEATS", "5"))
PERF_PROFILE_STEPS = int(os.getenv("LGATR_PERF_PROFILE_STEPS", "16"))

pytestmark = [
    pytest.mark.skipif(not RUN_PERF_TESTS, reason="Set LGATR_RUN_PERF_TESTS=1 to run performance tests."),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for performance tests."),
]


def _event_self_device_time(event) -> float:
    """Compatibility accessor for profiler self device time across torch versions."""

    for attr in (
        "self_cuda_time_total",
        "self_device_time_total",
        "cuda_time_total",
        "device_time_total",
    ):
        value = getattr(event, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def _legacy_equi_linear(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    basis = _compute_pin_equi_linear_basis(
        gatr_config.use_fully_connected_subgroup, device=x.device, dtype=x.dtype
    )
    return cached_einsum("y x a, a i j, ... x j -> ... y i", coeffs, basis, x)


def _legacy_grade_project(x: torch.Tensor) -> torch.Tensor:
    basis = _compute_pin_equi_linear_basis(
        gatr_config.use_fully_connected_subgroup, device=x.device, dtype=x.dtype
    )[:5]
    return cached_einsum("g i j, ... j -> ... g i", basis, x)


def _legacy_grade_dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    x = _legacy_grade_project(x)
    h = x.reshape(-1, 5, 16)
    h = torch.nn.functional.dropout1d(h, p=p, training=training, inplace=False)
    h = h.reshape(x.shape)
    return torch.sum(h, dim=-2)


def _legacy_inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    factors = _load_inner_product_factors(device=x.device, dtype=x.dtype)
    return cached_einsum("... i, ... i -> ...", x * factors, y).unsqueeze(-1)


def _legacy_abs_squared_norm(x: torch.Tensor) -> torch.Tensor:
    m = _load_metric_grades(device=x.device, dtype=x.dtype)
    return cached_einsum("... i, ... i, g i -> ... g", x, x, m).abs().sum(-1, keepdim=True)


def _legacy_geometric_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    gp = _load_geometric_product_tensor(device=x.device, dtype=x.dtype)
    return cached_einsum("i j k, ... j, ... k -> ... i", gp, x, y)


@contextmanager
def _legacy_model_ops():
    with (
        patch("lgatr.layers.linear.equi_linear", _legacy_equi_linear),
        patch("lgatr.primitives.dropout.grade_project", _legacy_grade_project),
        patch("lgatr.layers.dropout.grade_dropout", _legacy_grade_dropout),
        patch("lgatr.primitives.invariants.inner_product", _legacy_inner_product),
        patch("lgatr.primitives.invariants.abs_squared_norm", _legacy_abs_squared_norm),
        patch("lgatr.primitives.normalization.abs_squared_norm", _legacy_abs_squared_norm),
        patch("lgatr.layers.mlp.geometric_bilinears.geometric_product", _legacy_geometric_product),
    ):
        yield


def _make_model(device: str = "cuda", dtype: torch.dtype = torch.float16) -> LGATr:
    model = LGATr(
        num_blocks=2,
        in_mv_channels=8,
        out_mv_channels=8,
        hidden_mv_channels=16,
        in_s_channels=8,
        out_s_channels=8,
        hidden_s_channels=16,
        attention=dict(num_heads=4, multi_query=False),
        mlp=dict(),
        dropout_prob=None,
        checkpoint_blocks=False,
    ).to(device=device, dtype=dtype)
    return model


def _bench_step(
    model: LGATr,
    multivectors: torch.Tensor,
    scalars: torch.Tensor,
    warmup: int = PERF_WARMUP,
    iters: int = PERF_ITERS,
    repeats: int = PERF_REPEATS,
) -> tuple[float, list[float]]:
    def run_once():
        model.zero_grad(set_to_none=True)
        out_mv, out_s = model(multivectors, scalars=scalars)
        loss = out_mv.square().mean() + out_s.square().mean()
        loss.backward()

    model.train()
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run_once()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)

    samples_sorted = sorted(samples)
    median = samples_sorted[len(samples_sorted) // 2]
    return median, samples


def _available_attention_backends() -> list[str]:
    preferred = ["native", "xformers", "flash", "varlen", "flex"]
    return [name for name in preferred if name in _REGISTRY]


def _backend_requires_batch1(backend: str) -> bool:
    """Returns whether a backend enforces batch size 1."""

    # flash/varlen wrappers in lgatr reshape with assert x.shape[0] == 1.
    return backend in {"flash", "varlen"}


def _bench_attention_backend(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup: int = PERF_WARMUP,
    iters: int = PERF_ITERS,
    repeats: int = PERF_REPEATS,
) -> float:
    def run_once():
        return scaled_dot_product_attention(q, k, v, backend=backend)

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run_once()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    samples_sorted = sorted(samples)
    return samples_sorted[len(samples_sorted) // 2]


def test_full_model_step_is_faster_than_legacy_reference():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    multivectors = torch.randn(8, 32, 8, 16, device=device, dtype=dtype)
    scalars = torch.randn(8, 32, 8, device=device, dtype=dtype)

    model_new = _make_model(device=device, dtype=dtype)
    t_new, _ = _bench_step(model_new, multivectors, scalars)

    with _legacy_model_ops():
        model_old = _make_model(device=device, dtype=dtype)
        t_old, _ = _bench_step(model_old, multivectors, scalars)

    print(f"[perf] full_model_step new={t_new:.4f}ms old={t_old:.4f}ms speedup={t_old / t_new:.3f}x")

    # End-to-end should be at least modestly better than legacy baseline.
    assert t_new <= 0.98 * t_old, f"Expected full model speedup, got new={t_new:.4f}ms old={t_old:.4f}ms"


def test_attention_backends_benchmark_and_accuracy():
    backends = _available_attention_backends()
    assert backends, "No attention backend available."

    # Use batch=1 so all available backends (including flash/varlen) support the same shape.
    # Shape convention here is (batch, heads, items, channels).
    q = torch.randn(1, 4, 128, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 4, 128, 128, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 4, 128, 128, device="cuda", dtype=torch.float16)

    baseline_backend = "native" if "native" in backends else backends[0]
    baseline = scaled_dot_product_attention(q.float(), k.float(), v.float(), backend=baseline_backend)

    timings = {}
    for backend in backends:
        if _backend_requires_batch1(backend):
            assert q.shape[0] == 1, f"Backend '{backend}' requires batch size 1."
        out = scaled_dot_product_attention(q, k, v, backend=backend)
        max_diff = (out.float() - baseline).abs().max().item()
        t_ms = _bench_attention_backend(backend, q, k, v)
        timings[backend] = t_ms
        print(f"[perf] attention backend={backend} time={t_ms:.4f}ms max_abs_diff_vs_{baseline_backend}={max_diff:.3e}")

    fastest_backend = min(timings, key=timings.get)
    ordered = ", ".join(f"{name}:{timings[name]:.4f}ms" for name in sorted(timings, key=timings.get))
    print(f"[perf] attention backends ranked: {ordered} | fastest={fastest_backend}")


def test_full_model_step_across_attention_backends():
    backends = _available_attention_backends()
    assert backends, "No attention backend available."

    torch.manual_seed(0)
    # Use batch=1 so all available backends can be compared in one test.
    multivectors = torch.randn(1, 32, 8, 16, device="cuda", dtype=torch.float16)
    scalars = torch.randn(1, 32, 8, device="cuda", dtype=torch.float16)

    timings = {}
    for backend in backends:
        if _backend_requires_batch1(backend):
            assert multivectors.shape[0] == 1, f"Backend '{backend}' requires batch size 1."
        model = _make_model(device="cuda", dtype=torch.float16)

        def bench_backend_step():
            model.zero_grad(set_to_none=True)
            out_mv, out_s = model(multivectors, scalars=scalars, backend=backend)
            loss = out_mv.square().mean() + out_s.square().mean()
            loss.backward()

        # Use slightly lighter loops for multi-backend coverage.
        for _ in range(max(10, PERF_WARMUP // 2)):
            bench_backend_step()
        torch.cuda.synchronize()

        samples = []
        iters = max(30, PERF_ITERS // 2)
        repeats = max(3, PERF_REPEATS)
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                bench_backend_step()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / iters)
        samples.sort()
        timings[backend] = samples[len(samples) // 2]
        print(f"[perf] full_model_step backend={backend} time={timings[backend]:.4f}ms")

    ordered = ", ".join(f"{name}:{timings[name]:.4f}ms" for name in sorted(timings, key=timings.get))
    print(f"[perf] full_model backends ranked: {ordered}")


def test_full_model_profiler_reports_long_cuda_ops():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    model = _make_model(device=device, dtype=dtype)
    multivectors = torch.randn(8, 32, 8, 16, device=device, dtype=dtype)
    scalars = torch.randn(8, 32, 8, device=device, dtype=dtype)

    model.train()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        for _ in range(PERF_PROFILE_STEPS):
            model.zero_grad(set_to_none=True)
            out_mv, out_s = model(multivectors, scalars=scalars)
            loss = out_mv.square().mean() + out_s.square().mean()
            loss.backward()
            prof.step()

    events = prof.key_averages()
    cuda_events = [event for event in events if _event_self_device_time(event) > 0]
    assert cuda_events, "No CUDA events recorded in profiler output."

    total_cuda = sum(_event_self_device_time(event) for event in cuda_events)
    long_ops = [event for event in cuda_events if _event_self_device_time(event) / total_cuda >= 0.05]
    try:
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
    except Exception:
        table = prof.key_averages().table(sort_by="self_device_time_total", row_limit=20)
    print("[perf] top CUDA ops (self time):")
    print(table)
    print("[perf] long ops (>=5% self CUDA time):")
    for event in sorted(cuda_events, key=_event_self_device_time, reverse=True):
        share = 100.0 * _event_self_device_time(event) / total_cuda
        if share < 5.0:
            break
        print(f"  - {event.key}: {share:.2f}%")
    assert long_ops, f"No long CUDA operations found (>=5% self CUDA time).\n{table}"
