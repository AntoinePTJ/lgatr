"""GPU microbenchmarks for optimized primitives."""

from __future__ import annotations

import os

import pytest
import torch

from lgatr.primitives.bilinear import geometric_product
from lgatr.primitives.invariants import (
    _load_inner_product_factors,
    _load_metric_grades,
    abs_squared_norm,
    inner_product,
)
from lgatr.primitives.linear import _compute_pin_equi_linear_basis, equi_linear, grade_project

RUN_PERF_TESTS = os.getenv("LGATR_RUN_PERF_TESTS", "0") == "1"
PERF_WARMUP = int(os.getenv("LGATR_PERF_WARMUP", "40"))
PERF_ITERS = int(os.getenv("LGATR_PERF_ITERS", "300"))
PERF_REPEATS = int(os.getenv("LGATR_PERF_REPEATS", "5"))

pytestmark = [
    pytest.mark.skipif(not RUN_PERF_TESTS, reason="Set LGATR_RUN_PERF_TESTS=1 to run performance tests."),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for performance tests."),
]


def _bench_cuda(
    fn,
    *args,
    warmup: int = PERF_WARMUP,
    iters: int = PERF_ITERS,
    repeats: int = PERF_REPEATS,
    **kwargs,
) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)

    samples_sorted = sorted(samples)
    median = samples_sorted[len(samples_sorted) // 2]
    return median, samples


def _legacy_equi_linear_einsum(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    basis = _compute_pin_equi_linear_basis(True, device=x.device, dtype=x.dtype)
    return torch.einsum("y x a, a i j, ... x j -> ... y i", coeffs, basis, x)


def _legacy_grade_project_einsum(x: torch.Tensor) -> torch.Tensor:
    basis = _compute_pin_equi_linear_basis(True, device=x.device, dtype=x.dtype)[:5]
    return torch.einsum("g i j, ... j -> ... g i", basis, x)


def _legacy_inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    factors = _load_inner_product_factors(device=x.device, dtype=x.dtype)
    return torch.einsum("... i, ... i -> ...", x * factors, y).unsqueeze(-1)


def _legacy_abs_squared_norm(x: torch.Tensor) -> torch.Tensor:
    m = _load_metric_grades(device=x.device, dtype=x.dtype)
    return torch.einsum("... i, ... i, g i -> ... g", x, x, m).abs().sum(-1, keepdim=True)


def _legacy_geometric_product_fallback(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    os.environ["LGATR_DISABLE_TRITON_GP"] = "1"
    try:
        return geometric_product(x, y)
    finally:
        os.environ.pop("LGATR_DISABLE_TRITON_GP", None)


def test_equi_linear_matmul_is_faster_than_legacy_einsum():
    x = torch.randn(8192, 16, 16, device="cuda", dtype=torch.float16)
    coeffs = torch.randn(32, 16, 10, device="cuda", dtype=torch.float16)

    # Correctness sanity
    ref = _legacy_equi_linear_einsum(x.float(), coeffs.float())
    out = equi_linear(x.float(), coeffs.float())
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)

    t_new, _ = _bench_cuda(equi_linear, x, coeffs)
    t_old, _ = _bench_cuda(_legacy_equi_linear_einsum, x, coeffs)
    print(f"[perf] equi_linear new={t_new:.4f}ms old={t_old:.4f}ms speedup={t_old / t_new:.3f}x")
    assert t_new <= 0.98 * t_old, f"Expected matmul path faster, got new={t_new:.4f}ms old={t_old:.4f}ms"


def test_geometric_product_triton_is_faster_than_fallback():
    x = torch.randn(8192, 16, 16, device="cuda", dtype=torch.float16)
    y = torch.randn(8192, 16, 16, device="cuda", dtype=torch.float16)

    os.environ.pop("LGATR_DISABLE_TRITON_GP", None)
    t_new, _ = _bench_cuda(geometric_product, x, y)
    t_old, _ = _bench_cuda(_legacy_geometric_product_fallback, x, y)
    print(f"[perf] geometric_product new={t_new:.4f}ms old={t_old:.4f}ms speedup={t_old / t_new:.3f}x")
    assert t_new <= 0.98 * t_old, f"Expected Triton GP faster, got new={t_new:.4f}ms old={t_old:.4f}ms"


def test_grade_project_and_invariants_no_regression_against_legacy():
    x = torch.randn(32768, 16, device="cuda", dtype=torch.float16)
    y = torch.randn(32768, 16, device="cuda", dtype=torch.float16)

    # Correctness checks
    assert torch.allclose(grade_project(x.float()), _legacy_grade_project_einsum(x.float()), atol=1e-6, rtol=1e-6)
    assert torch.allclose(inner_product(x.float(), y.float()), _legacy_inner_product(x.float(), y.float()), atol=1e-6, rtol=1e-6)
    assert torch.allclose(abs_squared_norm(x.float()), _legacy_abs_squared_norm(x.float()), atol=1e-6, rtol=1e-6)

    # Conservative perf checks (no severe regression)
    t_gp_new, _ = _bench_cuda(grade_project, x)
    t_gp_old, _ = _bench_cuda(_legacy_grade_project_einsum, x)
    t_ip_new, _ = _bench_cuda(inner_product, x, y)
    t_ip_old, _ = _bench_cuda(_legacy_inner_product, x, y)
    t_n_new, _ = _bench_cuda(abs_squared_norm, x)
    t_n_old, _ = _bench_cuda(_legacy_abs_squared_norm, x)
    print(
        "[perf] grade_project new={:.4f}ms old={:.4f}ms | inner_product new={:.4f}ms old={:.4f}ms | "
        "abs_squared_norm new={:.4f}ms old={:.4f}ms".format(
            t_gp_new, t_gp_old, t_ip_new, t_ip_old, t_n_new, t_n_old
        )
    )
    assert t_gp_new <= 1.10 * t_gp_old, f"grade_project regression: new={t_gp_new:.4f}ms old={t_gp_old:.4f}ms"
    assert t_ip_new <= 1.10 * t_ip_old, f"inner_product regression: new={t_ip_new:.4f}ms old={t_ip_old:.4f}ms"
    assert t_n_new <= 1.10 * t_n_old, f"abs_squared_norm regression: new={t_n_new:.4f}ms old={t_n_old:.4f}ms"
