# Performance Tests

This folder contains opt-in performance tests for optimized primitives and end-to-end model steps.
It also benchmarks available attention backends (`native`, `xformers`, `flash`, `varlen`, `flex`)
for both raw attention kernels and full-model steps.
Backend-comparison tests use attention tensors with shape `(batch, heads, items, channels)`.
For compatibility across all backends, they run with `batch=1` (required by `flash` and `varlen` wrappers).

## Run

```bash
source .venv/bin/activate
LGATR_RUN_PERF_TESTS=1 python -m pytest -q tests/performance
```

With printed timings/profiler output:

```bash
source .venv/bin/activate
LGATR_RUN_PERF_TESTS=1 python -m pytest -q -s tests/performance
```

Save output to a file (while still showing it in terminal):

```bash
source .venv/bin/activate
LGATR_RUN_PERF_TESTS=1 LGATR_PERF_LOG_FILE=tests/performance/perf_gpu.log \
python -m pytest -q -s tests/performance
```

Run only backend-comparison tests:

```bash
source .venv/bin/activate
LGATR_RUN_PERF_TESTS=1 LGATR_PERF_LOG_FILE=tests/performance/perf_backends.log \
python -m pytest -q -s tests/performance -k "attention_backends or full_model_step_across_attention_backends"
```

Optional tuning knobs:
- `LGATR_PERF_WARMUP` (default `40` primitives / `20` model)
- `LGATR_PERF_ITERS` (default `300` primitives / `80` model)
- `LGATR_PERF_REPEATS` (default `5`)
- `LGATR_PERF_PROFILE_STEPS` (default `16`)
- `LGATR_ENABLE_COMPILE=1` to enable `torch.compile`
- `LGATR_COMPILE_SCOPE=full|module` (default `full`)
- `LGATR_COMPILE_MODE` (default `max-autotune-no-cudagraphs`)
- `LGATR_COMPILE_REQUIRE_CUDA=1` (default) to skip compile on CPU
- `LGATR_PERF_LOG_FILE` (default `tests/performance/perf_results.log`)

Notes:
- Tests require CUDA and are skipped otherwise.
- Benchmarks are relative (new path vs embedded legacy reference).
- Thresholds are conservative to reduce false failures across GPUs.
- Includes profiler-based checks that identify long CUDA operations in a full model step.
