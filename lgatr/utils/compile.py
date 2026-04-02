"""Utilities for optional torch.compile-based graph fusion."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
from torch import nn


def compile_enabled() -> bool:
    """Returns whether torch.compile acceleration is enabled via environment."""

    return os.getenv("LGATR_ENABLE_COMPILE", "0") == "1"


def compile_mode() -> str:
    """Returns torch.compile mode."""

    return os.getenv("LGATR_COMPILE_MODE", "max-autotune-no-cudagraphs")


def compile_dynamic() -> bool:
    """Returns whether to use dynamic shape compilation."""

    return os.getenv("LGATR_COMPILE_DYNAMIC", "0") == "1"


def compile_fullgraph() -> bool:
    """Returns whether to force full-graph compilation."""

    return os.getenv("LGATR_COMPILE_FULLGRAPH", "0") == "1"


def compile_scope() -> str:
    """Returns compile scope: ``module`` or ``full``."""

    return os.getenv("LGATR_COMPILE_SCOPE", "full").strip().lower()


def compile_requires_cuda() -> bool:
    """Returns whether compile is enabled only when CUDA is available."""

    return os.getenv("LGATR_COMPILE_REQUIRE_CUDA", "1") == "1"


def compile_supported_runtime() -> bool:
    """Checks whether runtime matches compile requirements."""

    if compile_requires_cuda() and not torch.cuda.is_available():
        return False
    return True


def maybe_compile_module(module: nn.Module) -> nn.Module:
    """Compiles a module with torch.compile if enabled and available."""

    if not compile_enabled():
        return module
    if not hasattr(torch, "compile"):
        return module
    if not compile_supported_runtime():
        return module
    return torch.compile(
        module,
        mode=compile_mode(),
        dynamic=compile_dynamic(),
        fullgraph=compile_fullgraph(),
    )


def maybe_compile_callable(fn: Callable) -> Callable:
    """Compiles a callable with torch.compile if enabled and available."""

    if not compile_enabled():
        return fn
    if not hasattr(torch, "compile"):
        return fn
    if not compile_supported_runtime():
        return fn
    return torch.compile(
        fn,
        mode=compile_mode(),
        dynamic=compile_dynamic(),
        fullgraph=compile_fullgraph(),
    )
