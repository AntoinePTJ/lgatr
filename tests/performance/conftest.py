"""Pytest logging helpers for performance tests."""

from __future__ import annotations

import builtins
import os
import sys
from datetime import datetime
from pathlib import Path
from threading import Lock

import pytest

_LOG_LOCK = Lock()


def _perf_log_path() -> Path:
    """Returns the performance log file path."""

    return Path(os.getenv("LGATR_PERF_LOG_FILE", "tests/performance/perf_results.log"))


def _append_log(text: str) -> None:
    """Appends one text chunk to the performance log."""

    path = _perf_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Starts a fresh performance log for each test session."""

    path = _perf_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# LGATR performance test log\n"
        f"# started: {datetime.now().isoformat()}\n"
        f"# run_perf_tests={os.getenv('LGATR_RUN_PERF_TESTS', '0')}\n"
        f"# perf_warmup={os.getenv('LGATR_PERF_WARMUP', '<default>')}\n"
        f"# perf_iters={os.getenv('LGATR_PERF_ITERS', '<default>')}\n"
        f"# perf_repeats={os.getenv('LGATR_PERF_REPEATS', '<default>')}\n"
        f"# perf_profile_steps={os.getenv('LGATR_PERF_PROFILE_STEPS', '<default>')}\n"
        f"# compile={os.getenv('LGATR_ENABLE_COMPILE', '0')}\n"
        f"# compile_scope={os.getenv('LGATR_COMPILE_SCOPE', '<default>')}\n"
        f"# compile_mode={os.getenv('LGATR_COMPILE_MODE', '<default>')}\n\n"
    )
    path.write_text(header, encoding="utf-8")


@pytest.fixture(autouse=True, scope="session")
def _tee_print_to_perf_log():
    """Duplicates print output to the performance log."""

    original_print = builtins.print

    def tee_print(*args, **kwargs):
        original_print(*args, **kwargs)

        destination = kwargs.get("file", None)
        if destination not in (None, sys.stdout, sys.stderr):
            return

        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        message = sep.join(str(arg) for arg in args) + end
        _append_log(message)

    builtins.print = tee_print
    try:
        yield
    finally:
        builtins.print = original_print


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Appends final pytest summary to the performance log."""

    tr = terminalreporter
    summary = (
        "\n# pytest summary\n"
        f"passed={len(tr.stats.get('passed', []))} "
        f"failed={len(tr.stats.get('failed', []))} "
        f"skipped={len(tr.stats.get('skipped', []))}\n"
    )
    _append_log(summary)
