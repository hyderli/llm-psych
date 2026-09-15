"""Tests for the sandboxed code-execution primitive.

Covers :func:`llm_psych.sandbox.run_function_tests`: correct pass/fail
behavior, compile-error handling, the denylist, and the timeout.
"""

from __future__ import annotations

import pytest

from llm_psych.sandbox import UnsafeCodeError, run_function_tests


def test_correct_solution_passes():
    source = "def add(a, b):\n    return a + b\n"
    result = run_function_tests(source, "add", {"t1": ((1, 2), 3), "t2": ((0, 0), 0)})
    assert result.compiled
    assert result.all_passed
    assert result.passed("t1")
    assert result.passed("t2")


def test_wrong_solution_fails_individual_test():
    source = "def add(a, b):\n    return a - b\n"
    result = run_function_tests(source, "add", {"t1": ((1, 2), 3)})
    assert result.compiled
    assert not result.passed("t1")
    assert not result.all_passed


def test_syntax_error_does_not_compile():
    source = "def add(a, b:\n    return a + b\n"
    result = run_function_tests(source, "add", {"t1": ((1, 2), 3)})
    assert not result.compiled
    assert result.compile_error is not None
    assert not result.passed("t1")


def test_missing_function_does_not_compile():
    source = "def other(a, b):\n    return a + b\n"
    result = run_function_tests(source, "add", {"t1": ((1, 2), 3)})
    assert not result.compiled
    assert not result.passed("t1")


def test_runtime_error_counts_as_fail_not_crash():
    source = "def add(a, b):\n    return a / 0\n"
    result = run_function_tests(source, "add", {"t1": ((1, 2), 3)})
    assert result.compiled  # defined fine
    assert not result.passed("t1")  # but raised at call time
    assert result.outcomes["t1"].error is not None


@pytest.mark.parametrize(
    "bad_source",
    [
        "import os\ndef f(x):\n    os.system('echo hi')\n    return x\n",
        "def f(x):\n    return eval(x)\n",
        "def f(x):\n    import subprocess\n    return x\n",
        "def f(x):\n    return open('/etc/passwd').read()\n",
    ],
)
def test_denylist_rejects_dangerous_patterns(bad_source):
    with pytest.raises(UnsafeCodeError):
        run_function_tests(bad_source, "f", {"t1": ((1,), 1)})


def test_infinite_loop_times_out():
    source = "def f(x):\n    while True:\n        pass\n"
    result = run_function_tests(source, "f", {"t1": ((1,), 1)}, timeout_s=1.0)
    assert not result.compiled
    assert "timed out" in (result.compile_error or "")


def test_multiple_test_cases_mixed_outcome():
    source = "def square(x):\n    return x * x\n"
    result = run_function_tests(
        source,
        "square",
        {"good": ((3,), 9), "bad": ((3,), 10)},
    )
    assert result.passed("good")
    assert not result.passed("bad")
    assert not result.all_passed
