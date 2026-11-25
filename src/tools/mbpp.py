"""MBPP tool: run a candidate solution against assert-based tests.

The candidate runs in a separate ``python -I`` process with a timeout and a
memory limit, so a broken solution cannot hang or crash the MCP server.
Each assertion is executed on its own so the report says exactly which
tests fail and, for ``assert a == b`` tests, what value was produced.
"""

import json
import resource
import subprocess
import sys
from typing import List, Optional

TIMEOUT_SECONDS = 10
MEMORY_LIMIT_BYTES = 256 * 1024 * 1024

# Runner executed in the child process. It receives a JSON payload on stdin.
_RUNNER = r'''
import ast, json, sys
data = json.loads(sys.stdin.read())
ns = {}
try:
    exec("\n".join(data["imports"] + [data["code"]]), ns)
except BaseException as e:
    print(json.dumps([["ERROR", "solution failed to load: %s: %s"
                       % (type(e).__name__, e)]]))
    sys.exit(0)
report = []
for test in data["tests"]:
    try:
        exec(test, ns)
        report.append(["PASS", test])
    except AssertionError:
        detail = ""
        node = ast.parse(test).body[0]
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            try:
                got = eval(compile(ast.Expression(node.test.left), "t", "eval"), ns)
                detail = " -> got %r" % (got,)
            except BaseException:
                pass
        report.append(["FAIL", test + detail])
    except BaseException as e:
        report.append(["FAIL", "%s -> %s: %s" % (test, type(e).__name__, e)])
print(json.dumps(report))
'''


def _limit_memory() -> None:
    """Child pre-exec hook: cap the address space of the test process."""
    resource.setrlimit(resource.RLIMIT_AS,
                       (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))


def run_mbpp_tests(code: str, tests: List[str],
                   imports: Optional[List[str]] = None) -> str:
    """Run ``code`` against ``tests`` and return a readable report."""
    if not tests:
        return "Error: no tests to run (the task has no public tests)."
    payload = json.dumps({"code": code, "tests": tests,
                          "imports": imports or []})
    try:
        done = subprocess.run(
            [sys.executable, "-I", "-c", _RUNNER], input=payload,
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
            preexec_fn=_limit_memory, env={"PATH": "/usr/bin:/bin"})
    except subprocess.TimeoutExpired:
        return (f"FAILED: the tests did not finish within {TIMEOUT_SECONDS}"
                " seconds (infinite loop or too slow).")
    try:
        report = json.loads(done.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return f"FAILED: the test process crashed.\n{done.stderr[-2000:]}"
    return format_report(report)


def format_report(report: List[List[str]]) -> str:
    """Summarise ``[[status, test], ...]`` into a short text report."""
    passed = sum(1 for status, _ in report if status == "PASS")
    if passed == len(report):
        return f"All {passed} tests passed."
    lines = [f"FAILED: {passed}/{len(report)} tests passed."]
    lines.extend(f"{status}: {text}" for status, text in report)
    return "\n".join(lines)
