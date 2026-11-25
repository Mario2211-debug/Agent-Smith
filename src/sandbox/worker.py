"""Sandbox worker: executes agent code inside an isolated interpreter.

The host starts this file with ``python -I worker.py`` (no site-packages,
no environment variables, clean env). It then talks to the host over
stdin/stdout using :mod:`protocol`. The namespace persists between
executions, so variables survive from one agent step to the next.
"""

import ast
import linecache
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol  # noqa: E402
from guard import INTERNAL_MODULES, Guard  # noqa: E402
from runtime import (FILENAME, FinalAnswer, OutputCapture,  # noqa: E402
                     ToolError, final_answer, format_error)

class Worker:
    """Owns the guard, the persistent namespace and the host channel."""

    def __init__(self, channel_in, channel_out, init: dict):
        """Build the guard and the namespace described by ``init``."""
        self.channel_in, self.channel_out = channel_in, channel_out
        self.guard = Guard(init["authorized_imports"],
                           init["allowed_directories"], init["max_memory_mb"])
        self.memory_mb = init["max_memory_mb"]
        self.capture = OutputCapture(channel_out, init["max_output_chars"])
        self.namespace = {"__name__": "__main__",
                          "__builtins__": self.guard.builtins_dict(),
                          "final_answer": final_answer}
        for tool in init.get("tools", []):
            self.namespace[tool["name"]] = self._make_tool(tool)
        for method in ("list_resources", "read_resource", "list_prompts",
                       "get_prompt"):
            self.namespace[method] = self._make_mcp_helper(method)

    def _rpc(self, message: dict):
        """Send a request to the host and return the value of its reply."""
        self.capture.flush()
        protocol.send(self.channel_out, message)
        reply = protocol.receive(self.channel_in)
        if reply is None:
            raise ToolError("Sandbox host disconnected")
        if not reply.get("ok"):
            raise ToolError(reply.get("error", "unknown tool error"))
        return reply.get("value")

    def _make_tool(self, spec: dict):
        """Create a Python function forwarding calls to an MCP tool."""
        name, params = spec["name"], spec["params"]

        def tool(*args, **kwargs):
            if len(args) > len(params):
                raise TypeError(f"{name}() takes at most {len(params)} "
                                f"positional arguments ({len(args)} given)")
            for pname, value in zip(params, args):
                if pname in kwargs:
                    raise TypeError(f"{name}() got multiple values for "
                                    f"'{pname}'")
                kwargs[pname] = value
            return self._rpc({"op": "tool", "name": name,
                              "arguments": kwargs})

        tool.__name__ = tool.__qualname__ = name
        tool.__doc__ = spec.get("doc", "")
        return tool

    def _make_mcp_helper(self, method: str):
        """Create a helper exposing MCP resources/prompts to agent code."""
        def helper(*args):
            return self._rpc({"op": "mcp", "method": method,
                              "args": list(args)})
        helper.__name__ = method
        return helper

    def execute(self, code: str) -> None:
        """Run ``code`` in the namespace and report a ``done`` message."""
        self.capture.reset()
        done = {"op": "done", "error": None, "final_answer": None,
                "control": None, "exit_code": None}
        linecache.cache[FILENAME] = (len(code), None,
                                     code.splitlines(True), FILENAME)
        saved = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = self.capture
        try:
            tree = ast.parse(code, FILENAME)
            self.guard.check_ast(tree)
            self.guard.active = True
            self._run_tree(tree)
        except FinalAnswer as answer:
            done["final_answer"] = answer.value
        except SyntaxError as error:
            done["error"] = (f"SyntaxError: {error.msg} (line {error.lineno})"
                             f"\n    {(error.text or '').strip()}")
        except MemoryError:
            done["error"] = (f"MemoryError: the sandbox memory limit "
                             f"({self.memory_mb} MB) was exceeded.")
        except (KeyboardInterrupt, SystemExit) as control:
            done["control"] = type(control).__name__
            done["exit_code"] = getattr(control, "code", None)
        except BaseException as error:  # noqa: BLE001 - reported to the LLM
            done["error"] = self.guard.trusted_call(format_error, error)
        finally:
            self.guard.active = False
            sys.stdout, sys.stderr = saved
            self.capture.flush()
        done["output_chars"] = self.capture.total
        protocol.send(self.channel_out, done)

    def _run_tree(self, tree: ast.Module) -> None:
        """Execute the module; echo the last bare expression like a REPL."""
        last = tree.body.pop() if tree.body and \
            isinstance(tree.body[-1], ast.Expr) else None
        exec(compile(tree, FILENAME, "exec"), self.namespace)
        if last is not None:
            value = eval(compile(ast.Expression(last.value), FILENAME,
                                 "eval"), self.namespace)
            if value is not None:
                print(value if isinstance(value, str) else repr(value))


def main() -> None:
    """Initialise the guard, then serve ``exec`` requests until shutdown."""
    channel_in, channel_out = sys.stdin, sys.stdout
    init = protocol.receive(channel_in)
    if not init:
        return
    worker = Worker(channel_in, channel_out, init)
    for module in INTERNAL_MODULES:  # preload before the guard is active
        try:
            __import__(module)
        except ImportError:
            pass
    os.chdir(init["cwd"])
    sys.__stdout__ = sys.__stderr__ = worker.capture
    worker.guard.install()
    protocol.send(channel_out, {"op": "ready"})
    while True:
        message = protocol.receive(channel_in)
        if message is None or message.get("op") == "shutdown":
            break
        if message.get("op") == "exec":
            worker.execute(message.get("code", ""))


if __name__ == "__main__":
    main()
