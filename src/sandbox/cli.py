"""``uv run sandbox``: interactive command-line front-end for the sandbox.

Examples:
    uv run sandbox
    uv run sandbox sandbox_template.json
    uv run sandbox --mcp-stdio "python mcp_tools_mbpp.py" sandbox_template.json
    uv run sandbox --mcp-server http://127.0.0.1:8000/mcp
    echo 'print(1 + 1)' | uv run sandbox          # non-interactive
"""

import argparse
import codeop
import sys
from typing import Any, Optional

from src.sandbox.config import SandboxSettings
from src.sandbox.sandbox import Sandbox

HELP_TEXT = """\n\nCommands:\n \n:manual show the LLM manual \n:tools list MCP tools \n:help this help \n:quit exit
\nType Python code; a blank line ends a multi-line block."""


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(prog="sandbox",
                                     description="Secure Python sandbox.")
    parser.add_argument("config", nargs="?", help="sandbox JSON config file")
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument("--mcp-stdio", metavar="COMMAND",
                           help="start an MCP server over stdio")
    transport.add_argument("--mcp-server", metavar="URL",
                           help="connect to a streamable HTTP MCP server")
    parser.add_argument("-c", "--code", help="execute CODE and exit")
    parser.add_argument("--manual", action="store_true",
                        help="print the generated manual and exit")
    return parser.parse_args(argv)


def connect_mcp(args: argparse.Namespace) -> Any:
    """Connect to the MCP server requested on the command line, if any."""
    if not (args.mcp_stdio or args.mcp_server):
        return None
    from src.mcp.client import MCPClient

    if args.mcp_stdio:
        return MCPClient.stdio(args.mcp_stdio).connect()
    return MCPClient.http(args.mcp_server).connect()


def run_once(sandbox: Sandbox, code: str) -> int:
    """Execute one block, print its observation; return an exit status.

    SystemExit / KeyboardInterrupt raised by the code are re-raised after a
    message, so they reach the caller as the subject requires.
    """
    try:
        result = sandbox.execute(code)
    except (SystemExit, KeyboardInterrupt) as control:
        print(f"Agent code raised {type(control).__name__}: propagated to "
              "the caller, the sandbox is shutting down.")
        raise
    print(result.to_observation())
    if result.is_final:
        print(f"\nFinal answer:\n{result.final_answer}")
    return 0 if result.success else 1


def read_block() -> Optional[str]:
    """Read one complete Python block from the terminal (None on EOF)."""
    lines = []
    while True:
        try:
            line = input("... " if lines else ">>> ")
        except EOFError:
            return "\n".join(lines) if lines else None
        lines.append(line)
        source = "\n".join(lines)
        if source.strip().startswith(":"):
            return source.strip()
        try:
            complete = codeop.compile_command(source, "<input>", "exec")
        except (SyntaxError, ValueError, OverflowError):
            return source  # let the sandbox report the error
        if complete is not None and (len(lines) == 1 or not line.strip()):
            return source


def interactive(sandbox: Sandbox) -> None:
    """Run the read-execute-print loop until :quit or EOF."""
    print("Agent Smith sandbox. " + HELP_TEXT)
    while True:
        block = read_block()
        if block is None or block in (":quit", ":q", ":exit"):
            break
        if block == ":manual":
            print("\n", sandbox.manual())
        elif block == ":tools":
            print("\n".join(t.signature for t in sandbox.tools)
                  or "No MCP tools.")
        elif block == ":help":
            print(HELP_TEXT)
        elif block.strip():
            run_once(sandbox, block)


def main(argv: Optional[list] = None) -> int:
    """Entry point of the ``sandbox`` console script."""
    args = parse_args(argv)
    config = SandboxSettings.load(args.config)
    mcp = None
    try:
        mcp = connect_mcp(args)
        with Sandbox(config, mcp) as sandbox:
            if args.manual:
                print(sandbox.manual())
            elif args.code is not None:
                return run_once(sandbox, args.code)
            elif not sys.stdin.isatty():
                return run_once(sandbox, sys.stdin.read())
            else:
                interactive(sandbox)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 - never crash with a traceback
        print(f"sandbox: {error}", file=sys.stderr)
        return 1
    finally:
        if mcp is not None:
            mcp.close()


if __name__ == "__main__":
    sys.exit(main())
