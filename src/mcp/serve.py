"""Shared command-line handling for the project's MCP servers.

Both servers support the two transports required by the subject:
stdio (default) and streamable HTTP (``--http``).
"""

import argparse
import json
import os
from typing import Any, Dict, Optional


def base_parser(description: str) -> argparse.ArgumentParser:
    """Return a parser with the transport options common to all servers."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--http", action="store_true",
                        help="serve over streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    return parser


def load_task(path: Optional[str], env_var: str) -> Dict[str, Any]:
    """Load a task JSON from ``path`` or from the ``env_var`` variable.

    Returns an empty dict when no task is configured.

    Raises:
        SystemExit: if the file exists but cannot be parsed.
    """
    path = path or os.environ.get(env_var)
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot load task file '{path}': {error}") from None


def serve(server: Any, args: argparse.Namespace) -> None:
    """Run a FastMCP server with the transport selected on the CLI.

    Over HTTP the endpoint is ``http://HOST:PORT/mcp``.
    """
    if args.http:
        server.settings.host = args.host
        server.settings.port = args.port
        server.run(transport="streamable-http")
    else:
        server.run()
