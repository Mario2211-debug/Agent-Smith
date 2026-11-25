"""JSON-lines protocol between the sandbox host and its worker process.

Messages are single-line JSON objects with an ``op`` field:

host -> worker
    ``init``    configuration and tool list (sent once)
    ``exec``    code to execute
    ``reply``   answer to a ``tool`` / ``mcp`` request
    ``shutdown`` stop the worker

worker -> host
    ``ready``   worker initialised
    ``out``     a chunk of captured stdout/stderr
    ``tool``    call an MCP tool (host answers with ``reply``)
    ``mcp``     call an MCP helper: resources / prompts
    ``done``    execution finished (error, final answer, control flow)

This module only uses the standard library because the worker runs in an
isolated interpreter without site-packages.
"""

import json
from typing import IO, Any, Dict, Optional


def send(stream: IO[str], message: Dict[str, Any]) -> None:
    """Write one message as a JSON line and flush it."""
    stream.write(json.dumps(message, default=repr) + "\n")
    stream.flush()


def decode(line: str) -> Optional[Dict[str, Any]]:
    """Decode one JSON line; return ``None`` for an empty line (EOF)."""
    if not line:
        return None
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        return {"op": "out", "data": line}
    return message if isinstance(message, dict) else {"op": "out",
                                                      "data": line}


def receive(stream: IO[str]) -> Optional[Dict[str, Any]]:
    """Read and decode the next message; ``None`` means the peer is gone."""
    return decode(stream.readline())
