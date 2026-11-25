"""Secure, configurable execution sandbox for LLM-generated Python code.

Public API:
    Sandbox          run code, expose MCP tools, generate the manual
    SandboxSettings  Pydantic configuration (loadable from JSON)
    ExecutionResult  outcome of one execution and its LLM observation
"""

from src.sandbox.config import SandboxSettings
from src.sandbox.result import ExecutionResult
from src.sandbox.sandbox import Sandbox

__all__ = ["ExecutionResult", "Sandbox", "SandboxSettings"]
