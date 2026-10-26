"""Attribute shared tool invocations without changing the native AgentLoop."""
from contextvars import ContextVar
TOOL_ORIGIN = ContextVar('fastrsi_tool_origin', default='agent')
