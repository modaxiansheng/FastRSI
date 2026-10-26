"""Agent tool for explicit runtime session verification."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from PhyAgentOS.agent.tools.base import Tool

if TYPE_CHECKING:
    from PhyAgentOS.agent.session_verifier import SessionVerifier


class VerifySessionTool(Tool):
    """Thin tool adapter over the Agent-owned session verifier."""

    def __init__(self, verifier: "SessionVerifier"):
        self.verifier = verifier

    @property
    def name(self) -> str:
        return "verify_session"

    @property
    def description(self) -> str:
        return (
            "Verify a runtime session waiting for semantic validation, or review an already "
            "verified session when its RGB evidence is still available."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Runtime session ID from SESSIONS.md",
                }
            },
            "required": ["session_id"],
        }

    async def execute(self, session_id: str, **kwargs: Any) -> str:
        outcome = await self.verifier.verify_session(session_id, source="tool")
        return json.dumps(outcome, ensure_ascii=False, indent=2)
