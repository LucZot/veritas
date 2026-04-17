"""Meeting orchestration.

VERITAS drives agent meetings through a LangGraph state machine. The graph
handles node/edge routing, tool execution, and checkpointing; the router
module exposes a simple ``run_meeting`` entry point that the 4-phase
workflow (and external callers) can use without touching the graph directly.
"""

from veritas.meetings.router import run_meeting

__all__ = [
    "run_meeting",
]
