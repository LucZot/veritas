"""LangGraph meeting workflow builders.

Provides graph builders for team and individual meetings:
- team: Multi-agent team meetings with lead coordination
- individual: Single-agent meetings with optional critic

Functions exported for backward compatibility with existing code.
"""

from veritas.langgraph.team import build_team_meeting_graph
from veritas.langgraph.individual import build_individual_meeting_graph

__all__ = [
    "build_team_meeting_graph",
    "build_individual_meeting_graph",
]
