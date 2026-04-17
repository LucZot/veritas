"""Utility nodes for LangGraph workflows.

Simple helper nodes for graph state management.
"""

from typing import Any
from veritas.graph_state import MeetingState, IndividualMeetingState


def _print_round_header(round_num: int, max_rounds: int) -> None:
    """Print a visual round separator to the terminal."""
    label = f" Round {round_num}/{max_rounds} "
    print(f"──{label}{'─' * max(0, 46 - len(label))}")


def increment_round_node(state: MeetingState | IndividualMeetingState) -> dict[str, Any]:
    """Node that increments the round counter and checks if final round.

    Also resets per-round execution counters.

    Args:
        state: Current meeting state.

    Returns:
        Updated state with incremented round, final round flag, and reset counters.
    """
    new_round = state["current_round"] + 1
    is_final = new_round >= state["max_rounds"]

    # Print round header for the upcoming round
    if not is_final:
        _print_round_header(new_round + 1, state["max_rounds"])

    return {
        "current_round": new_round,
        "is_final_round": is_final,
        "code_executions_this_round": 0,  # Reset counters for new round
        "mcp_tool_calls_this_round": 0,
        "critic_feedback_pending": False,
    }
