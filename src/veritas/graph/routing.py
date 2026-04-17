"""Routing and decision functions for LangGraph workflows.

Contains conditional edge functions that determine graph execution flow:
- Tool execution checks
- Discussion continuation logic
- Critic response triggers
- Agent routing after tool execution
"""

from typing import Any
from veritas.constants import MAX_CODE_EXECUTIONS_PER_ROUND
from veritas.graph_state import MeetingState, IndividualMeetingState


def should_execute_tools(state: MeetingState | IndividualMeetingState) -> str:
    """Router to check if the last message contains tool calls.

    This is used as a conditional edge after agent nodes to decide whether
    to execute tools or continue to the next agent.

    Args:
        state: Current meeting state.

    Returns:
        "execute_tools" if tool calls present, "continue" otherwise.
    """
    messages = state.get("messages", [])
    if not messages:
        return "continue"

    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # Safety limit: max 16 MCP tool calls per round to prevent infinite loops
        MAX_MCP_TOOL_CALLS_PER_ROUND = 16
        calls_this_round = state.get("mcp_tool_calls_this_round", 0)
        if calls_this_round >= MAX_MCP_TOOL_CALLS_PER_ROUND:
            print(f"⚠️  [Tool Call Limit] Reached maximum ({MAX_MCP_TOOL_CALLS_PER_ROUND}) MCP tool calls this round — skipping")
            return "continue"
        return "execute_tools"

    return "continue"


def should_execute_code(state: MeetingState | IndividualMeetingState) -> str:
    """Router to check if the last message contains executable Python code blocks.

    Looks for ```python code blocks in the message content.
    Also enforces a safety limit on code executions per round to prevent infinite loops.

    Args:
        state: Current meeting state.

    Returns:
        "execute_code" if code blocks present and under limit, "continue" otherwise.
    """
    messages = state.get("messages", [])
    if not messages:
        return "continue"

    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, 'content') else ""

    # Check for Python code blocks
    if "```python" not in content:
        return "continue"

    executions_this_round = state.get("code_executions_this_round", 0)

    if executions_this_round >= MAX_CODE_EXECUTIONS_PER_ROUND:
        # Limit reached - don't execute more code
        print(f"⚠️  [Code Execution Limit] Reached maximum ({MAX_CODE_EXECUTIONS_PER_ROUND}) executions for this round")
        return "continue"

    return "execute_code"


def should_continue_discussion(state: MeetingState | IndividualMeetingState) -> str:
    """Router function to decide whether to continue discussion or end.

    This is a conditional edge function that determines the next step
    based on the current state.

    IMPORTANT: This matches LangChain's behavior where num_rounds=1 means:
    - Round 0: Agent → Critic (both speak)
    - Round 1: Agent addresses critic (final round, agent only)

    So we allow current_round to go up to max_rounds (inclusive) for the agent
    to respond to the critic in the final round.

    ADDITIONALLY: If tools were just executed, allow one more agent turn to
    analyze results, even if at max_rounds. This prevents truncating the
    discussion right after successful tool execution.

    Args:
        state: Current meeting state.

    Returns:
        "continue" to continue discussion, "end" to terminate.
    """
    # Special case: If tools were just executed, allow agent to respond
    # even if we're at or past max_rounds
    if state.get("tools_just_executed", False):
        return "continue"

    # Allow agent to respond in final round (matches LangChain's num_rounds + 1 behavior)
    # After agent responds in round == max_rounds, we end
    if state["current_round"] > state["max_rounds"]:
        return "end"

    # Check explicit termination flag
    if not state.get("should_continue", True):
        return "end"

    # Otherwise continue
    return "continue"


def should_critic_respond(state: IndividualMeetingState) -> str:
    """Router to decide if critic should respond after agent.

    Matches LangChain behavior where in the final round, only the agent speaks
    to address the critic's previous feedback.

    Args:
        state: Current individual meeting state.

    Returns:
        "critic" if critic should respond, "skip_critic" to skip to increment.
    """
    # If we're in the final round (current_round == max_rounds), skip critic
    if state["current_round"] >= state["max_rounds"]:
        return "skip_critic"
    return "critic"


def should_critic_review_team(state: MeetingState) -> str:
    """Router to decide if critic should review team discussion after a round.

    Team meetings differ from individual meetings: the critic should still
    review the last completed discussion round, and only then should the team
    lead produce the final summary. Therefore we skip the critic only once the
    workflow has moved past the configured round budget.

    Args:
        state: Current meeting state.

    Returns:
        "critic" if critic should review, "skip_critic" to continue to next round.
    """
    # Allow critic review in the final configured round; skip only once the
    # workflow has already advanced past it.
    # NOTE: safe because team meetings currently have no tool execution, so
    # tools_just_executed cannot force should_continue_discussion to return
    # "continue" past max_rounds.
    if state["current_round"] > state["max_rounds"]:
        return "skip_critic"
    return "critic"


def check_agenda_completion(state: MeetingState) -> str:
    """Advanced router that checks if agenda questions are adequately addressed.

    This is an optional advanced node that can replace simple round counting.
    It analyzes the discussion to determine if the agenda has been satisfactorily
    addressed, enabling adaptive termination.

    Args:
        state: Current meeting state.

    Returns:
        "complete" if agenda is addressed, "continue" otherwise.
    """
    # For now, this is a placeholder for future implementation
    # In a full implementation, this would:
    # 1. Extract key points from the discussion
    # 2. Check if each agenda question has been addressed
    # 3. Optionally use an LLM to judge discussion quality
    # 4. Return "complete" if quality threshold met, else "continue"

    # Simple fallback: check round count
    if state["current_round"] >= state["max_rounds"]:
        return "complete"

    return "continue"


def route_after_team_lead(state: MeetingState) -> str:
    """Router to decide where to go after team lead speaks (without tools).

    If it's the final round, the meeting should end.
    Otherwise, proceed to first member or increment_round.

    Args:
        state: Current meeting state.

    Returns:
        Node name to route to, or END.
    """
    # In final round after team lead's summary, end the meeting
    if state.get("is_final_round", False):
        return "end"

    # Otherwise, continue to next step
    team_members = state.get("team_member_names", [])
    if team_members:
        # Go to first member
        first_member = team_members[0]
        return f"member_0_{first_member.replace(' ', '_').lower()}"
    else:
        # No members, increment round
        return "increment_round"


def route_after_tools(state: MeetingState) -> str:
    """Router to send control back to the agent that called tools.

    Looks at the last tool call message to identify which agent made it,
    then routes back to that agent's node.

    Args:
        state: Current meeting state.

    Returns:
        Node name of the agent that called the tools.
    """
    # Check if last_tool_caller was set
    if "last_tool_caller" in state and state["last_tool_caller"]:
        return state["last_tool_caller"]

    # Fallback: look at message history to find who called tools
    messages = state.get("messages", [])

    # Find the last message with tool calls
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            # Get agent name from message
            if hasattr(msg, "name") and msg.name:
                agent_name = msg.name
                # Convert agent title to node name
                # For team lead: "team_lead"
                # For members: "member_N_agent_title"
                if agent_name == state.get("team_lead_name"):
                    return "team_lead"
                else:
                    # Find member index
                    team_members = state.get("team_member_names", [])
                    if agent_name in team_members:
                        idx = team_members.index(agent_name)
                        node_name = f"member_{idx}_{agent_name.replace(' ', '_').lower()}"
                        return node_name
            break

    # Default fallback
    return "team_lead"


def route_after_code_execution(state: MeetingState | IndividualMeetingState) -> str:
    """Router to send control back to the agent that wrote code.

    For individual meetings, returns to the main agent.
    For team meetings, identifies which agent wrote the code.

    Args:
        state: Current meeting state.

    Returns:
        Node name of the agent that wrote the code.
    """
    # Check if last_code_caller was set
    if "last_code_caller" in state and state["last_code_caller"]:
        caller_name = state["last_code_caller"]

        # For individual meetings, return to agent
        if "meeting_type" not in state or state.get("meeting_type") == "individual":
            return "agent"

        # For team meetings, determine node name
        if caller_name == state.get("team_lead_name"):
            return "team_lead"
        else:
            # Find member index
            team_members = state.get("team_member_names", [])
            if caller_name in team_members:
                idx = team_members.index(caller_name)
                node_name = f"member_{idx}_{caller_name.replace(' ', '_').lower()}"
                return node_name

    # Default fallback
    if "meeting_type" not in state or state.get("meeting_type") == "individual":
        return "agent"
    return "team_lead"
