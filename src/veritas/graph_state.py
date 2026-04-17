"""State definitions for LangGraph-based meeting workflows.

This module defines the typed state schema used across LangGraph meeting workflows.
The state pattern allows each node in the graph to read and update shared state,
enabling complex multi-agent conversations with proper state management.
"""

from typing import TypedDict, Annotated, Sequence, Literal, Any
from operator import add
from langchain_core.messages import BaseMessage


class MeetingState(TypedDict):
    """State schema for LangGraph meeting workflows.

    This typed dictionary defines all state that flows through the meeting graph.
    Each node can read from and update this state. The graph ensures state is
    properly passed between nodes.

    Attributes:
        messages: Conversation history (HumanMessage, AIMessage, SystemMessage).
                  Annotated with 'add' operator to append new messages.
        current_round: Current discussion round number (0-indexed).
        max_rounds: Maximum number of discussion rounds configured.
        current_agent_index: Index of agent currently speaking in the team.
        team_lead_name: Name of the team lead agent.
        team_member_names: List of team member agent names.
        agenda: The meeting agenda/research question.
        agenda_questions: Specific questions to answer.
        agenda_rules: Rules/constraints for the meeting.
        summaries: Summaries from previous related meetings.
        contexts: Additional context for the meeting.
        meeting_type: Type of meeting ("team" or "individual").
        temperature: Sampling temperature for LLM calls.
        prompt_verbosity: Prompt verbosity level ("minimal", "standard", "verbose").
            Controls input prompt redundancy, NOT output verbosity.
        summary_instructions: Custom instructions appended to final summary prompt.
            Used to specify output format requirements (e.g., JSON schema).
        tool_outputs: Results from tool calls.
        is_final_round: Whether this is the final summary round.
        should_continue: Whether to continue to next round or terminate.
    """

    # Core conversation state
    messages: Annotated[Sequence[BaseMessage], add]

    # Round tracking
    current_round: int
    max_rounds: int
    is_final_round: bool

    # Agent tracking
    current_agent_index: int
    team_lead_name: str
    team_member_names: list[str]
    critic_name: str

    # Meeting configuration
    meeting_type: Literal["team", "individual"]
    agenda: str
    agenda_questions: tuple[str, ...]
    agenda_rules: tuple[str, ...]
    summaries: tuple[str, ...]
    contexts: tuple[str, ...]
    temperature: float
    prompt_verbosity: str  # "minimal", "standard", "verbose"
    summary_instructions: str  # Custom instructions for final summary (e.g., JSON output format)

    # Tool outputs
    tool_outputs: Annotated[list[dict[str, str]], add]

    # Code execution tracking
    code_executions: Annotated[list[dict[str, Any]], add]  # Track code-as-output executions
    last_code_caller: str  # Track which agent wrote code (for routing back)
    code_executions_this_round: int  # Count code executions in current round (safety limit)
    mcp_tool_calls_this_round: int  # Count MCP tool calls in current round (safety limit)

    # Control flow
    should_continue: bool
    last_tool_caller: str  # Track which node called tools (for routing back after execution)
    tools_just_executed: bool  # Flag to allow agent to respond after tool/code execution
    critic_feedback_pending: bool  # Whether the most recent critic feedback still needs to be addressed


class IndividualMeetingState(TypedDict):
    """State schema for individual (single-agent) meetings with critic.

    Simplified state for individual meetings where one agent proposes ideas
    and a scientific critic provides feedback.

    Attributes:
        messages: Conversation history.
        current_round: Current discussion round.
        max_rounds: Maximum rounds configured.
        agent_name: Name of the main agent.
        critic_name: Name of the critic agent.
        agenda: Meeting agenda.
        agenda_questions: Questions to answer.
        agenda_rules: Meeting rules.
        summaries: Previous meeting summaries.
        contexts: Additional context.
        temperature: Sampling temperature.
        prompt_verbosity: Prompt verbosity level ("minimal", "standard", "verbose").
            Controls input prompt redundancy, NOT output verbosity.
        summary_instructions: Custom instructions for final summary (e.g., JSON schema).
        tool_outputs: Tool execution results.
        is_final_round: Whether this is the final round.
        should_continue: Whether to continue discussion.
    """

    # Core conversation state
    messages: Annotated[Sequence[BaseMessage], add]

    # Round tracking
    current_round: int
    max_rounds: int
    is_final_round: bool

    # Agent tracking
    agent_name: str
    critic_name: str

    # Meeting configuration
    agenda: str
    agenda_questions: tuple[str, ...]
    agenda_rules: tuple[str, ...]
    summaries: tuple[str, ...]
    contexts: tuple[str, ...]
    temperature: float
    prompt_verbosity: str  # "minimal", "standard", "verbose"
    summary_instructions: str  # Custom instructions for final summary

    # Tool outputs
    tool_outputs: Annotated[list[dict[str, str]], add]

    # Code execution tracking
    code_executions: Annotated[list[dict[str, Any]], add]  # Track code-as-output executions
    last_code_caller: str  # Track which agent wrote code (for routing back)
    code_executions_this_round: int  # Count code executions in current round (safety limit)
    mcp_tool_calls_this_round: int  # Count MCP tool calls in current round (safety limit)

    # Control flow
    should_continue: bool
    tools_just_executed: bool  # Flag to allow agent to respond after tool/code execution
    critic_feedback_pending: bool  # Whether the most recent critic feedback still needs to be addressed


def create_initial_team_meeting_state(
    team_lead_name: str,
    team_member_names: list[str],
    critic_name: str,
    agenda: str,
    max_rounds: int,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
    summaries: tuple[str, ...] = (),
    contexts: tuple[str, ...] = (),
    temperature: float = 0.2,
    prompt_verbosity: str = "verbose",
    summary_instructions: str = "",
) -> MeetingState:
    """Create initial state for a team meeting.

    Args:
        team_lead_name: Name of the team lead agent.
        team_member_names: Names of team member agents.
        critic_name: Name of the critic agent (empty when critic disabled).
        agenda: The meeting agenda/research question.
        max_rounds: Maximum number of discussion rounds.
        agenda_questions: Specific questions to answer.
        agenda_rules: Rules/constraints for the meeting.
        summaries: Summaries from previous meetings.
        contexts: Additional context.
        temperature: Sampling temperature for LLM calls.
        prompt_verbosity: Prompt verbosity level ("minimal", "standard", "verbose").
            Controls input prompt redundancy, NOT output verbosity.
            Default "verbose" for backward compatibility.
        summary_instructions: Custom instructions for final summary (e.g., JSON schema).

    Returns:
        Initial MeetingState ready for graph execution.
    """
    return MeetingState(
        messages=[],
        current_round=0,
        max_rounds=max_rounds,
        is_final_round=False,
        current_agent_index=0,
        team_lead_name=team_lead_name,
        team_member_names=team_member_names,
        critic_name=critic_name,
        meeting_type="team",
        agenda=agenda,
        agenda_questions=agenda_questions,
        agenda_rules=agenda_rules,
        summaries=summaries,
        contexts=contexts,
        temperature=temperature,
        prompt_verbosity=prompt_verbosity,
        summary_instructions=summary_instructions,
        tool_outputs=[],
        code_executions=[],
        last_code_caller="",
        code_executions_this_round=0,
        mcp_tool_calls_this_round=0,
        should_continue=True,
        last_tool_caller="",
        tools_just_executed=False,
        critic_feedback_pending=False,
    )


def create_initial_individual_meeting_state(
    agent_name: str,
    critic_name: str,
    agenda: str,
    max_rounds: int,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
    summaries: tuple[str, ...] = (),
    contexts: tuple[str, ...] = (),
    temperature: float = 0.2,
    prompt_verbosity: str = "verbose",
    summary_instructions: str = "",
) -> IndividualMeetingState:
    """Create initial state for an individual meeting.

    Args:
        agent_name: Name of the main agent.
        critic_name: Name of the critic agent.
        agenda: The meeting agenda.
        max_rounds: Maximum number of rounds.
        agenda_questions: Questions to answer.
        agenda_rules: Meeting rules.
        summaries: Previous meeting summaries.
        contexts: Additional context.
        temperature: Sampling temperature.
        prompt_verbosity: Prompt verbosity level ("minimal", "standard", "verbose").
            Controls input prompt redundancy, NOT output verbosity.
            Default "verbose" for backward compatibility.
        summary_instructions: Custom instructions for final summary (e.g., JSON schema).

    Returns:
        Initial IndividualMeetingState ready for graph execution.
    """
    return IndividualMeetingState(
        messages=[],
        current_round=0,
        max_rounds=max_rounds,
        is_final_round=False,
        agent_name=agent_name,
        critic_name=critic_name,
        agenda=agenda,
        agenda_questions=agenda_questions,
        agenda_rules=agenda_rules,
        summaries=summaries,
        contexts=contexts,
        temperature=temperature,
        prompt_verbosity=prompt_verbosity,
        summary_instructions=summary_instructions,
        tool_outputs=[],
        code_executions=[],
        last_code_caller="",
        code_executions_this_round=0,
        mcp_tool_calls_this_round=0,
        should_continue=True,
        tools_just_executed=False,
        critic_feedback_pending=False,
    )
