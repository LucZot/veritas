"""Agent node creation functions for LangGraph workflows.

Contains factory functions that create agent nodes with specific behaviors:
- Generic agents
- Team leads with round-aware prompting
- Team members
- Individual meeting agents
- Critics (individual and team)
"""

from typing import Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from veritas.agent import Agent
from veritas.graph_state import MeetingState, IndividualMeetingState
from veritas.prompts import (
    team_meeting_team_lead_initial_prompt,
    team_meeting_team_lead_intermediate_prompt,
    team_meeting_team_lead_final_prompt,
    team_meeting_team_member_prompt,
    individual_meeting_agent_prompt,
    individual_meeting_critic_prompt,
)
from veritas.verbosity import get_verbosity_config


def _log_agent_response(agent_title: str, state: dict, is_critic: bool = False) -> None:
    """Print a one-line status when an agent finishes responding."""
    verb = "reviewed" if is_critic else "responded"
    has_tool_calls = False
    # Check if the response had tool calls (will be logged separately by tool execution)
    # We still log the agent, but note the tool call
    print(f"   {agent_title} {verb}")


def _append_team_critic_guidance(prompt: str, state: MeetingState, *, for_team_lead: bool) -> str:
    """Append explicit critic-resolution instructions for the next team turn."""
    if not state.get("critic_feedback_pending", False):
        return prompt

    critic_name = state.get("critic_name") or "the critic"
    if for_team_lead:
        guidance = (
            f"\n\nBefore responding, explicitly address {critic_name}'s most recent feedback. "
            "Resolve each blocking point or briefly justify why it does not apply."
        )
    else:
        guidance = (
            f"\n\nAddress {critic_name}'s most recent feedback where it is relevant to your role. "
            "If you disagree, explain why."
        )
    return prompt + guidance


def _append_individual_summary_instructions(prompt: str, state: IndividualMeetingState) -> str:
    """Append final-output instructions/checklist for individual meetings when provided."""
    summary_instructions = state.get("summary_instructions", "").strip()
    if not summary_instructions:
        return prompt
    return prompt + f"\n\n{summary_instructions}"


def _individual_no_critic_prompt(
    agent: Agent,
    *,
    is_final_round: bool,
    state: IndividualMeetingState,
) -> str:
    """Build the main-agent prompt for individual meetings without a critic."""
    if not is_final_round:
        return f"{agent.title}, please continue addressing the agenda and make progress toward your goal."

    prompt = (
        f"{agent.title}, this is your FINAL response. "
        "Summarize what was accomplished, report the key findings with exact values, provide your recommendation, "
        "and list the next steps. "
    )
    if "coding" in agent.title.lower():
        prompt += "Do not write more code unless there is an actual error or a required output is still missing."
    else:
        prompt += "Do not re-run tools unless there is an actual error or missing required output."
    return _append_individual_summary_instructions(prompt, state)


def _individual_post_execution_prompt(
    agent: Agent,
    *,
    is_final_round: bool,
    state: IndividualMeetingState,
) -> str:
    """Build a compact follow-up prompt after tool/code execution."""
    prompt = (
        f"{agent.title}, review the latest execution results and decide whether to debug or finalize. "
        "Only write new code or re-run tools if there is an actual error or a required output is still missing."
    )
    if is_final_round:
        prompt += " This is your FINAL response if all required outputs are now present."
        prompt = _append_individual_summary_instructions(prompt, state)
    return prompt


def create_agent_node(
    agent: Agent,
    model: Any,
    prompt_override: str = None,
) -> callable:
    """Factory function to create an agent node for the graph.

    Args:
        agent: The agent that this node represents.
        model: The LangChain model instance (with or without tools bound).
        prompt_override: Optional custom prompt to use instead of agent.prompt.

    Returns:
        A node function that can be added to a LangGraph StateGraph.

    Example:
        >>> agent = Agent("Scientist", "research", "discover", "lead", "llama3.1")
        >>> model = ChatOllama(model="llama3.1")
        >>> node = create_agent_node(agent, model)
        >>> # Add to graph: graph.add_node("scientist", node)
    """

    def agent_node(state: MeetingState | IndividualMeetingState) -> dict[str, Any]:
        """Node function that invokes the agent and updates state.

        Args:
            state: Current meeting state from the graph.

        Returns:
            Dictionary with updated state fields (messages, tool_outputs, etc.).
        """
        # Build messages list from state
        messages = list(state["messages"])

        # Add system message with agent's prompt
        system_prompt = prompt_override if prompt_override else agent.prompt
        messages_with_system = [SystemMessage(content=system_prompt)] + messages

        # Invoke the model
        response = model.invoke(messages_with_system)
        _log_agent_response(agent.title, state)

        # Check if response contains tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Tool calls are handled by a separate ToolNode in the graph
            # For now, just return the response message
            return {"messages": [response]}

        # Return updated state with agent's response
        return {"messages": [response]}

    return agent_node


def create_team_lead_node(
    agent: Agent,
    model: Any,
    prompt_override: str = None,
) -> callable:
    """Create a team lead node with round-aware prompting.

    The team lead has different prompts depending on the round:
    - Round 0: Initial prompt to kick off discussion
    - Middle rounds: Intermediate prompt to guide discussion
    - Final round: Summary prompt to synthesize findings

    Args:
        agent: The team lead agent.
        model: The LangChain model for the team lead.
        prompt_override: Optional custom prompt prefix.

    Returns:
        A node function for the team lead.
    """

    def team_lead_node(state: MeetingState) -> dict[str, Any]:
        """Team lead node with round-aware prompting."""
        messages = list(state["messages"])
        current_round = state["current_round"]
        max_rounds = state["max_rounds"]
        is_final_round = state["is_final_round"]

        # Check if last message is a tool result
        last_msg = messages[-1] if messages else None
        is_tool_result = (
            isinstance(last_msg, HumanMessage) and
            last_msg.content and
            ("Tool '" in last_msg.content or "Results:" in last_msg.content)
        )

        # Get verbosity config from state
        verbosity = get_verbosity_config(state.get("prompt_verbosity", "verbose"))

        if is_tool_result:
            # Responding to tool results - don't add new prompt
            system_prompt = prompt_override if prompt_override else agent.prompt
            messages_with_system = [SystemMessage(content=system_prompt)] + messages

            response = model.invoke(messages_with_system)
            response.name = agent.title
            _log_agent_response(agent.title, state)

            return {"messages": [response]}
        else:
            # Normal turn - determine which prompt to use based on round
            if current_round == 0:
                # Initial round
                prompt = team_meeting_team_lead_initial_prompt(team_lead=agent)
            elif is_final_round:
                # Final summary round - pass verbosity for re-injection control
                prompt = team_meeting_team_lead_final_prompt(
                    team_lead=agent,
                    agenda=state["agenda"],
                    agenda_questions=state["agenda_questions"],
                    agenda_rules=state["agenda_rules"],
                    verbosity=verbosity,
                    summary_instructions=state.get("summary_instructions", ""),
                )
            else:
                # Intermediate rounds
                prompt = team_meeting_team_lead_intermediate_prompt(
                    team_lead=agent,
                    round_num=current_round,
                    num_rounds=max_rounds,
                )

            prompt = _append_team_critic_guidance(
                prompt,
                state,
                for_team_lead=True,
            )

            # Add the round-specific prompt as a user message
            messages.append(HumanMessage(content=prompt))

            # Build full message list with system prompt
            system_prompt = prompt_override if prompt_override else agent.prompt
            messages_with_system = [SystemMessage(content=system_prompt)] + messages

            # Invoke model
            response = model.invoke(messages_with_system)

            # Set team lead name on the response
            response.name = agent.title
            _log_agent_response(agent.title, state)

            return {"messages": [HumanMessage(content=prompt), response]}

    return team_lead_node


def create_team_member_node(
    agent: Agent,
    model: Any,
    prompt_override: str = None,
) -> callable:
    """Create a team member node with round-aware prompting.

    Args:
        agent: The team member agent.
        model: The LangChain model for the team member.
        prompt_override: Optional custom prompt prefix.

    Returns:
        A node function for the team member.
    """

    def team_member_node(state: MeetingState) -> dict[str, Any]:
        """Team member node with round-aware prompting."""
        messages = list(state["messages"])
        current_round = state["current_round"]
        max_rounds = state["max_rounds"]

        # Check if last message is a tool/code execution result (HumanMessage containing results)
        # If so, don't add a new prompt - the agent should respond to the results
        last_msg = messages[-1] if messages else None
        is_tool_result = (
            isinstance(last_msg, HumanMessage) and
            last_msg.content and
            ("Tool '" in last_msg.content or
             "Result:" in last_msg.content or  # Matches "Execution Result:" and "Results:"
             "Code Executed Successfully" in last_msg.content or  # Code execution success
             "Code Execution Failed" in last_msg.content)  # Code execution failure
        )

        # Get verbosity config from state
        verbosity = get_verbosity_config(state.get("prompt_verbosity", "verbose"))

        if is_tool_result:
            # Responding to tool results - don't add new prompt
            system_prompt = prompt_override if prompt_override else agent.prompt
            messages_with_system = [SystemMessage(content=system_prompt)] + messages

            # Invoke model
            response = model.invoke(messages_with_system)
            response.name = agent.title
            _log_agent_response(agent.title, state)

            return {"messages": [response]}
        else:
            # Normal turn - add prompt with verbosity control
            prompt = team_meeting_team_member_prompt(
                team_member=agent,
                round_num=current_round + 1,  # Display as 1-indexed
                num_rounds=max_rounds,
                verbosity=verbosity,
            )
            prompt = _append_team_critic_guidance(
                prompt,
                state,
                for_team_lead=False,
            )

            # Add prompt as user message
            messages.append(HumanMessage(content=prompt))

            # Build full message list
            system_prompt = prompt_override if prompt_override else agent.prompt
            messages_with_system = [SystemMessage(content=system_prompt)] + messages

            # Invoke model
            response = model.invoke(messages_with_system)
            response.name = agent.title
            _log_agent_response(agent.title, state)

            return {"messages": [HumanMessage(content=prompt), response]}

    return team_member_node


def create_individual_agent_node(
    agent: Agent,
    model: Any,
    critic_agent: Agent,
    prompt_override: str = None,
    enable_critic: bool = True,
) -> callable:
    """Create a node for the main agent in an individual meeting.

    Args:
        agent: The main agent.
        model: The LangChain model for the agent.
        critic_agent: The critic agent (for context in prompts).
        prompt_override: Optional custom prompt.
        enable_critic: Whether to include critic feedback in prompts (default True).

    Returns:
        A node function for the individual meeting agent.
    """

    def individual_agent_node(state: IndividualMeetingState) -> dict[str, Any]:
        """Individual meeting agent node."""
        messages = list(state["messages"])
        current_round = state["current_round"]
        tools_just_executed = state.get("tools_just_executed", False)
        appended_prompt = False

        if current_round == 0:
            # First round - use start prompt (already added by initialization)
            pass
        else:
            is_final = (current_round == state["max_rounds"])

            if tools_just_executed:
                if is_final:
                    prompt = _individual_post_execution_prompt(
                        agent,
                        is_final_round=is_final,
                        state=state,
                    )
                    messages.append(HumanMessage(content=prompt))
                    appended_prompt = True
            else:
                if enable_critic:
                    prompt = individual_meeting_agent_prompt(
                        critic=critic_agent,
                        agent=agent,
                        is_final_round=is_final,
                        summary_instructions=state.get("summary_instructions", ""),
                    )
                else:
                    prompt = _individual_no_critic_prompt(
                        agent,
                        is_final_round=is_final,
                        state=state,
                    )
                messages.append(HumanMessage(content=prompt))
                appended_prompt = True

        # Build message list
        system_prompt = prompt_override if prompt_override else agent.prompt
        messages_with_system = [SystemMessage(content=system_prompt)] + messages

        # Invoke model
        response = model.invoke(messages_with_system)

        # Set agent name on the response for proper display
        response.name = agent.title
        _log_agent_response(agent.title, state)

        if current_round == 0:
            return {
                "messages": [response],
                "tools_just_executed": False  # Clear flag after agent responds
            }

        if appended_prompt:
            prompt = messages[-1].content
            return {
                "messages": [HumanMessage(content=prompt), response],
                "tools_just_executed": False  # Clear flag after agent responds
            }

        return {
            "messages": [response],
            "tools_just_executed": False  # Clear flag after agent responds
        }

    return individual_agent_node


def create_critic_node(
    critic: Agent,
    model: Any,
    main_agent: Agent,
) -> callable:
    """Create a node for the critic in an individual meeting.

    Args:
        critic: The critic agent.
        model: The LangChain model for the critic.
        main_agent: The main agent being critiqued.

    Returns:
        A node function for the critic.
    """

    def critic_node(state: IndividualMeetingState) -> dict[str, Any]:
        """Critic node for individual meetings."""
        messages = list(state["messages"])

        # Generate critic prompt
        prompt = individual_meeting_critic_prompt(
            critic=critic,
            agent=main_agent,
        )

        messages.append(HumanMessage(content=prompt))

        # Build message list
        messages_with_system = [SystemMessage(content=critic.prompt)] + messages

        # Invoke model
        response = model.invoke(messages_with_system)

        # Set critic name on the response
        response.name = critic.title
        _log_agent_response(critic.title, state, is_critic=True)

        return {"messages": [HumanMessage(content=prompt), response]}

    return critic_node


def create_team_critic_node(
    critic: Agent,
    model: Any,
    team_lead: Agent,
) -> callable:
    """Create a node for the critic in a team meeting.

    The critic reviews the team discussion after each round and provides feedback.

    Args:
        critic: The critic agent.
        model: The LangChain model for the critic.
        team_lead: The team lead agent.

    Returns:
        A node function for the critic.
    """

    def team_critic_node(state: MeetingState) -> dict[str, Any]:
        """Critic node for team meetings."""
        from veritas.prompts import team_meeting_critic_prompt

        messages = list(state["messages"])

        # Generate critic prompt - review the team discussion
        prompt = team_meeting_critic_prompt(
            critic=critic,
            team_lead=team_lead,
        )

        messages.append(HumanMessage(content=prompt))

        # Build message list
        messages_with_system = [SystemMessage(content=critic.prompt)] + messages

        # Invoke model
        response = model.invoke(messages_with_system)

        # Set critic name on the response
        response.name = critic.title
        _log_agent_response(critic.title, state, is_critic=True)

        return {
            "messages": [HumanMessage(content=prompt), response],
            "critic_feedback_pending": True,
        }

    return team_critic_node
