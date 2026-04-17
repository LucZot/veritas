"""Team meeting graph builder for LangGraph workflows.

Builds a state graph for team-based meetings with:
- Team lead coordination
- Multiple team members
- Optional critic review
- Tool execution support (PubMed + MCP)
"""

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from veritas.agent import Agent, resolve_temperature, resolve_top_p
from veritas.config import get_provider_type
from veritas.graph_state import MeetingState
from veritas.langchain_utils import create_chat_model
from veritas.graph import (
    create_team_lead_node,
    create_team_member_node,
    create_team_critic_node,
    create_tool_execution_node,
    increment_round_node,
    should_execute_tools,
    should_execute_code,
    should_continue_discussion,
    should_critic_review_team,
    route_after_tools,
    route_after_code_execution,
    route_after_team_lead,
)
from veritas.graph.code_execution import create_code_execution_node


def agent_has_tools(agent: Agent, ollama_config: dict) -> bool:
    """Check if a specific agent will have tools bound in their model.

    Args:
        agent: Agent to check
        ollama_config: Ollama config dict

    Returns:
        True if agent will have tools bound, False otherwise
    """
    from veritas.tool_binding import _model_supports_tools

    model_name = agent.model or ollama_config.get("model", "")
    supports_tools = _model_supports_tools(model_name)
    return bool(hasattr(agent, "available_tools") and agent.available_tools and supports_tools)


def agent_is_code_output_mode(agent: Agent) -> bool:
    """Check if agent is in code-as-output mode.

    Code-as-output agents write Python code in markdown blocks instead of using tools.
    They have empty available_tools list and no MCP servers.

    Args:
        agent: Agent to check

    Returns:
        True if agent is in code-as-output mode, False otherwise
    """
    # Check if agent has explicitly empty tools list
    has_empty_tools = hasattr(agent, "available_tools") and agent.available_tools == []

    # Check if agent has no MCP servers
    has_no_mcp = not hasattr(agent, "mcp_servers") or not agent.mcp_servers

    # Code-as-output mode: no tools AND no MCP servers
    return has_empty_tools and has_no_mcp


def build_team_meeting_graph(
    team_lead: Agent,
    team_members: tuple[Agent, ...],
    ollama_config: dict,
    temperature: float,
    top_p: float | None = None,
    enable_checkpointing: bool = False,
    critic: Agent = None,
    enable_critic: bool = True,
) -> StateGraph:
    """Build a LangGraph StateGraph for team meetings.

    The graph structure (with critic):
    START → team_lead_initial → team_member_1 → ... → team_member_N
          → increment_round → [if not final] → critic → team_lead_intermediate
          → ... (repeat rounds) → team_lead_final → END

    The graph structure (without critic):
    START → team_lead_initial → team_member_1 → ... → team_member_N
          → increment_round → [continue or end]
          → team_lead_intermediate → ... (repeat rounds)
          → team_lead_final → END

    Args:
        team_lead: Team lead agent
        team_members: Team member agents
        ollama_config: Ollama configuration dict
        critic: Optional critic agent to review discussions
        enable_critic: Whether to include critic in workflow
        temperature: Sampling temperature
        enable_checkpointing: Whether to enable state persistence

    Returns:
        Compiled StateGraph application
    """
    # Create graph
    graph = StateGraph(MeetingState)

    # Build unified tool registry (LangChain + MCP) - shared by all team members
    from veritas.tools import write_file, read_file, list_files
    from veritas.vision.datasets.dataset_tools import get_dataset_patient_info, list_dataset_patients
    from veritas.mcp import discover_mcp_tools_for_agents

    tool_registry = {
        "write_file": write_file,
        "read_file": read_file,
        "list_files": list_files,
        "get_dataset_patient_info": get_dataset_patient_info,
        "list_dataset_patients": list_dataset_patients,
    }

    # Discover MCP tools from all agents
    all_agents = [team_lead] + list(team_members)
    mcp_result = discover_mcp_tools_for_agents(all_agents)
    mcp_tool_info = mcp_result['mcp_tool_info']
    tool_registry.update(mcp_result['tool_registry'])

    if mcp_tool_info:
        tool_names = list(mcp_tool_info.keys())
        print(f"🔧 Discovered {len(tool_names)} MCP tools: {', '.join(tool_names)}")

    # Bind tools to agent models
    from veritas.tool_binding import bind_tools_to_agents

    agent_models, agent_prompts = bind_tools_to_agents(
        agents=all_agents,
        ollama_config=ollama_config,
        temperature=temperature,
        top_p=top_p,
        tool_registry=tool_registry,
        mcp_tool_info=mcp_tool_info,
    )

    # Add team lead node
    team_lead_node = create_team_lead_node(
        team_lead,
        agent_models[team_lead],
        agent_prompts.get(team_lead),
    )
    graph.add_node("team_lead", team_lead_node)

    # Add team member nodes
    for i, member in enumerate(team_members):
        node_name = f"member_{i}_{member.title.replace(' ', '_').lower()}"
        member_node = create_team_member_node(
            member,
            agent_models[member],
            agent_prompts.get(member),
        )
        graph.add_node(node_name, member_node)

    # Add utility nodes
    graph.add_node("increment_round", increment_round_node)

    # Add critic node if enabled
    if enable_critic and critic:
        # Critic never uses tools — wrap function-calling models to recover from
        # Ollama tool-call parsing errors on plain text responses
        critic_model_name = critic.model or ollama_config["model"]
        provider_type = get_provider_type()
        from veritas.tool_binding import _model_supports_tools
        from veritas.tool_binding import ToolCallSafeModel
        critic_is_fc_model = _model_supports_tools(critic_model_name, provider_type)
        critic_model = create_chat_model(
            model_name=critic_model_name,
            temperature=resolve_temperature(critic, temperature),
            top_p=resolve_top_p(critic, top_p),
            ollama_config=ollama_config,
        )
        if provider_type == "ollama" and critic_is_fc_model:
            critic_model = ToolCallSafeModel(critic_model)
        # Use team-specific critic node that reviews entire team discussion
        critic_node = create_team_critic_node(critic, critic_model, team_lead)
        graph.add_node("critic", critic_node)

    # Check if any agent has tools (for execute_tools node)
    has_any_tools = any(
        agent_has_tools(agent, ollama_config)
        for agent in all_agents
    )

    # If tools enabled, add tool execution node with proper agent context
    if has_any_tools:
        # Create agent map for tool execution node to lookup agents
        agent_name_map = {agent.title: agent for agent in all_agents}
        # Create tool node that can handle tools from any agent
        tool_node = create_tool_execution_node(agent_map=agent_name_map)
        graph.add_node("execute_tools", tool_node)

    # Check if any agent is in code-as-output mode
    has_code_output_agents = any(
        agent_is_code_output_mode(agent)
        for agent in all_agents
    )

    # If code-as-output agents exist, add code execution node
    if has_code_output_agents:
        # Create agent map for code execution node
        agent_name_map = {agent.title: agent for agent in all_agents}
        # Create code execution node that can handle code from any agent
        code_node = create_code_execution_node(agent_map=agent_name_map)
        graph.add_node("execute_code", code_node)
        print(f"🔧 Added code execution node for code-as-output agents")

    # Connect nodes with conditional tool routing
    # START → team_lead
    graph.add_edge(START, "team_lead")

    # Check if THIS agent (team_lead) has tools
    team_lead_has_tools = agent_has_tools(team_lead, ollama_config)

    # Always create continue node for team_lead
    # Purpose: routing junction that handles BOTH tool execution AND round decisions
    # Even if team_lead has no tools, the continue node is needed for:
    # - Deciding if meeting is over (is_final_round? → END)
    # - Routing to next phase (members or increment_round)
    graph.add_node("team_lead_continue", lambda state: {})

    # Routing FROM team_lead TO continue node
    if team_lead_has_tools:
        # Team lead CAN use tools - check for tool calls
        graph.add_conditional_edges(
            "team_lead",
            should_execute_tools,
            {
                "execute_tools": "execute_tools",  # Has tool calls → execute them
                "continue": "team_lead_continue",  # No tool calls → go to continue for round decision
            }
        )
    else:
        # Team lead cannot use tools - direct to continue for round decision
        graph.add_edge("team_lead", "team_lead_continue")

    # Routing FROM continue node - decide what to do based on meeting progress
    if team_members:
        first_member_node = f"member_0_{team_members[0].title.replace(' ', '_').lower()}"
        graph.add_conditional_edges(
            "team_lead_continue",
            route_after_team_lead,
            {
                "end": END,
                first_member_node: first_member_node,
                "increment_round": "increment_round",
            }
        )
    else:
        graph.add_conditional_edges(
            "team_lead_continue",
            route_after_team_lead,
            {
                "end": END,
                "increment_round": "increment_round",
            }
        )

    # Chain team members with tool execution and code execution support
    if team_members:
        for i in range(len(team_members)):
            current_node = f"member_{i}_{team_members[i].title.replace(' ', '_').lower()}"
            member = team_members[i]

            # Check if THIS member has tools or is code-as-output
            member_has_tools = agent_has_tools(member, ollama_config)
            member_is_code_output = agent_is_code_output_mode(member)

            # Determine next destination (either next member or increment_round)
            if i < len(team_members) - 1:
                next_destination = f"member_{i+1}_{team_members[i+1].title.replace(' ', '_').lower()}"
            else:
                next_destination = "increment_round"

            if member_has_tools:
                # Member has tools - check for tool calls
                # Create continue node for this member
                continue_node = f"{current_node}_continue"
                graph.add_node(continue_node, lambda state: {})

                # Check if this member needs to execute tools
                graph.add_conditional_edges(
                    current_node,
                    should_execute_tools,
                    {
                        "execute_tools": "execute_tools",  # Has tool calls → execute them
                        "continue": continue_node,  # No tool calls → proceed to next
                    }
                )

                # From continue node, proceed to next member/round
                graph.add_edge(continue_node, next_destination)

            elif member_is_code_output:
                # Member is code-as-output - check for code blocks
                graph.add_conditional_edges(
                    current_node,
                    should_execute_code,
                    {
                        "execute_code": "execute_code",  # Has code blocks → execute them
                        "continue": next_destination,  # No code → proceed to next
                    }
                )

            else:
                # No tools, no code - direct routing
                graph.add_edge(current_node, next_destination)

    # Add conditional routing FROM execute_tools TO the agent that called the tool
    if has_any_tools:
        # Build routing map: only include agents that actually have tools
        tool_routing_map = {}

        # Add team lead if they have tools
        if team_lead_has_tools:
            tool_routing_map["team_lead"] = "team_lead"

        # Add team members that have tools
        for i, member in enumerate(team_members):
            if agent_has_tools(member, ollama_config):
                member_node = f"member_{i}_{member.title.replace(' ', '_').lower()}"
                tool_routing_map[member_node] = member_node

        graph.add_conditional_edges(
            "execute_tools",
            route_after_tools,
            tool_routing_map
        )

    # Add conditional routing FROM execute_code TO the agent that wrote the code
    if has_code_output_agents:
        # Build routing map: only include agents in code-as-output mode
        code_routing_map = {}

        # Check team lead
        if agent_is_code_output_mode(team_lead):
            code_routing_map["team_lead"] = "team_lead"

        # Add team members that are code-as-output
        for i, member in enumerate(team_members):
            if agent_is_code_output_mode(member):
                member_node = f"member_{i}_{member.title.replace(' ', '_').lower()}"
                code_routing_map[member_node] = member_node

        # Route from execute_code back to the agent
        graph.add_conditional_edges(
            "execute_code",
            route_after_code_execution,  # Uses last_code_caller to route back
            code_routing_map
        )

    # Conditional routing from increment_round
    if enable_critic and critic:
        # With critic: increment_round → check if should continue
        # If continuing, critic reviews the completed round before the next
        # team_lead turn (including the final summary round).
        def route_after_increment_with_critic(state):
            # First check if we should continue at all
            continue_decision = should_continue_discussion(state)
            if continue_decision == "end":
                return "end"
            # We're continuing - should critic review?
            critic_decision = should_critic_review_team(state)
            return critic_decision  # "critic" or "skip_critic"

        graph.add_conditional_edges(
            "increment_round",
            route_after_increment_with_critic,
            {
                "critic": "critic",        # Critic reviews the completed round
                "skip_critic": "team_lead",  # Past final round → team lead/end routing
                "end": END,                # Meeting complete
            }
        )
        # After critic reviews, go back to team_lead for next round
        graph.add_edge("critic", "team_lead")
    else:
        # Without critic: increment_round → continue to team_lead or end
        graph.add_conditional_edges(
            "increment_round",
            should_continue_discussion,
            {
                "continue": "team_lead",  # Loop back for another round
                "end": END,               # Terminate
            }
        )

    # Compile graph
    checkpointer = MemorySaver() if enable_checkpointing else None
    return graph.compile(checkpointer=checkpointer)
