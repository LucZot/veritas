"""Individual meeting graph builder for LangGraph workflows.

Builds a state graph for single-agent meetings with:
- Main agent
- Optional critic feedback
- Tool execution support (PubMed + MCP)
"""

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from veritas.agent import Agent, resolve_temperature, resolve_top_p
from veritas.config import get_provider_type
from veritas.graph_state import IndividualMeetingState
from veritas.langchain_utils import create_chat_model
from veritas.graph import (
    create_individual_agent_node,
    create_critic_node,
    create_tool_execution_node,
    increment_round_node,
    should_execute_tools,
    should_execute_code,
    should_continue_discussion,
    should_critic_respond,
)
from veritas.graph.code_execution import create_code_execution_node


def build_individual_meeting_graph(
    agent: Agent,
    critic: Agent,
    ollama_config: dict,
    temperature: float,
    top_p: float | None = None,
    enable_checkpointing: bool = False,
    enable_critic: bool = True,
) -> StateGraph:
    """Build a LangGraph StateGraph for individual meetings.

    The graph structure:
    - With critic: START → agent → critic → increment_round → [continue or end]
    - Without critic: START → agent → increment_round → [continue or end]

    Args:
        agent: Main agent
        critic: Critic agent (only used if enable_critic=True)
        ollama_config: Ollama configuration
        temperature: Sampling temperature
        enable_checkpointing: Whether to enable state persistence
        enable_critic: Whether to include critic in the workflow

    Returns:
        Compiled StateGraph application
    """
    # Create graph
    graph = StateGraph(IndividualMeetingState)

    # Create models using factory function (per-agent temperature/top_p with global fallback)
    agent_model_name = agent.model or ollama_config["model"]
    agent_model = create_chat_model(
        model_name=agent_model_name,
        temperature=resolve_temperature(agent, temperature),
        top_p=resolve_top_p(agent, top_p),
        ollama_config=ollama_config,
    )
    critic_model_name = critic.model or ollama_config["model"]
    # Critic never uses tools — wrap function-calling models to recover from
    # Ollama tool-call parsing errors on plain text responses
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

    # Build unified tool registry (LangChain + MCP)
    from veritas.tools import write_file, read_file, list_files
    from veritas.vision.datasets.dataset_tools import get_dataset_patient_info, list_dataset_patients

    # Start with LangChain tools
    tool_registry = {
        "write_file": write_file,
        "read_file": read_file,
        "list_files": list_files,
        "get_dataset_patient_info": get_dataset_patient_info,
        "list_dataset_patients": list_dataset_patients,
    }

    # Add MCP tools to unified registry
    from veritas.mcp import discover_mcp_tools_for_agents
    mcp_result = discover_mcp_tools_for_agents([agent])
    mcp_tool_info = mcp_result['mcp_tool_info']
    tool_registry.update(mcp_result['tool_registry'])

    if mcp_tool_info:
        tool_names = list(mcp_tool_info.keys())
        print(f"🔧 Discovered {len(tool_names)} MCP tools for {agent.title}: {', '.join(tool_names)}")

    # Bind tools to agent model
    from veritas.tool_binding import bind_tools_to_agent

    agent_model, agent_prompt = bind_tools_to_agent(
        agent=agent,
        ollama_config=ollama_config,
        temperature=temperature,
        top_p=top_p,
        tool_registry=tool_registry,
        mcp_tool_info=mcp_tool_info,
    )

    # Create nodes
    agent_node = create_individual_agent_node(agent, agent_model, critic, agent_prompt, enable_critic=enable_critic)

    graph.add_node("agent", agent_node)
    graph.add_node("increment_round", increment_round_node)

    # Only add critic node if enabled
    if enable_critic:
        critic_node = create_critic_node(critic, critic_model, agent)
        graph.add_node("critic", critic_node)

    # Determine execution mode: tools, code-as-output, or neither
    has_tools = hasattr(agent, 'available_tools') and agent.available_tools
    is_code_output_mode = hasattr(agent, 'available_tools') and not agent.available_tools

    # Add execution nodes
    if has_tools:
        tool_node = create_tool_execution_node(agent=agent)
        graph.add_node("execute_tools", tool_node)

    if is_code_output_mode:
        code_node = create_code_execution_node(agent=agent)
        graph.add_node("execute_code", code_node)

    # Connect nodes
    graph.add_edge(START, "agent")

    # Conditional routing after agent
    if has_tools:
        # First check for tool calls
        graph.add_conditional_edges(
            "agent",
            should_execute_tools,
            {
                "execute_tools": "execute_tools",  # Has tool calls → execute them
                "continue": "agent_continue",  # No tool calls → check if critic should respond
            }
        )

        # After tool execution, loop back to agent to process results
        graph.add_edge("execute_tools", "agent")

        # Add a passthrough node for routing after tool check
        if enable_critic:
            def agent_continue_router(state):
                return should_critic_respond(state)

            graph.add_conditional_edges(
                "agent_continue",
                agent_continue_router,
                {
                    "critic": "critic",
                    "skip_critic": "increment_round",
                }
            )
        else:
            # No critic: go directly to increment_round
            graph.add_edge("agent_continue", "increment_round")

        # Add passthrough node
        graph.add_node("agent_continue", lambda state: {})

    elif is_code_output_mode:
        # Code-as-output mode: check for code blocks instead of tool calls
        graph.add_conditional_edges(
            "agent",
            should_execute_code,
            {
                "execute_code": "execute_code",  # Has code blocks → execute them
                "continue": "agent_continue",  # No code → check if critic should respond
            }
        )

        # After code execution, loop back to agent to process results
        graph.add_edge("execute_code", "agent")

        # Add a passthrough node for routing after code check
        if enable_critic:
            def agent_continue_router(state):
                return should_critic_respond(state)

            graph.add_conditional_edges(
                "agent_continue",
                agent_continue_router,
                {
                    "critic": "critic",
                    "skip_critic": "increment_round",
                }
            )
        else:
            # No critic: go directly to increment_round
            graph.add_edge("agent_continue", "increment_round")

        # Add passthrough node
        graph.add_node("agent_continue", lambda state: {})

    else:
        # No tools or code execution
        if enable_critic:
            # Original behavior: skip critic in final round
            graph.add_conditional_edges(
                "agent",
                should_critic_respond,
                {
                    "critic": "critic",  # Normal flow: agent → critic
                    "skip_critic": "increment_round",  # Final round: skip critic
                }
            )
        else:
            # No critic: go directly to increment_round
            graph.add_edge("agent", "increment_round")

    # Add critic edge only if critic is enabled
    if enable_critic:
        graph.add_edge("critic", "increment_round")

    # Conditional routing after round increment
    graph.add_conditional_edges(
        "increment_round",
        should_continue_discussion,
        {
            "continue": "agent",  # Loop back for another round
            "end": END,  # Terminate
        }
    )

    # Compile
    checkpointer = MemorySaver() if enable_checkpointing else None
    return graph.compile(checkpointer=checkpointer)
