"""LangGraph node and routing functions for VERITAS agent meetings.

This package provides a modular structure for graph-based meeting workflows:
- nodes: Agent node creation functions
- routing: Conditional edge and decision functions
- tool_execution: Tool execution node with MCP support
- utils: Helper utilities
"""

# Node creation functions
from veritas.graph.nodes import (
    create_agent_node,
    create_team_lead_node,
    create_team_member_node,
    create_individual_agent_node,
    create_critic_node,
    create_team_critic_node,
)

# Routing functions
from veritas.graph.routing import (
    should_execute_tools,
    should_execute_code,
    should_continue_discussion,
    should_critic_respond,
    should_critic_review_team,
    check_agenda_completion,
    route_after_team_lead,
    route_after_tools,
    route_after_code_execution,
)

# Tool execution
from veritas.graph.tool_execution import (
    create_tool_execution_node,
)

# Utilities
from veritas.graph.utils import (
    increment_round_node,
)

__all__ = [
    # Nodes
    "create_agent_node",
    "create_team_lead_node",
    "create_team_member_node",
    "create_individual_agent_node",
    "create_critic_node",
    "create_team_critic_node",
    # Routing
    "should_execute_tools",
    "should_execute_code",
    "should_continue_discussion",
    "should_critic_respond",
    "should_critic_review_team",
    "check_agenda_completion",
    "route_after_team_lead",
    "route_after_tools",
    "route_after_code_execution",
    # Tool execution
    "create_tool_execution_node",
    # Utils
    "increment_round_node",
]
