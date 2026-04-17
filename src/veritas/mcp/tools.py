"""MCP Tool Discovery and Integration.

This module handles the discovery, registration, and LangChain conversion of
Model Context Protocol (MCP) tools for both team and individual meetings.
"""

import asyncio
from typing import Any, Dict, List, Sequence

from langchain_core.tools import Tool
from pydantic import BaseModel

from veritas.agent import Agent
from veritas.utils.schema_conversion import json_schema_to_pydantic, tool_name_to_model_name


# ============================================================================
# Internal Helpers
# ============================================================================

def _collect_mcp_servers(agents: Sequence[Agent]) -> set[str]:
    """Collect unique MCP server names from all agents."""
    servers = set()
    for agent in agents:
        if hasattr(agent, 'mcp_servers') and agent.mcp_servers:
            servers.update(agent.mcp_servers)
    return servers


async def _discover_tools_from_servers(server_names: set[str]) -> Dict[str, Any]:
    """Async discovery of tools from specified MCP servers."""
    try:
        from veritas.mcp import MCPClientManager, load_mcp_config
    except ImportError:
        print("Warning: MCP client not available")
        return {}

    mcp_config = load_mcp_config()
    server_lookup = {s['name']: s for s in mcp_config.get('servers', [])}

    manager = MCPClientManager()
    all_tools = {}

    for server_name in server_names:
        if server_name not in server_lookup:
            print(f"Warning: MCP server '{server_name}' not found in mcp_servers.json")
            continue

        manager.register_server(server_name, server_lookup[server_name])
        tools = await manager.list_tools(server_name)

        for tool in tools:
            tool_name = tool.get('name', 'unknown')
            all_tools[tool_name] = {
                'server': server_name,
                'info': tool,
                'type': 'mcp'
            }

    return all_tools


def _convert_to_langchain_tools(mcp_tool_info: Dict[str, Any]) -> Dict[str, Tool]:
    """Convert MCP tools to LangChain Tool objects."""
    tool_registry = {}
    for tool_name, tool_data in mcp_tool_info.items():
        tool_description = tool_data['info'].get('description', f'MCP tool: {tool_name}')
        input_schema = tool_data['info'].get('inputSchema', {})

        pydantic_schema = mcp_schema_to_pydantic(tool_name, input_schema)
        tool_registry[tool_name] = create_mcp_tool_wrapper(
            tool_name, tool_description, pydantic_schema
        )
    return tool_registry


# ============================================================================
# Public API
# ============================================================================

def discover_mcp_tools_for_agents(agents: Sequence[Agent]) -> Dict[str, Any]:
    """Discover and register MCP tools from agent configurations.

    Handles the complete MCP tool discovery workflow: connects to servers,
    discovers tools, converts schemas to Pydantic, creates LangChain wrappers.

    Args:
        agents: Sequence of agents with optional mcp_servers attribute.

    Returns:
        Dictionary with 'mcp_tool_info' and 'tool_registry' keys.
    """
    server_names = _collect_mcp_servers(agents)
    if not server_names:
        return {'mcp_tool_info': {}, 'tool_registry': {}}

    try:
        mcp_tool_info = asyncio.run(_discover_tools_from_servers(server_names))
        tool_registry = _convert_to_langchain_tools(mcp_tool_info)
    except Exception as e:
        print(f"Warning: Could not initialize MCP tools: {e}")
        return {'mcp_tool_info': {}, 'tool_registry': {}}

    return {'mcp_tool_info': mcp_tool_info, 'tool_registry': tool_registry}


def mcp_schema_to_pydantic(tool_name: str, mcp_schema: dict) -> type[BaseModel]:
    """Convert MCP JSON Schema to Pydantic BaseModel for LangChain compatibility.

    Args:
        tool_name: Name of the tool (used for model naming)
        mcp_schema: MCP JSON Schema dict with properties, required fields, etc.

    Returns:
        Dynamically created Pydantic BaseModel class
    """
    model_name = tool_name_to_model_name(tool_name, suffix="Input")
    return json_schema_to_pydantic(mcp_schema, model_name)


def create_mcp_tool_wrapper(
    tool_name: str,
    description: str,
    pydantic_schema: type[BaseModel]
) -> Tool:
    """Create LangChain-compatible tool wrapper for MCP tool.

    Args:
        tool_name: Name of the MCP tool
        description: Human-readable description for the model
        pydantic_schema: Pydantic model defining tool parameters

    Returns:
        LangChain Tool object
    """
    def mcp_placeholder_func(**kwargs):
        """Placeholder - actual execution in tool_execution_node."""
        return f"MCP tool {tool_name} will be executed by tool_execution_node"

    return Tool(
        name=tool_name,
        description=description,
        func=mcp_placeholder_func,
        args_schema=pydantic_schema
    )


def build_tool_documentation(
    mcp_tool_info: Dict[str, Any],
    tool_names: List[str]
) -> str:
    """Build detailed tool documentation for agent prompts.

    Args:
        mcp_tool_info: MCP tool metadata from discover_mcp_tools_for_agents()
        tool_names: List of tool names to document

    Returns:
        Formatted documentation string
    """
    tool_docs = []

    for tool_name in tool_names:
        if tool_name not in mcp_tool_info:
            continue

        tool_data = mcp_tool_info[tool_name]
        tool_info = tool_data['info']
        schema = tool_info.get('inputSchema', {})

        doc = f"\n**{tool_name}**"
        doc += f"\n  Description: {tool_info.get('description', 'N/A')}"

        props = schema.get('properties', {})
        required = schema.get('required', [])

        if props:
            doc += "\n  Parameters:"
            for param, details in props.items():
                req_marker = " (REQUIRED)" if param in required else " (optional)"
                doc += f"\n    - {param}{req_marker}: {details.get('description', 'N/A')}"

                if 'enum' in details:
                    doc += f"\n      Valid values: {', '.join(repr(v) for v in details['enum'])}"
                if 'default' in details:
                    doc += f"\n      Default: {repr(details['default'])}"

        tool_docs.append(doc)

    return "".join(tool_docs)
