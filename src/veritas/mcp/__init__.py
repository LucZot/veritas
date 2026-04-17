"""Model Context Protocol (MCP) integration.

This package provides MCP client management, configuration loading,
and tool discovery/wrapping for LangChain/LangGraph compatibility.

Modules:
- client: MCPClientManager for connecting to MCP servers
- config: Loading mcp_servers.json configuration
- tools: Discovering and converting MCP tools to LangChain format
"""

from veritas.mcp.client import MCPClientManager
from veritas.mcp.config import load_mcp_config
from veritas.mcp.tools import (
    discover_mcp_tools_for_agents,
    mcp_schema_to_pydantic,
    create_mcp_tool_wrapper,
    build_tool_documentation,
)

__all__ = [
    # Client
    "MCPClientManager",
    # Config
    "load_mcp_config",
    # Tools
    "discover_mcp_tools_for_agents",
    "mcp_schema_to_pydantic",
    "create_mcp_tool_wrapper",
    "build_tool_documentation",
]
