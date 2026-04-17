"""MCP (Model Context Protocol) client for VERITAS.

Provides a client manager for connecting to MCP servers (SAT segmentation,
sandboxed code execution) and exposing their tools to VERITAS agents.

Uses a per-request connection pattern: a fresh MCP connection is opened for
each tool call, following the official MCP SDK pattern.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning(
        "MCP package not installed. Install with: pip install -e '.[mcp]'"
    )


class MCPClientManager:
    """Manages MCP server configurations and tool calls.

    Uses a per-request connection pattern: creates a fresh subprocess connection
    for each tool call. This follows the official MCP SDK design and avoids
    async context manager lifecycle issues.

    Attributes:
        server_configs: Dictionary mapping server names to their configurations
        _tool_cache: Cached tool listings to avoid repeated discovery
    """

    def __init__(self):
        """Initialize the MCP client manager."""
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP package not available. Install with: pip install -e '.[mcp]'"
            )

        self.server_configs: Dict[str, Dict[str, Any]] = {}
        self._tool_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.servers: Dict[str, Any] = {}  # Persistent connections for testing/debugging

        logger.info("MCPClientManager initialized (supports both per-request and persistent connections)")

    def register_server(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> None:
        """Register an MCP server configuration (no connection created yet).

        This is a synchronous operation that just stores the configuration.
        Actual connections are created on-demand when calling tools.

        Args:
            name: Unique identifier for this server
            config: Server configuration with keys:
                - command: The command to run (e.g., "python", "node")
                - args: List of arguments (e.g., ["vlm_server.py"])
                - env: Optional environment variables

        Raises:
            ValueError: If server name already registered

        Example:
            >>> manager.register_server("sat", {
            ...     "command": "python",
            ...     "args": ["/path/to/sat_server.py"],
            ...     "env": {"SAT_REPO_PATH": "/home/user/SAT"}
            ... })
        """
        if name in self.server_configs:
            raise ValueError(f"Server '{name}' already registered")

        env = config.get("env")
        if env is None:
            merged_env = os.environ.copy()
        else:
            merged_env = os.environ.copy()
            merged_env.update(env)
        config["env"] = merged_env

        self.server_configs[name] = config
        logger.info(f"Registered MCP server '{name}': {config['command']} {' '.join(config.get('args', []))}")

    async def connect_server(self, server_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Connect to an MCP server and maintain persistent connection.
        
        This method is useful for testing and debugging scenarios where you want
        to maintain a connection across multiple tool calls.
        
        Args:
            server_name: Name of the server to connect to
            config: Server configuration (if not already registered)
            
        Raises:
            ValueError: If server not registered and no config provided
            RuntimeError: If connection fails
        """
        if config:
            self.register_server(server_name, config)
        elif server_name not in self.server_configs:
            raise ValueError(f"Server '{server_name}' not registered. Provide config or call register_server() first.")
            
        config = self.server_configs[server_name]
        server_params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env=config.get("env"),
        )

        try:
            logger.info(f"Connecting to MCP server '{server_name}'...")
            _conn_timeout = int(os.environ.get("VERITAS_MCP_CONNECT_TIMEOUT", "60"))

            # Create persistent connection using proper context managers
            # Wrap in wait_for to prevent indefinite hangs if server crashes at startup
            stdio_context = stdio_client(server_params)
            read, write = await asyncio.wait_for(
                stdio_context.__aenter__(), timeout=_conn_timeout
            )

            session_context = ClientSession(read, write)
            session = await asyncio.wait_for(
                session_context.__aenter__(), timeout=_conn_timeout
            )
            await asyncio.wait_for(session.initialize(), timeout=_conn_timeout)
            
            # Store connection with context managers for proper cleanup
            self.servers[server_name] = {
                'session': session,
                'read': read,
                'write': write,
                'stdio_context': stdio_context,
                'session_context': session_context
            }
            
            logger.info(f"Successfully connected to MCP server '{server_name}'")
            
        except Exception as e:
            logger.error(f"Failed to connect to server '{server_name}': {e}")
            raise RuntimeError(f"Could not connect to server '{server_name}': {e}")

    async def disconnect_server(self, server_name: str) -> None:
        """Disconnect from a specific MCP server.
        
        Args:
            server_name: Name of the server to disconnect from
        """
        if server_name not in self.servers:
            logger.warning(f"Server '{server_name}' not connected")
            return
            
        try:
            server_info = self.servers[server_name]
            
            # Close session and stdio connection in proper order
            try:
                await server_info['session_context'].__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing session context: {e}")
                
            try:
                await server_info['stdio_context'].__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing stdio context: {e}")
            
            del self.servers[server_name]
            logger.info(f"Disconnected from MCP server '{server_name}'")
            
        except Exception as e:
            logger.error(f"Error disconnecting from server '{server_name}': {e}")
            # Still remove from servers dict even if cleanup failed
            if server_name in self.servers:
                del self.servers[server_name]

    async def disconnect_all(self) -> None:
        """Disconnect from all connected MCP servers."""
        for server_name in list(self.servers.keys()):
            await self.disconnect_server(server_name)

    async def discover_tools(self, server_name: str, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Discover tools available from an MCP server.

        Uses persistent connection if available, otherwise creates temporary connection.

        Args:
            server_name: Name of the server to query
            force_refresh: If True, bypass cache and rediscover tools

        Returns:
            List of tool definitions with 'name', 'description', 'inputSchema'

        Raises:
            ValueError: If server not registered
            RuntimeError: If tool discovery fails

        Example:
            >>> tools = await manager.discover_tools("sat")
            >>> for tool in tools:
            ...     print(f"{tool['name']}: {tool['description']}")
        """
        if server_name not in self.server_configs:
            raise ValueError(f"Server '{server_name}' not registered. Call register_server() first.")

        # Return cached tools if available
        if not force_refresh and server_name in self._tool_cache:
            logger.debug(f"Returning cached tools for server '{server_name}'")
            return self._tool_cache[server_name]

        # Check if we have a persistent connection
        if server_name in self.servers:
            # Use persistent connection
            try:
                logger.info(f"Discovering tools from MCP server '{server_name}' using persistent connection...")
                session = self.servers[server_name]['session']
                
                # List available tools
                response = await session.list_tools()
                
                tools = []
                for tool in response.tools:
                    tool_info = {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema,
                    }
                    tools.append(tool_info)

                # Cache the results
                self._tool_cache[server_name] = tools
                logger.info(f"Discovered {len(tools)} tools from server '{server_name}'")
                return tools
                
            except Exception as e:
                logger.error(f"Failed to discover tools from persistent connection '{server_name}': {e}")
                raise RuntimeError(f"Could not discover tools from server '{server_name}': {e}")

        # Fall back to per-request connection
        config = self.server_configs[server_name]
        server_params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env=config.get("env"),
        )

        try:
            logger.info(f"Discovering tools from MCP server '{server_name}' using per-request connection...")

            # Create temporary connection for tool discovery
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # List available tools
                    response = await session.list_tools()

                    tools = []
                    for tool in response.tools:
                        tool_info = {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.inputSchema,
                        }
                        tools.append(tool_info)

                    # Cache the results
                    self._tool_cache[server_name] = tools
                    logger.info(f"Discovered {len(tools)} tools from server '{server_name}'")

                    return tools

        except Exception as e:
            logger.error(f"Failed to discover tools from server '{server_name}': {e}")
            raise RuntimeError(f"Could not discover tools from server '{server_name}': {e}")

    async def list_tools(self, server_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List tools available from MCP servers.

        Args:
            server_name: If provided, list tools from this server only.
                        If None, list tools from all registered servers.

        Returns:
            List of tool definitions, each with 'name', 'description', 'inputSchema',
            and 'server' (which server provides it)

        Example:
            >>> tools = await manager.list_tools()
            >>> for tool in tools:
            ...     print(f"{tool['server']}.{tool['name']}: {tool['description']}")
        """
        if server_name:
            # List tools from specific server
            tools = await self.discover_tools(server_name)
            # Add server name to each tool
            tools_with_server = []
            for tool in tools:
                tool_copy = tool.copy()
                tool_copy["server"] = server_name
                tools_with_server.append(tool_copy)
            return tools_with_server

        # List all tools from all registered servers
        all_tools = []
        for srv_name in self.server_configs.keys():
            tools = await self.discover_tools(srv_name)
            for tool in tools:
                tool_copy = tool.copy()
                tool_copy["server"] = srv_name
                all_tools.append(tool_copy)

        return all_tools

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """Call a tool on an MCP server.

        Args:
            server_name: Name of the server providing the tool
            tool_name: Name of the tool to call
            arguments: Dictionary of arguments to pass to the tool

        Returns:
            The result from the tool execution

        Raises:
            ValueError: If server or tool not found
            RuntimeError: If tool execution fails

        Example:
            >>> result = await manager.call_tool(
            ...     "sat",
            ...     "segment_medical_structure",
            ...     {"image_path": "medical_image.nii.gz", "structures": ["left heart ventricle"]}
            ... )
        """
        # Check if we have a persistent connection
        if server_name in self.servers:
            session = self.servers[server_name]['session']
            # Health check: verify connection is still alive
            try:
                await asyncio.wait_for(session.list_tools(), timeout=10)
            except Exception:
                logger.warning(f"MCP server '{server_name}' connection stale, reconnecting...")
                await self.disconnect_server(server_name)
                await self.connect_server(server_name)
                session = self.servers[server_name]['session']
            return await self._execute_tool_call(session, tool_name, arguments, server_name)
        
        # Fall back to per-request connection
        if server_name not in self.server_configs:
            raise ValueError(f"Server '{server_name}' not registered. Call register_server() first.")

        config = self.server_configs[server_name]
        server_params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env=config.get("env"),
        )

        try:
            logger.debug(f"Creating per-request connection to call '{tool_name}' on server '{server_name}'")
            
            # Create temporary connection for this tool call
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await self._execute_tool_call(session, tool_name, arguments, server_name)

        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}' on server '{server_name}': {e}")
            raise RuntimeError(f"Tool call failed: {e}")

    async def _execute_tool_call(self, session, tool_name: str, arguments: Dict[str, Any], server_name: str) -> Any:
        """Execute a tool call on an established session.

        Args:
            session: MCP ClientSession
            tool_name: Name of the tool to call
            arguments: Tool arguments
            server_name: Server name (for logging)

        Returns:
            The tool result. If the tool returns a JSON string, this method will
            parse it and return the corresponding Python object (dict/list).
            Otherwise, the raw text/string is returned.
        """
        try:
            logger.debug(f"Calling tool '{tool_name}' on server '{server_name}' with args: {arguments}")

            # Call the tool
            result = await session.call_tool(tool_name, arguments)

            # Helper: try to parse text as JSON, otherwise return raw text
            def _maybe_parse_json(text: str):
                try:
                    return json.loads(text)
                except Exception:
                    return text

            # Extract content from result
            if hasattr(result, 'content') and len(result.content) > 0:
                # MCP returns content as a list of content items
                content_item = result.content[0]
                if hasattr(content_item, 'text'):
                    parsed = _maybe_parse_json(content_item.text)
                    return parsed
                else:
                    # Non-text content, return string representation
                    return str(content_item)

            # Fallback: try to stringify and parse
            result_str = str(result)
            return _maybe_parse_json(result_str)

        except Exception as e:
            logger.error(f"Failed to execute tool '{tool_name}' on server '{server_name}': {e}")
            raise

    def get_connected_servers(self) -> List[str]:
        """Get list of currently connected server names.

        Returns:
            List of server names
        """
        return list(self.servers.keys())

    def is_connected(self, server_name: str) -> bool:
        """Check if a server is currently connected.

        Args:
            server_name: Name of the server to check

        Returns:
            True if connected, False otherwise
        """
        return server_name in self.servers


class MCPClientError(Exception):
    """Base exception for MCP client errors."""
    pass


class ServerNotFoundError(MCPClientError):
    """Raised when trying to use a server that isn't connected."""
    pass


class ToolNotFoundError(MCPClientError):
    """Raised when trying to call a tool that doesn't exist."""
    pass
