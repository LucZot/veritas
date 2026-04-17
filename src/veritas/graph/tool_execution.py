"""Tool execution node for LangGraph workflows.

Handles execution of PubMed searches and MCP tools, with proper error handling
and context-aware result formatting for agent consumption.
"""

from typing import Any
import json
from langchain_core.messages import HumanMessage, ToolMessage
from veritas.agent import Agent
from veritas.graph_state import MeetingState, IndividualMeetingState


def truncate_tool_output(result: Any, max_length: int = 2000) -> str:
    """Truncate tool output to prevent context overflow.

    Args:
        result: Tool output (can be string, dict, list, etc.)
        max_length: Maximum character length for output

    Returns:
        Truncated string representation
    """
    result_str = str(result) if not isinstance(result, str) else result

    if len(result_str) <= max_length:
        return result_str

    # Try to be smart about truncation based on content type
    try:
        # If it's JSON, try to create a summary
        if isinstance(result, dict):
            # For batch operations with "images" or "results" lists
            if "images" in result and isinstance(result["images"], list):
                summary = {**result}
                n_images = len(result["images"])
                summary["images"] = f"[{n_images} images - details truncated for brevity]"
                if n_images > 0:
                    summary["sample_image"] = result["images"][0]  # Show first as example
                return json.dumps(summary, indent=2)

            if "results" in result and isinstance(result["results"], dict):
                summary = {**result}
                n_results = len(result["results"])
                summary["results"] = f"[{n_results} results - details truncated for brevity]"
                # Show summary statistics
                if n_results > 0:
                    first_key = list(result["results"].keys())[0]
                    summary["sample_result"] = {first_key: result["results"][first_key]}
                return json.dumps(summary, indent=2)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Generic truncation with ellipsis
    truncated = result_str[:max_length]
    remaining = len(result_str) - max_length
    return f"{truncated}\n\n... [truncated {remaining} characters for brevity]"


def create_tool_execution_node(agent: Agent = None, agent_map: dict[str, Agent] = None) -> callable:
    """Create a node that handles tool execution (PubMed, Vision, and MCP tools).

    This node checks the last message for tool calls, executes them,
    and adds the results back to the conversation with context-aware prompting.

    Args:
        agent: The agent that requested the tools (for individual meetings).
        agent_map: Dictionary mapping agent names to Agent objects (for team meetings).
                   If provided, will lookup agent by name from message metadata.

    Returns:
        A node function that executes tools.
    """

    def tool_execution_node(state: MeetingState | IndividualMeetingState) -> dict[str, Any]:
        """Execute tools requested in the last message."""
        from veritas.langchain_utils import count_tokens_langchain
        import re
        import asyncio

        messages = list(state["messages"])
        if not messages:
            return {}

        last_message = messages[-1]

        # Check for tool calls
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {}

        tool_outputs = []
        new_messages = []

        def append_tool_result_message(tool_call_id: str | None, content: str) -> None:
            """Append provider-compatible tool output messages.

            OpenAI-compatible providers require tool results to be emitted as ToolMessage
            with a matching tool_call_id before the next assistant turn.
            """
            if tool_call_id and tool_call_id != "unknown":
                new_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
            else:
                # Fallback for malformed/missing IDs
                new_messages.append(HumanMessage(content=content))

        # Determine which agent made the tool call
        # For team meetings: look up by message name or use agent_map
        # For individual meetings: use the provided agent
        current_agent = agent
        if agent_map and hasattr(last_message, 'name'):
            # Team meeting: lookup agent by name from message
            agent_name_from_msg = last_message.name
            current_agent = agent_map.get(agent_name_from_msg)

        # Agent name for logging
        agent_name = current_agent.title if current_agent else "Agent"

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call.get("args", {})

            args_summary = ", ".join(f"{k}={v!r}" for k, v in args.items()) if args else ""
            print(f"🔧 [Tool Execution] {agent_name} is calling '{tool_name}({args_summary})'...")

            # First, check if this is an MCP tool by discovering MCP tools
            is_mcp_tool = False
            mcp_server_name = None
            
            if current_agent and hasattr(current_agent, 'mcp_servers') and current_agent.mcp_servers:
                try:
                    from veritas.mcp import discover_mcp_tools_for_agents
                    mcp_result = discover_mcp_tools_for_agents([current_agent])
                    mcp_tool_info = mcp_result.get('mcp_tool_info', {})
                    
                    if tool_name in mcp_tool_info:
                        is_mcp_tool = True
                        mcp_server_name = mcp_tool_info[tool_name]['server']
                        print(f"📍 [Tool Routing] {tool_name} -> MCP server '{mcp_server_name}'")
                    else:
                        print(f"📍 [Tool Routing] {tool_name} -> LangChain tool")
                except Exception as e:
                    print(f"⚠️  [Tool Routing] MCP discovery failed: {e}")

            # Route to appropriate execution method
            if is_mcp_tool and mcp_server_name:
                try:
                    # Import MCP client
                    from veritas.mcp import MCPClientManager

                    # Create MCP client and register servers first
                    mcp_manager = MCPClientManager()

                    # Load MCP config and register servers BEFORE calling tools
                    try:
                        from veritas.mcp import load_mcp_config
                        mcp_config = load_mcp_config()

                        # Create lookup for servers by name
                        server_lookup = {server['name']: server for server in mcp_config.get('servers', [])}

                        # Register all agent's MCP servers
                        for server_name in current_agent.mcp_servers:
                            if server_name in server_lookup:
                                server_config = server_lookup[server_name]
                                mcp_manager.register_server(server_name, server_config)
                                print(f"🔌 [MCP Setup] Registered server: {server_name}")
                            else:
                                print(f"⚠️  [MCP Setup] Server '{server_name}' not found in config")

                    except Exception as e:
                        print(f"❌ [MCP Setup] Failed to load MCP config: {e}")
                        raise

                    # Execute MCP tool call asynchronously
                    async def execute_mcp_tool():
                        # Extract actual arguments from nested structure
                        # Debug: print the raw args structure

                        # Tool call format can be:
                        # 1. {'args': {'arguments': {...}}}
                        # 2. {'args': {'__arg1': {...}}}
                        # 3. {'args': {'param1': 'value1', ...}}

                        # Start with the args we received
                        actual_arguments = args

                        # Try 'arguments' key first (some models use this)
                        if "arguments" in actual_arguments:
                            actual_arguments = actual_arguments["arguments"]

                        # Handle __arg1 structure (common with gpt-oss model)
                        elif "__arg1" in actual_arguments:
                            arg1_value = actual_arguments["__arg1"]

                            # If __arg1 contains an object with the real parameters, use those
                            if isinstance(arg1_value, dict):
                                actual_arguments = arg1_value
                            else:
                                # If __arg1 is a simple value, create appropriate arguments
                                if tool_name == "check_sat_status":
                                    actual_arguments = {"model_variant": arg1_value}
                                else:
                                    # For other tools, try to use as image_path
                                    actual_arguments = {"image_path": arg1_value}

                        # Infrastructure-level parameter injection for execute_code_file
                        # These are NOT exposed in the tool schema but added automatically
                        if tool_name == "execute_code_file":
                            import os

                            # Add workspace_base_dir from environment if configured
                            workspace_dir = os.getenv('CODE_EXEC_WORKSPACE_DIR')
                            if workspace_dir:
                                actual_arguments['workspace_base_dir'] = workspace_dir

                            # Add data_paths from MCP server config if available
                            # (This is handled in the MCP server environment, so we don't need to add it here)

                        # Use the discovered server name for this specific tool
                        print(f"🔧 [MCP Call] Server: {mcp_server_name}, Tool: {tool_name}, Args: {actual_arguments}")
                        return await mcp_manager.call_tool(mcp_server_name, tool_name, actual_arguments)

                    # Run the async call
                    result = asyncio.run(execute_mcp_tool())

                    tool_outputs.append({
                        "tool_call_id": tool_call.get("id", "unknown"),
                        "output": result,
                    })

                    # Check if result indicates success or failure
                    result_str = str(result).lower()
                    
                    # Prefer explicit success/failure flags when present.
                    is_json_failure = (
                        '"success": false' in result_str
                        or '"success":false' in result_str
                        or "'success': false" in result_str
                    )
                    is_json_success = (
                        '"success": true' in result_str
                        or '"success":true' in result_str
                        or "'success': true" in result_str
                    )

                    # Check for general error keywords only when no explicit status exists
                    has_error_keywords = any(
                        error_keyword in result_str
                        for error_keyword in ["error", "failed", "invalid", "exception", "traceback"]
                    )

                    if is_json_failure:
                        is_error = True
                    elif is_json_success:
                        is_error = False
                    else:
                        is_error = has_error_keywords

                    # Truncate result to prevent context overflow
                    truncated_result = truncate_tool_output(result, max_length=2000)

                    # Add context for agent
                    if is_error:
                        # If tool returned an error, allow agent to retry
                        tool_context = (
                            f"Tool '{tool_name}' Results:\n\n{truncated_result}\n\n"
                            f"Interpret this error and determine if you should retry with different parameters."
                        )
                    else:
                        # If tool succeeded, have agent analyze and continue if needed
                        tool_context = (
                            f"Tool '{tool_name}' Results:\n\n{truncated_result}\n\n"
                            f"Use these results to continue your task. Call more tools if needed, "
                            f"or provide a summary if complete."
                        )

                    append_tool_result_message(tool_call.get("id", "unknown"), tool_context)

                    # Print status based on actual success/failure
                    if is_error:
                        print(f"❌ [Tool Failed] {tool_name} execution failed")
                    else:
                        print(f"✅ [Tool Success] {tool_name} executed successfully")

                except Exception as e:
                    print(f"❌ [Tool Error] Failed to execute {tool_name}: {e}")

                    # Add error message
                    error_context = (
                        f"Error executing tool '{tool_name}': {e}\n\n"
                        f"Please acknowledge this error and continue with other available information."
                    )

                    append_tool_result_message(tool_call.get("id", "unknown"), error_context)

                    tool_outputs.append({
                        "tool_call_id": tool_call.get("id", "unknown"),
                        "output": f"Error: {e}",
                    })
            else:
                # Handle LangChain tools
                try:
                    print(f"🔧 [LangChain Tool] Executing {tool_name}")

                    # Import and execute LangChain tools dynamically
                    from veritas.tools import write_file, read_file, list_files
                    from veritas.vision.datasets.dataset_tools import get_dataset_patient_info, list_dataset_patients

                    langchain_tools = {
                        "write_file": write_file,
                        "read_file": read_file,
                        "list_files": list_files,
                        "get_dataset_patient_info": get_dataset_patient_info,
                        "list_dataset_patients": list_dataset_patients,
                    }

                    if tool_name in langchain_tools:
                        # Infrastructure-level parameter injection for workspace tools
                        # Add workspace_base_dir from environment if configured
                        import os
                        workspace_dir = os.getenv('CODE_EXEC_WORKSPACE_DIR')
                        if workspace_dir and tool_name in ["write_file", "read_file", "list_files"]:
                            # ALWAYS override workspace_base_dir with environment value (agent shouldn't set this)
                            args['workspace_base_dir'] = workspace_dir

                        tool_func = langchain_tools[tool_name]
                        result = tool_func.invoke(args)
                        
                        tool_outputs.append({
                            "tool_call_id": tool_call.get("id", "unknown"),
                            "output": result,
                        })

                        # Truncate result to prevent context overflow
                        truncated_result = truncate_tool_output(result, max_length=2000)

                        # Add context for agent
                        tool_context = (
                            f"Tool '{tool_name}' Results:\n\n{truncated_result}\n\n"
                            f"Interpret these results and continue with your task."
                        )

                        append_tool_result_message(tool_call.get("id", "unknown"), tool_context)
                        print(f"✅ [LangChain Tool Success] {tool_name} executed successfully")
                    else:
                        # Unknown LangChain tool
                        print(f"⚠️  [Unknown LangChain Tool] {tool_name} not found")
                        raise Exception(f"Unknown LangChain tool: {tool_name}")
                        
                except Exception as e:
                    print(f"❌ [LangChain Tool Error] Failed to execute {tool_name}: {e}")
                    
                    error_context = (
                        f"Error executing LangChain tool '{tool_name}': {e}\n\n"
                        f"Please acknowledge this error and continue with other available information."
                    )
                    
                    append_tool_result_message(tool_call.get("id", "unknown"), error_context)
                    
                    tool_outputs.append({
                        "tool_call_id": tool_call.get("id", "unknown"),
                        "output": f"Error: {e}",
                    })

            # Fallback: completely unknown tool
            known_langchain_tools = ["write_file", "read_file", "list_files", "get_dataset_patient_info", "list_dataset_patients"]
            if tool_name not in known_langchain_tools and not is_mcp_tool:
                # Unknown tool and no MCP fallback available
                print(f"⚠️  [Unknown Tool] {agent_name} tried to call unknown tool: {tool_name}")

                error_context = f"Unknown tool '{tool_name}'. Available tools depend on your configuration."
                append_tool_result_message(tool_call.get("id", "unknown"), error_context)

                tool_outputs.append({
                    "tool_call_id": tool_call.get("id", "unknown"),
                    "output": f"Error: Unknown tool '{tool_name}'",
                })

        num_calls = len(last_message.tool_calls) if hasattr(last_message, "tool_calls") else 0
        return {
            "messages": new_messages,
            "tool_outputs": tool_outputs,
            "tools_just_executed": True,  # Flag to allow agent response even at max_rounds
            "mcp_tool_calls_this_round": state.get("mcp_tool_calls_this_round", 0) + num_calls,
        }

    return tool_execution_node
