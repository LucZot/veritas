"""Tool Binding for Agent Models.

This module handles binding tools to agent models and enhancing prompts
with tool documentation for both team and individual meetings.
"""

from typing import Dict, List, Tuple, Any, Sequence
from langchain_core.messages import AIMessage
import logging
import json
import re

from veritas.agent import Agent, resolve_temperature, resolve_top_p
from veritas.config import get_provider_type
from veritas.constants import OLLAMA_FUNCTION_CALLING_MODELS
from veritas.langchain_utils import create_chat_model
from veritas.mcp import build_tool_documentation

logger = logging.getLogger(__name__)


def _extract_raw_from_tool_error(error_msg: str) -> str | None:
    """Extract raw model output from an Ollama tool-call parsing error.

    Ollama errors look like:
        error parsing tool call: raw='import os, json...', err=invalid character...

    Returns the raw text if found, None otherwise.
    """
    # Use greedy match so we capture up to the LAST ', err=' in case the
    # raw output itself contains that substring (e.g. in a string literal).
    match = re.search(r"raw='(.*)',\s*err=", error_msg, re.DOTALL)
    if match:
        return match.group(1)
    return None


class ToolCallSafeModel:
    """Wrapper around ChatOllama that recovers from Ollama tool-call parsing errors.

    Some models (e.g., gpt-oss:20b) have tool-calling baked into their chat template.
    Even without tools bound, Ollama tries to parse their output as JSON tool calls,
    which fails when the model outputs plain text. This wrapper catches the error
    and extracts the raw model output from the error message.
    """

    def __init__(self, model: Any):
        self._model = model

    def invoke(self, *args, **kwargs):
        try:
            return self._model.invoke(*args, **kwargs)
        except Exception as e:
            raw_text = _extract_raw_from_tool_error(str(e))
            if raw_text is not None:
                return AIMessage(content=raw_text)
            raise

    def __getattr__(self, name):
        return getattr(self._model, name)


def _patch_ollama_json_for_escape_sequences():
    """Monkey-patch json.loads to fix invalid escape sequences from Ollama models.

    Some Ollama models (like gpt-oss:20b) generate invalid JSON escape sequences
    like \\' when creating tool calls with Python code containing apostrophes.
    This patches json.loads to replace \\' with ' before parsing.
    """
    try:
        if get_provider_type() != "ollama":
            return
        # Save original
        original_json_loads = json.loads

        def fixed_json_loads(s, *args, **kwargs):
            """json.loads with automatic fix for common Ollama model issues."""
            try:
                return original_json_loads(s, *args, **kwargs)
            except json.JSONDecodeError as e:
                error_msg = str(e).lower()

                # Fix 1: Invalid escape sequences (\\')
                if ("escape" in error_msg or "invalid character" in error_msg) and isinstance(s, str):
                    if "\\'" in s:
                        fixed = s.replace("\\'", "'")
                        try:
                            return original_json_loads(fixed, *args, **kwargs)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass

                # Fix 2: Empty key in object (model generated {""})
                if isinstance(s, str) and s.strip() in ['{""}', '{"":}', '{}']:
                    logger.warning(f"Model generated empty/invalid JSON: {s}, returning empty dict")
                    return {}

                raise  # Re-raise original error if no fix worked

        # Apply patch globally
        json.loads = fixed_json_loads
        logger.info("✓ Applied JSON escape sequence fix for Ollama compatibility")

    except Exception as e:
        logger.warning(f"Could not patch JSON parsing: {e}")


# Apply patch at module import
_patch_ollama_json_for_escape_sequences()


def _model_supports_tools(model_name: str, provider: str | None = None) -> bool:
    """Check if a model should be treated as tool-call capable."""
    provider_type = (provider or get_provider_type()).lower()
    if provider_type == "ollama":
        return any(m in model_name for m in OLLAMA_FUNCTION_CALLING_MODELS)
    return True


def bind_tools_to_agents(
    agents: Sequence[Agent],
    ollama_config: Dict[str, Any],
    temperature: float,
    top_p: float | None = None,
    tool_registry: Dict[str, Any] = None,
    mcp_tool_info: Dict[str, Any] = None,
) -> Tuple[Dict[Agent, Any], Dict[Agent, str]]:
    """Bind tools to multiple agents and enhance their prompts.

    Creates models for each agent, binds appropriate tools based on
    agent.available_tools, and enhances prompts with tool documentation.

    Args:
        agents: Sequence of agents to create models for
        ollama_config: Ollama configuration dict
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter (fallback for agents without per-agent top_p)
        tool_registry: Unified registry of all available tools
        mcp_tool_info: MCP tool metadata for documentation

    Returns:
        Tuple of (agent_models, agent_prompts) dictionaries
    """
    agent_models = {}
    agent_prompts = {}

    for agent in agents:
        model, prompt = bind_tools_to_agent(
            agent=agent,
            ollama_config=ollama_config,
            temperature=temperature,
            top_p=top_p,
            tool_registry=tool_registry,
            mcp_tool_info=mcp_tool_info,
        )
        agent_models[agent] = model
        agent_prompts[agent] = prompt

    return agent_models, agent_prompts


def bind_tools_to_agent(
    agent: Agent,
    ollama_config: Dict[str, Any],
    temperature: float,
    top_p: float | None = None,
    tool_registry: Dict[str, Any] = None,
    mcp_tool_info: Dict[str, Any] = None,
) -> Tuple[Any, str]:
    """Bind tools to a single agent's model and enhance prompt.

    Args:
        agent: Agent to create model for
        ollama_config: Ollama configuration
        temperature: Sampling temperature
        tool_registry: Unified tool registry
        mcp_tool_info: MCP tool metadata

    Returns:
        Tuple of (model, enhanced_prompt)
    """
    # Create base model using factory function
    model_name = agent.model or ollama_config["model"]
    provider_type = get_provider_type()
    effective_temperature = resolve_temperature(agent, temperature)
    effective_top_p = resolve_top_p(agent, top_p)

    # Check if model supports function calling
    supports_tools = _model_supports_tools(model_name, provider_type)

    # Check if this is code-as-output mode (empty available_tools list)
    is_code_output_mode = (
        hasattr(agent, 'available_tools') and
        agent.available_tools is not None and
        len(agent.available_tools) == 0
    )

    # For code-as-output mode with function-calling models, wrap in
    # ToolCallSafeModel to recover from Ollama's tool-call parsing errors.
    # Models like gpt-oss:20b have tool-calling baked into their chat template;
    # even without tools bound, Ollama tries to parse output as JSON tool calls.
    if provider_type == "ollama" and is_code_output_mode and supports_tools:
        model = create_chat_model(
            model_name=model_name,
            temperature=effective_temperature,
            top_p=effective_top_p,
            ollama_config=ollama_config,
        )
        return ToolCallSafeModel(model), agent.prompt

    # Create model normally for non-code-output agents
    model = create_chat_model(
        model_name=model_name,
        temperature=effective_temperature,
        top_p=effective_top_p,
        ollama_config=ollama_config,
    )

    # Bind explicit tools if agent has them, otherwise return unbound model
    if hasattr(agent, 'available_tools') and agent.available_tools:
        return _bind_explicit_tools(
            agent, model, supports_tools, tool_registry, mcp_tool_info
        )

    return model, agent.prompt


def _bind_explicit_tools(
    agent: Agent,
    model: Any,
    supports_tools: bool,
    tool_registry: Dict[str, Any],
    mcp_tool_info: Dict[str, Any],
) -> Tuple[Any, str]:
    """Bind tools explicitly listed in agent.available_tools.

    Args:
        agent: Agent with available_tools attribute
        model: Base ChatOllama model
        supports_tools: Whether model supports function calling
        tool_registry: Unified tool registry
        mcp_tool_info: MCP tool metadata

    Returns:
        Tuple of (enhanced_model, enhanced_prompt)
    """
    # Find tools from registry
    tools_to_bind = []
    found_tools = []
    missing_tools = []

    for tool_name in agent.available_tools:
        if tool_name in tool_registry:
            tools_to_bind.append(tool_registry[tool_name])
            found_tools.append(tool_name)

            # Debug output: track tool source
            if tool_name in mcp_tool_info:
                server = mcp_tool_info[tool_name]['server']
                print(f"  📍 {tool_name} -> MCP server '{server}'")
            else:
                print(f"  📍 {tool_name} -> LangChain tool")
        else:
            missing_tools.append(tool_name)

    # Warn about missing tools
    if missing_tools:
        print(f"⚠️  Missing tools for {agent.title}: {missing_tools}")

    # Bind tools if model supports it
    if tools_to_bind and supports_tools:
        model = model.bind_tools(tools_to_bind)

        # Build enhanced prompt with tool documentation
        tool_docs = build_tool_documentation(mcp_tool_info, found_tools)

        if tool_docs:
            # Rich documentation available (for MCP tools)
            enhanced_prompt = (
                f"{agent.prompt}\n\n"
                f"You have access to the following tools:{tool_docs}\n\n"
                f"IMPORTANT: Read the parameter requirements carefully. "
                f"Call tools with the EXACT values specified above."
                f"IMPORTANT: You should ACTUALLY CALL these tools to complete your tasks."
            )
        else:
            # Basic tool list (for LangChain tools)
            tool_list = ', '.join(found_tools)
            enhanced_prompt = (
                f"{agent.prompt}\n\n"
                f"You have access to the following tools: {tool_list}.\n\n"
                f"IMPORTANT: You should ACTUALLY CALL these tools to complete your tasks."
            )

        print(f"🛠️  {agent.title}: Bound {len(tools_to_bind)} tools ({', '.join(found_tools)})")
        return model, enhanced_prompt

    else:
        # No tools or model doesn't support function calling
        if not supports_tools and tools_to_bind:
            print(f"📝 Model {agent.model or 'default'} doesn't support function calling")

        return model, agent.prompt
