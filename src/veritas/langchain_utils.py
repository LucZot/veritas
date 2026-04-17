"""Utilities for LangChain integration.

This module implements the Adapter Pattern - it translates between
VERITAS's message format and LangChain's message format.
"""

from typing import Any, List, Dict, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage


def create_chat_model(
    model_name: str,
    base_url: str = None,
    temperature: float = None,
    top_p: float = None,
    reasoning: bool = False,
    ollama_config: dict = None,
    num_ctx: int = None,
    keep_alive: int | str | None = None,
    max_output_tokens: int | None = None,
    provider: str | None = None,
) -> Any:
    """Factory function to create provider-specific chat models with consistent config.

    This centralizes model creation logic for Ollama and OpenRouter/OpenAI backends.
    For OpenRouter/OpenAI providers, Ollama-specific parameters (`reasoning`,
    `num_ctx`, `keep_alive`, `ollama_config`) are ignored.

    Args:
        model_name: Provider-specific model name (e.g., "qwen3:8b", "openai/gpt-5.2")
        base_url: Optional provider base URL override.
        temperature: Sampling temperature. If None, no temperature override.
        top_p: Nucleus sampling parameter. If None, no top_p override.
        reasoning: Ollama only - whether to enable thinking/reasoning mode.
        ollama_config: Optional Ollama config dict with 'base_url' and 'enable_thinking' keys.
                      Used when explicit parameters aren't provided.
        num_ctx: Ollama only - context window size.
        keep_alive: Ollama only - model keep_alive setting.
        max_output_tokens: Optional cap for generated output tokens.
        provider: Optional explicit provider override.

    Returns:
        Configured provider model instance

    Example:
        >>> # Direct parameters
        >>> model = create_chat_model("llama3.1:8b", base_url="http://localhost:11434")

        >>> # Using config dict
        >>> config = {"base_url": "http://localhost:11434", "enable_thinking": False}
        >>> model = create_chat_model("llama3.1:8b", ollama_config=config)

        >>> # With Agent object
        >>> from veritas.agent import Agent
        >>> agent = Agent("Scientist", "research", "discover", "lead", "llama3.1:8b")
        >>> model = create_chat_model(agent.model, ollama_config=config)
    """
    import os
    from veritas.config import get_provider_type

    provider_type = (provider or get_provider_type()).lower()

    # Resolve output token cap: explicit > env > default
    if max_output_tokens is None:
        cap_env = os.environ.get("VERITAS_MAX_OUTPUT_TOKENS")
        if cap_env:
            try:
                max_output_tokens = int(cap_env)
            except ValueError:
                max_output_tokens = None
        else:
            max_output_tokens = 16384
    if isinstance(max_output_tokens, int) and max_output_tokens <= 0:
        max_output_tokens = None

    if provider_type in {"openrouter", "openai"}:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required for OpenRouter/OpenAI providers. "
                "Install with: pip install -e '.[openrouter]'"
            ) from exc

        kwargs: dict[str, Any] = {"model": model_name}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens
        # Timeout: prevent indefinite hangs on stalled API calls
        _timeout = int(os.environ.get("VERITAS_LLM_TIMEOUT", "600"))
        kwargs["request_timeout"] = _timeout

        if provider_type == "openrouter":
            from veritas.config import get_openrouter_config

            openrouter_config = get_openrouter_config()
            resolved_base_url = base_url or openrouter_config["base_url"]
            if resolved_base_url:
                kwargs["base_url"] = resolved_base_url
            if openrouter_config["api_key"]:
                kwargs["api_key"] = openrouter_config["api_key"]

            default_headers: dict[str, str] = {}
            if openrouter_config["http_referer"]:
                default_headers["HTTP-Referer"] = str(openrouter_config["http_referer"])
            if openrouter_config["x_title"]:
                default_headers["X-Title"] = str(openrouter_config["x_title"])
            if default_headers:
                kwargs["default_headers"] = default_headers
        else:
            resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
            if resolved_base_url:
                kwargs["base_url"] = resolved_base_url
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                kwargs["api_key"] = api_key

        return ChatOpenAI(**kwargs)

    if provider_type != "ollama":
        raise ValueError(f"Unsupported provider for chat model creation: {provider_type}")

    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is required for Ollama provider. "
            "Install with: pip install -e '.[ollama]'"
        ) from exc

    # Resolve base_url: explicit parameter > config > default
    if base_url is None:
        if ollama_config and "base_url" in ollama_config:
            base_url = ollama_config["base_url"]
        else:
            from veritas.config import get_ollama_config
            base_url = get_ollama_config()["base_url"]

    # Resolve reasoning: explicit parameter > config logic
    if not reasoning and ollama_config:
        # Check if thinking should be enabled from config
        enable_thinking = ollama_config.get("enable_thinking", False)
        if enable_thinking:
            from veritas.constants import is_thinking_model
            reasoning = is_thinking_model(model_name)

    # Resolve num_ctx: explicit parameter > env var > None (use Ollama default)
    if num_ctx is None:
        ctx_env = os.environ.get("OLLAMA_CONTEXT_LENGTH")
        if ctx_env:
            try:
                num_ctx = int(ctx_env)
            except ValueError:
                pass  # Invalid value, use Ollama default

    # Resolve keep_alive: explicit parameter > config > None
    if keep_alive is None and ollama_config:
        keep_alive = ollama_config.get("keep_alive")

    # Build kwargs for ChatOllama
    kwargs = {
        "model": model_name,
        "base_url": base_url,
    }

    if temperature is not None:
        kwargs["temperature"] = temperature

    if top_p is not None:
        kwargs["top_p"] = top_p

    if reasoning:
        kwargs["reasoning"] = True

    if num_ctx is not None:
        kwargs["num_ctx"] = num_ctx

    if keep_alive is not None:
        kwargs["keep_alive"] = keep_alive

    if max_output_tokens is not None:
        kwargs["num_predict"] = max_output_tokens

    # Timeout: prevent indefinite hangs on stalled Ollama inference
    _timeout = int(os.environ.get("VERITAS_LLM_TIMEOUT", "600"))
    kwargs["timeout"] = _timeout

    return ChatOllama(**kwargs)


def build_messages(
    system_prompt: str, message_history: List[Dict[str, str]], user_prompt: str = None
) -> List[BaseMessage]:
    """Build LangChain message list from conversation history.

    Args:
        system_prompt: The system message (agent's role/instructions)
        message_history: Previous conversation as list of {"role": ..., "content": ...}
        user_prompt: Optional new user message to add

    Returns:
        List of LangChain BaseMessage objects

    Example:
        >>> messages = build_messages(
        ...     system_prompt="You are a helpful scientist",
        ...     message_history=[
        ...         {"role": "user", "content": "Hello"},
        ...         {"role": "assistant", "content": "Hi there!"}
        ...     ],
        ...     user_prompt="How are you?"
        ... )
        >>> len(messages)
        4
        >>> type(messages[0])
        <class 'langchain_core.messages.SystemMessage'>
    """
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    # Convert history to LangChain messages
    for msg in message_history:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            # Additional system messages (rare but possible)
            messages.append(SystemMessage(content=content))

    # Add new user prompt if provided
    if user_prompt:
        messages.append(HumanMessage(content=user_prompt))

    return messages


def convert_langchain_to_discussion(
    messages: List[BaseMessage], agent_id_to_title: Dict[str, str] = None
) -> List[Dict[str, str]]:
    """Convert LangChain messages to VERITAS discussion format.

    Args:
        messages: List of LangChain messages
        agent_id_to_title: Optional mapping of agent IDs to titles

    Returns:
        Discussion format: [{"agent": "...", "message": "..."}, ...]

    Example:
        >>> msgs = [
        ...     SystemMessage(content="You are helpful"),
        ...     HumanMessage(content="Hello"),
        ...     AIMessage(content="Hi!")
        ... ]
        >>> discussion = convert_langchain_to_discussion(msgs)
        >>> discussion[0]
        {'agent': 'User', 'message': 'Hello'}
    """
    discussion = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            # Skip system messages in discussion (they're agent prompts, not conversation)
            continue
        elif isinstance(msg, HumanMessage):
            discussion.append({"agent": "User", "message": msg.content})
        elif isinstance(msg, AIMessage):
            # Try to extract agent name from metadata, default to "Assistant"
            agent_name = "Assistant"
            if hasattr(msg, "name") and msg.name:
                agent_name = msg.name
            discussion.append({"agent": agent_name, "message": msg.content})

    return discussion


def count_tokens_langchain(messages: List[BaseMessage], model: str = "llama3.1") -> int:
    """Token count for LangChain messages.

    Uses tiktoken for OpenAI/OpenRouter models (accurate), falls back to
    char-based approximation for Ollama/unknown models.

    Args:
        messages: List of LangChain messages
        model: Model name for tokenizer selection

    Returns:
        Token count (exact for OpenAI models, approximate for others)
    """
    text = "".join(msg.content for msg in messages)
    # Use tiktoken for OpenAI/OpenRouter models
    model_lower = model.lower() if model else ""
    if any(tok in model_lower for tok in ("gpt", "openai", "o1", "o3", "o4")):
        try:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            pass
    # Fallback: char-based approximation for Ollama/unknown models
    return len(text) // 4


def extract_tool_calls(response: AIMessage) -> List[Dict[str, Any]]:
    """Extract tool/function calls from an AI response.

    Args:
        response: AIMessage that may contain tool calls

    Returns:
        List of tool call dictionaries with normalized format

    Example:
        >>> response = AIMessage(content="Let me search...", tool_calls=[
        ...     {"name": "pubmed_search", "args": {"query": "COVID-19"}}
        ... ])
        >>> calls = extract_tool_calls(response)
        >>> calls[0]["name"]
        'pubmed_search'
    """
    # Check for tool calls in standard LangChain location
    if hasattr(response, "tool_calls") and response.tool_calls:
        return _normalize_tool_calls(response.tool_calls)
    
    # Fallback: Check additional_kwargs (some providers put them here)
    if hasattr(response, "additional_kwargs") and response.additional_kwargs:
        if "tool_calls" in response.additional_kwargs:
            return _normalize_tool_calls(response.additional_kwargs["tool_calls"])
    
    return []


def _normalize_tool_calls(tool_calls: List[Any]) -> List[Dict[str, Any]]:
    """Normalize tool calls to consistent format."""
    import re

    def clean_tool_name(name: str) -> str:
        """Remove model-specific tokens from tool names.

        Some models (especially thinking/reasoning models) append special tokens
        like <|channel|>commentary to tool names. This removes them.
        """
        if not isinstance(name, str):
            return name
        # Remove <|...|> tokens (e.g., <|channel|>commentary)
        cleaned = re.sub(r'<\|[^|]*\|>[^<]*', '', name).strip()
        return cleaned

    normalized = []
    for i, call in enumerate(tool_calls):
        if isinstance(call, dict):
            raw_name = call.get("name", call.get("function", {}).get("name", ""))
            normalized.append({
                "id": call.get("id", f"call_{i}"),
                "name": clean_tool_name(raw_name),
                "args": call.get("args", call.get("function", {}).get("arguments", {})),
            })
        else:
            # Object format (has attributes)
            raw_name = getattr(call, "name", "")
            normalized.append({
                "id": getattr(call, "id", f"call_{i}"),
                "name": clean_tool_name(raw_name),
                "args": getattr(call, "args", {}),
            })
    return normalized


def format_tool_results(tool_outputs: List[Dict[str, str]]) -> str:
    """Format tool execution results for display.

    Args:
        tool_outputs: List of {"tool_call_id": ..., "output": ...}

    Returns:
        Formatted string for adding to conversation

    Example:
        >>> results = [{"tool_call_id": "call_123", "output": "Found 3 papers..."}]
        >>> formatted = format_tool_results(results)
        >>> "Found 3 papers" in formatted
        True
    """
    if not tool_outputs:
        return ""

    formatted_parts = []
    for output in tool_outputs:
        formatted_parts.append(f"Tool Output:\n{output['output']}")

    return "\n\n".join(formatted_parts)


