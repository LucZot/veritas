"""Configuration management for LLM providers and workspace management.

This module implements the Factory Pattern to create the appropriate
provider based on environment configuration, and provides centralized
workspace configuration.

Environment Variables:
    LLM_PROVIDER: "openai", "openrouter", or "ollama" (default: "openai")
    OLLAMA_BASE_URL: Ollama server URL (default: "http://localhost:11434")
    OLLAMA_MODEL: Default model name (default: "llama3.1")
    OPENROUTER_BASE_URL: OpenRouter API URL (default: "https://openrouter.ai/api/v1")
    OPENROUTER_API_KEY: OpenRouter API key
    OPENROUTER_HTTP_REFERER: Optional attribution header for OpenRouter
    OPENROUTER_X_TITLE: Optional attribution header for OpenRouter
    CODE_EXEC_WORKSPACE_DIR: Base directory for code execution workspaces
    VERITAS_OUTPUT_DIR: Base directory for all outputs (default: "./outputs")
"""

import os
from pathlib import Path
from typing import Literal

ProviderType = Literal["openai", "openrouter", "ollama"]


# ============================================================================
# Workspace Configuration
# ============================================================================

# Base output directory for all VERITAS outputs (meetings, discussions, etc.)
BASE_OUTPUT_DIR = Path(os.getenv('VERITAS_OUTPUT_DIR', './outputs')).absolute()

# Code execution workspace configuration
# This determines where code execution sessions store their files (code, plots, data).
#
# Priority:
# 1. Environment variable CODE_EXEC_WORKSPACE_DIR (if set, use this specific path)
# 2. None (default) - workspaces are nested inside meeting output directories
#
# Examples:
#   - None: outputs/code_execution/073_test1/workspace/
#   - Custom: /tmp/code_execution_outputs/ (old behavior for backwards compatibility)
CODE_EXEC_WORKSPACE_DIR = os.getenv('CODE_EXEC_WORKSPACE_DIR', None)

# Default workspace base directory for standalone/legacy usage
# Used when CODE_EXEC_WORKSPACE_DIR is not set and no meeting context is available
DEFAULT_WORKSPACE_BASE = Path('.tmp/code_execution_outputs').absolute()


# ============================================================================
# LLM Provider Configuration
# ============================================================================


def get_provider_type() -> ProviderType:
    """Get the configured LLM provider type.

    Reads from LLM_PROVIDER environment variable.
    Defaults to "openai" for backward compatibility.

    Returns:
        The provider type: "openai", "openrouter", or "ollama"

    Example:
        >>> import os
        >>> os.environ["LLM_PROVIDER"] = "ollama"
        >>> get_provider_type()
        'ollama'
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider not in ["openai", "openrouter", "ollama"]:
        raise ValueError(
            f"Invalid LLM_PROVIDER: {provider}. Must be 'openai', 'openrouter', or 'ollama'."
        )
    return provider


def get_ollama_config() -> dict[str, str]:
    """Get Ollama configuration from environment variables.

    Returns:
        Dictionary with keys:
            - base_url: Ollama server URL
            - model: Default model name
            - enable_thinking: Whether to enable thinking/reasoning mode
                              (defaults to True to align with Ollama's default behavior)

    Example:
        >>> config = get_ollama_config()
        >>> config["base_url"]
        'http://localhost:11434'
    """
    keep_alive_env = os.getenv("OLLAMA_KEEP_ALIVE")
    keep_alive_value = None
    if keep_alive_env is not None and keep_alive_env != "":
        try:
            keep_alive_value = int(keep_alive_env)
        except ValueError:
            keep_alive_value = keep_alive_env

    return {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "model": os.getenv("OLLAMA_MODEL", "llama3.1"),
        "enable_thinking": os.getenv("OLLAMA_ENABLE_THINKING", "true").lower() == "true",
        "keep_alive": keep_alive_value,
    }


def get_openrouter_config() -> dict[str, str | None]:
    """Get OpenRouter configuration from environment variables.

    Returns:
        Dictionary with keys:
            - base_url: OpenRouter API URL
            - api_key: API key (OPENROUTER_API_KEY preferred, OPENAI_API_KEY fallback)
            - http_referer: Optional attribution header
            - x_title: Optional attribution header
    """
    return {
        "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "api_key": os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        "http_referer": os.getenv("OPENROUTER_HTTP_REFERER"),
        "x_title": os.getenv("OPENROUTER_X_TITLE"),
    }


def validate_ollama_available() -> bool:
    """Check if Ollama server is running and accessible.

    Returns:
        True if Ollama is available, False otherwise

    Educational Note:
        This is an example of defensive programming - checking
        preconditions before attempting an operation that might fail.
    """
    import requests

    config = get_ollama_config()
    try:
        response = requests.get(f"{config['base_url']}/api/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def validate_ollama_model(model: str) -> bool:
    """Check if a specific model is installed in Ollama.

    Args:
        model: Model name to check (e.g., "llama3.1")

    Returns:
        True if model is installed, False otherwise

    Example:
        >>> validate_ollama_model("llama3.1")
        True
    """
    import requests

    config = get_ollama_config()
    try:
        response = requests.get(f"{config['base_url']}/api/tags", timeout=2)
        if response.status_code != 200:
            return False

        data = response.json()
        installed_models = [m["name"] for m in data.get("models", [])]

        # Check for exact match or partial match (e.g., "llama3.1" matches "llama3.1:latest")
        return any(model in installed_model for installed_model in installed_models)
    except requests.exceptions.RequestException:
        return False


def check_provider_ready() -> tuple[bool, str]:
    """Check if the configured provider is ready to use.

    Returns:
        Tuple of (is_ready, message)
            - is_ready: True if provider is configured correctly
            - message: Description of status or error

    Example:
        >>> is_ready, msg = check_provider_ready()
        >>> if not is_ready:
        ...     print(f"Error: {msg}")
    """
    provider = get_provider_type()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False, "OPENAI_API_KEY environment variable not set"
        return True, "OpenAI configured"

    elif provider == "openrouter":
        config = get_openrouter_config()
        if not config["api_key"]:
            return False, "OPENROUTER_API_KEY (or OPENAI_API_KEY) environment variable not set"
        return True, f"OpenRouter configured ({config['base_url']})"

    elif provider == "ollama":
        if not validate_ollama_available():
            config = get_ollama_config()
            return (
                False,
                f"Ollama server not running at {config['base_url']}. "
                f"Start it with: ollama serve",
            )

        model = get_ollama_config()["model"]
        if not validate_ollama_model(model):
            return (
                False,
                f"Model '{model}' not installed. Install with: ollama pull {model}",
            )

        return True, f"Ollama configured with model '{model}'"

    return False, f"Unknown provider: {provider}"


def get_execution_summary() -> str:
    """Get a human-readable summary of current execution configuration.

    Useful for debugging and logging.

    Returns:
        Formatted configuration summary
    """
    provider = get_provider_type()
    lines = [f"Provider: {provider}"]

    if provider == "ollama":
        config = get_ollama_config()
        lines.append(f"Model: {config['model']}")
        lines.append(f"Ollama URL: {config['base_url']}")
    elif provider == "openrouter":
        config = get_openrouter_config()
        lines.append(f"OpenRouter URL: {config['base_url']}")

    return "\n".join(lines)


def get_default_model() -> str:
    """Get the default model for the current LLM provider.

    Returns:
        Model name string appropriate for the configured provider.

    Example:
        >>> import os
        >>> os.environ["LLM_PROVIDER"] = "ollama"
        >>> get_default_model()
        'llama3.1:8b'
    """
    from veritas.constants import DEFAULT_OPENROUTER_MODEL, DEFAULT_OLLAMA_MODEL

    provider = get_provider_type()
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    return os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
