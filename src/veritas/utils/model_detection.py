"""Utilities for detecting model types and capabilities."""

from veritas.agent import Agent


def detect_model_type(agent: Agent) -> str:
    """Detect whether an agent's model is a coding-specialized or general model.

    Args:
        agent: Agent to check

    Returns:
        "coding" for code-specialized models, "general" for general-purpose models

    Examples:
        >>> agent = Agent(..., model="qwen3-coder:30b")
        >>> detect_model_type(agent)
        'coding'

        >>> agent = Agent(..., model="qwen3:8b")
        >>> detect_model_type(agent)
        'general'
    """
    model_name = agent.model.lower() if agent.model else ""

    # Keywords that indicate coding-specialized models
    coding_keywords = [
        "coder",
        "code",
        "codellama",
        "deepseek-coder",
        "starcoder",
        "wizardcoder",
    ]

    return "coding" if any(keyword in model_name for keyword in coding_keywords) else "general"


def is_coding_model(agent: Agent) -> bool:
    """Check if agent uses a coding-specialized model.

    Args:
        agent: Agent to check

    Returns:
        True if agent uses a coding model, False otherwise
    """
    return detect_model_type(agent) == "coding"
