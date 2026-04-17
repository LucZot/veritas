"""Static constants for VERITAS.

Model definitions and other immutable values.
For runtime configuration, see config.py.
"""

# Execution safety limits
MAX_CODE_EXECUTIONS_PER_ROUND = 8

# Default models by provider
DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.2"
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"

CONSISTENT_TEMPERATURE = 0.2

# Ollama models that support function calling
OLLAMA_FUNCTION_CALLING_MODELS = [
    "gpt-oss:20b",
    "gpt-oss:120b",
    "qwen3:4b",
    "qwen3:8b",
    "qwen3:30b",
    "qwen3:235b",
]

# Ollama models that support thinking/reasoning mode
OLLAMA_THINKING_MODELS = [
    "qwen3:8b",
    "qwen3:30b",
    "qwen3:235b",
    "gpt-oss:20b",
    "gpt-oss:120b",
]

def is_thinking_model(model_name: str) -> bool:
    """Check if a model supports thinking/reasoning mode.

    Args:
        model_name: The name of the model to check (e.g., "llama3.1:8b", "qwen3:8b")

    Returns:
        True if the model is in OLLAMA_THINKING_MODELS list, False otherwise
    """
    return any(supported in model_name for supported in OLLAMA_THINKING_MODELS)

