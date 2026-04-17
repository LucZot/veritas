"""MCP server configuration management for VERITAS.

This module handles loading and validating MCP server configurations from
JSON files. It supports environment variable substitution for flexible
deployment across different environments.

Configuration Format:
    {
        "servers": [
            {
                "name": "vlm",
                "command": "python",
                "args": ["examples/mcp_servers/vision_language_model/vlm_server.py"],
                "env": {
                    "MODEL_PATH": "${VLM_MODEL_PATH}",
                    "CUDA_VISIBLE_DEVICES": "0"
                }
            }
        ]
    }

Example:
    >>> from veritas.mcp_config import load_mcp_config
    >>> config = load_mcp_config("mcp_servers.json")
    >>> for server in config["servers"]:
    ...     print(f"Found server: {server['name']}")
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATHS = [
    "mcp_servers.json",
    ".mcp_servers.json",
    "config/mcp_servers.json",
]


def load_mcp_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load MCP server configuration from a JSON file.

    Args:
        config_path: Path to the configuration file. If None, searches
                    default locations: mcp_servers.json, .mcp_servers.json,
                    config/mcp_servers.json

    Returns:
        Dictionary with 'servers' key containing list of server configs

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config is invalid

    Example:
        >>> config = load_mcp_config("my_servers.json")
        >>> print(f"Found {len(config['servers'])} servers")
    """
    # Find config file
    if config_path is None:
        config_file = _find_config_file()
        if config_file is None:
            logger.info("No MCP configuration file found, MCP support disabled")
            return {"servers": []}
    else:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"MCP config file not found: {config_path}")

    # Load and parse JSON
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in MCP config file: {e}")

    # Validate structure
    _validate_config(config)

    # Substitute environment variables
    config = _substitute_env_vars(config)

    logger.info(f"Loaded MCP configuration from {config_file} with {len(config['servers'])} servers")

    return config


def _find_config_file() -> Optional[Path]:
    """Search for MCP configuration file in default locations.

    Returns:
        Path to config file if found, None otherwise
    """
    for path_str in DEFAULT_CONFIG_PATHS:
        path = Path(path_str)
        if path.exists():
            logger.debug(f"Found MCP config at: {path}")
            return path

    return None


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate MCP configuration structure.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    if "servers" not in config:
        raise ValueError("MCP config must have 'servers' key")

    if not isinstance(config["servers"], list):
        raise ValueError("'servers' must be a list")

    for i, server in enumerate(config["servers"]):
        if not isinstance(server, dict):
            raise ValueError(f"Server {i} must be a dictionary")

        # Required fields
        if "name" not in server:
            raise ValueError(f"Server {i} missing required 'name' field")
        if "command" not in server:
            raise ValueError(f"Server {i} ('{server['name']}') missing required 'command' field")

        # Validate types
        if not isinstance(server["name"], str):
            raise ValueError(f"Server {i} 'name' must be a string")
        if not isinstance(server["command"], str):
            raise ValueError(f"Server {i} ('{server['name']}') 'command' must be a string")

        # Optional fields with type validation
        if "args" in server and not isinstance(server["args"], list):
            raise ValueError(f"Server '{server['name']}' 'args' must be a list")
        if "env" in server and not isinstance(server["env"], dict):
            raise ValueError(f"Server '{server['name']}' 'env' must be a dictionary")

    # Check for duplicate names
    names = [s["name"] for s in config["servers"]]
    if len(names) != len(set(names)):
        duplicates = [name for name in names if names.count(name) > 1]
        raise ValueError(f"Duplicate server names: {duplicates}")


def _substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Substitute environment variables in configuration.

    Supports patterns like ${VAR_NAME} or $VAR_NAME.
    If environment variable is not set, keeps the original placeholder.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with environment variables substituted
    """
    config_str = json.dumps(config)

    # Pattern to match ${VAR} or $VAR
    pattern = r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)'

    def replace_var(match):
        var_name = match.group(1) or match.group(2)
        value = os.getenv(var_name)
        if value is None:
            logger.warning(f"Environment variable '{var_name}' not set, keeping placeholder")
            return match.group(0)  # Keep original placeholder
        return value

    config_str = re.sub(pattern, replace_var, config_str)

    return json.loads(config_str)


def get_server_config(config: Dict[str, Any], server_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a specific server by name.

    Args:
        config: Full MCP configuration
        server_name: Name of the server to find

    Returns:
        Server configuration dictionary if found, None otherwise

    Example:
        >>> config = load_mcp_config()
        >>> vlm_config = get_server_config(config, "vlm")
        >>> if vlm_config:
        ...     print(f"VLM command: {vlm_config['command']}")
    """
    for server in config["servers"]:
        if server["name"] == server_name:
            return server

    return None


def validate_server_requirements(server_config: Dict[str, Any]) -> List[str]:
    """Validate that a server's requirements are met.

    Checks:
    - Command executable exists
    - Required environment variables are set
    - Script files exist (for Python/Node scripts)

    Args:
        server_config: Server configuration dictionary

    Returns:
        List of validation error messages (empty if all checks pass)

    Example:
        >>> config = load_mcp_config()
        >>> vlm_config = get_server_config(config, "vlm")
        >>> errors = validate_server_requirements(vlm_config)
        >>> if errors:
        ...     for error in errors:
        ...         print(f"Error: {error}")
    """
    errors = []

    # Check if command exists in PATH (basic check)
    command = server_config["command"]
    import shutil
    if not shutil.which(command):
        errors.append(f"Command '{command}' not found in PATH")

    # Check if script files exist (for Python/Node)
    args = server_config.get("args", [])
    if args and command in ["python", "python3", "node"]:
        script_path = Path(args[0])
        if not script_path.exists():
            errors.append(f"Script file not found: {script_path}")

    # Check environment variables (if they contain placeholders)
    env = server_config.get("env", {})
    for key, value in env.items():
        if isinstance(value, str) and ("${" in value or value.startswith("$")):
            # Still has placeholder - env var wasn't set
            errors.append(f"Environment variable referenced in '{key}' not set: {value}")

    return errors


def create_example_config(output_path: str = "mcp_servers.example.json") -> None:
    """Create an example MCP configuration file.

    Args:
        output_path: Where to save the example config

    Example:
        >>> from veritas.mcp_config import create_example_config
        >>> create_example_config()
    """
    example_config = {
        "servers": [
            {
                "name": "vlm",
                "command": "python",
                "args": ["examples/mcp_servers/vision_language_model/vlm_server.py"],
                "env": {
                    "MODEL_PATH": "${VLM_MODEL_PATH}",
                    "CUDA_VISIBLE_DEVICES": "0"
                }
            },
            {
                "name": "code_executor",
                "command": "python",
                "args": ["examples/mcp_servers/code_executor/executor_server.py"],
                "env": {
                    "SANDBOX_DIR": "/tmp/code_execution"
                }
            }
        ]
    }

    with open(output_path, "w") as f:
        json.dump(example_config, f, indent=2)

    logger.info(f"Created example MCP config at: {output_path}")


if __name__ == "__main__":
    # When run as script, create example config
    create_example_config()
    print("Created mcp_servers.example.json")
