"""Schema conversion utilities for tool handling.

This module provides shared utilities for converting JSON schemas (from MCP
and other sources) to Pydantic models for LangChain compatibility.
"""

from typing import Optional, Type
from pydantic import BaseModel, Field, create_model


def json_schema_to_pydantic(
    schema: dict,
    model_name: str,
    required_fields: list[str] = None,
) -> Type[BaseModel]:
    """Convert JSON Schema to Pydantic BaseModel for LangChain compatibility.

    Critical for Ollama models which require Pydantic schemas for proper
    structured tool call generation.

    Args:
        schema: JSON Schema dict with 'properties', 'required' fields, etc.
               Can be a full schema or just the properties dict.
        model_name: Name for the generated Pydantic model class
        required_fields: List of required field names. If None, extracted from schema.

    Returns:
        Dynamically created Pydantic BaseModel class

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "query": {"type": "string", "description": "Search query"},
        ...         "limit": {"type": "integer", "description": "Max results"}
        ...     },
        ...     "required": ["query"]
        ... }
        >>> Model = json_schema_to_pydantic(schema, "SearchInput")
        >>> Model.__name__
        'SearchInput'
        >>> Model.model_fields.keys()
        dict_keys(['query', 'limit'])
    """
    # Handle empty schema
    if not schema:
        schema = {'type': 'object', 'properties': {}}

    # Extract properties and required fields from schema
    properties = schema.get('properties', {})
    if required_fields is None:
        required_fields = schema.get('required', [])

    # Type mapping from JSON Schema to Python types
    type_mapping = {
        'string': str,
        'integer': int,
        'number': float,
        'boolean': bool,
        'array': list,
        'object': dict,
    }

    # Build Pydantic field definitions
    fields = {}
    for field_name, field_schema in properties.items():
        field_desc = field_schema.get('description', '')
        json_type = field_schema.get('type', 'string')
        field_type = type_mapping.get(json_type, str)

        if field_name in required_fields:
            # Required field: (type, Field(description=...))
            fields[field_name] = (field_type, Field(description=field_desc))
        else:
            # Optional field: (Optional[type], Field(default=None, description=...))
            fields[field_name] = (
                Optional[field_type],
                Field(default=None, description=field_desc)
            )

    # Create and return dynamic Pydantic model
    return create_model(model_name, **fields)


def tool_name_to_model_name(tool_name: str, suffix: str = "Input") -> str:
    """Convert tool name to PascalCase model name.

    Args:
        tool_name: Tool name (e.g., "check_sat_status", "pubmed_search")
        suffix: Suffix to append to model name (default: "Input")

    Returns:
        PascalCase model name (e.g., "CheckSatStatusInput", "PubmedSearchInput")

    Example:
        >>> tool_name_to_model_name("check_sat_status")
        'CheckSatStatusInput'
        >>> tool_name_to_model_name("read_file", suffix="Args")
        'ReadFileArgs'
    """
    # Replace underscores with spaces, title case, remove spaces
    pascal_case = tool_name.replace('_', ' ').title().replace(' ', '')
    return f"{pascal_case}{suffix}"
