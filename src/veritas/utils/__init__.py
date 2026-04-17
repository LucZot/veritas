"""VERITAS utility functions.

This package organizes utilities by concern:
- tokens: Token counting for discussions
- discussion: Discussion formatting and manipulation
- io: File I/O for saving meetings
- schema_conversion: JSON Schema to Pydantic conversion for tool handling
"""

# Token counting
from veritas.utils.tokens import (
    count_tokens,
    update_token_counts,
    count_discussion_tokens,
)

# Discussion formatting
from veritas.utils.discussion import (
    extract_message_with_reasoning,
    build_correction_summary_sources,
    convert_messages_to_discussion,
    get_summary,
    load_summaries,
)

# File I/O
from veritas.utils.io import (
    archive_meeting_artifacts,
    save_meeting,
    print_cost_and_time,
)

# Schema conversion
from veritas.utils.schema_conversion import (
    json_schema_to_pydantic,
    tool_name_to_model_name,
)

__all__ = [
    # Tokens
    "count_tokens",
    "update_token_counts",
    "count_discussion_tokens",
    # Discussion
    "extract_message_with_reasoning",
    "build_correction_summary_sources",
    "convert_messages_to_discussion",
    "get_summary",
    "load_summaries",
    # I/O
    "archive_meeting_artifacts",
    "save_meeting",
    "print_cost_and_time",
    # Schema conversion
    "json_schema_to_pydantic",
    "tool_name_to_model_name",
]
