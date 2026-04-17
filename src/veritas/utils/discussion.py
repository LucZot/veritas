"""Discussion formatting and manipulation utilities."""

import json
from pathlib import Path
from typing import Any


def extract_message_with_reasoning(response: Any) -> str:
    """Extract message content from LLM response, including thinking/reasoning and tool calls.

    For models with thinking mode enabled, this combines the reasoning process
    with the final answer for complete transparency in meeting transcripts.
    Also includes tool call information when present.

    The SHOW_THINKING_IN_OUTPUT environment variable controls whether thinking
    process is included in the output:
    - "true" (default): Include thinking process in output
    - "false": Only show final response (cleaner output)

    :param response: LangChain response object (AIMessage)
    :return: Message content, optionally prefixed with reasoning and tool calls
    """
    import os

    # Check if thinking process should be shown in output
    show_thinking = os.getenv("SHOW_THINKING_IN_OUTPUT", "true").lower() == "true"

    parts = []

    # Get reasoning first (thinking before acting) - only if enabled
    if show_thinking:
        if hasattr(response, 'additional_kwargs') and response.additional_kwargs:
            reasoning = response.additional_kwargs.get('reasoning_content')
            if reasoning:
                parts.append(f"**[Thinking Process]**\n\n{reasoning}")

    # Get tool calls after reasoning (actions based on thinking)
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_call_descriptions = []
        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name", "unknown_tool")
            tool_args = tool_call.get("args", {})
            if tool_args:
                args_str = ", ".join(f"{k}={v}" for k, v in tool_args.items())
                tool_call_descriptions.append(f"[Calling {tool_name}({args_str})]")
            else:
                tool_call_descriptions.append(f"[Calling {tool_name}()]")
        if tool_call_descriptions:
            parts.append(f"**[Tool Calls]**\n\n" + "\n".join(tool_call_descriptions))

    # Get main content
    content = response.content if hasattr(response, 'content') else str(response)
    if content:
        if parts:  # If we have reasoning or tool calls, label the content
            parts.append(f"**[Response]**\n\n{content}")
        else:
            parts.append(content)

    return "\n\n".join(parts) if parts else ""


def convert_messages_to_discussion(
    messages: list[dict], assistant_id_to_title: dict[str, str]
) -> list[dict[str, str]]:
    """Converts OpenAI messages into discussion format (list of message dictionaries).

    :param messages: The messages to convert.
    :param assistant_id_to_title: A dictionary mapping assistant IDs to titles.
    :return: The discussion format (list of message dictionaries).
    """
    return [
        {
            "agent": (
                assistant_id_to_title[message["assistant_id"]]
                if message["assistant_id"] is not None
                else "User"
            ),
            "message": message["content"][0]["text"]["value"],
        }
        for message in messages
    ]


def get_summary(discussion: list[dict[str, str]]) -> str:
    """Get the summary from a discussion.

    :param discussion: The discussion to extract the summary from.
    :return: The summary.
    """
    return discussion[-1]["message"]


def strip_thinking_process(message: str) -> str:
    """Strip thinking process and tool calls sections from a message, keeping only the response.

    This is useful when loading summaries for downstream agents - they only need
    the response content, not the verbose thinking process that was used to generate it.

    :param message: Message that may contain **[Thinking Process]**, **[Tool Calls]**, and **[Response]** sections
    :return: Just the response content without thinking process or tool call details
    """
    import re

    # If the message has the **[Response]** marker, extract only that section
    response_match = re.search(r'\*\*\[Response\]\*\*\s*\n\n(.*)', message, re.DOTALL)
    if response_match:
        return response_match.group(1).strip()

    # Otherwise, check if it has thinking or tool call sections and remove them
    # Pattern: Remove everything from **[Thinking Process]** to just before **[Response]**
    cleaned = re.sub(
        r'\*\*\[Thinking Process\]\*\*.*?(?=\*\*\[Response\]\*\*|\Z)',
        '',
        message,
        flags=re.DOTALL
    )

    # Pattern: Remove **[Tool Calls]** section
    cleaned = re.sub(
        r'\*\*\[Tool Calls\]\*\*.*?(?=\*\*\[Response\]\*\*|\Z)',
        '',
        cleaned,
        flags=re.DOTALL
    )

    # If we still have the **[Response]** marker after cleaning, extract just the content
    response_match = re.search(r'\*\*\[Response\]\*\*\s*\n\n(.*)', cleaned, re.DOTALL)
    if response_match:
        return response_match.group(1).strip()

    # If no markers found, return original message (it doesn't have thinking sections)
    return message.strip()


def extract_recommendations_section(message: str) -> str:
    """Extract only Recommendation, Answers, and Next Steps sections from a summary.

    This provides a condensed summary for token-efficient inter-phase communication.
    Used when verbosity is set to MINIMAL.

    :param message: Full summary message with sections like Agenda, Team Member Input,
                   Recommendation, Answers, Next Steps
    :return: Condensed summary with only Recommendation, Answers, and Next Steps
    """
    import re

    sections = []

    # Extract ### Recommendation section
    rec_match = re.search(
        r'###\s*Recommendation\s*\n(.*?)(?=###|\Z)',
        message,
        re.DOTALL | re.IGNORECASE
    )
    if rec_match:
        sections.append(f"### Recommendation\n{rec_match.group(1).strip()}")

    # Extract ### Answers section (if present)
    answers_match = re.search(
        r'###\s*Answers\s*\n(.*?)(?=###|\Z)',
        message,
        re.DOTALL | re.IGNORECASE
    )
    if answers_match:
        sections.append(f"### Answers\n{answers_match.group(1).strip()}")

    # Extract ### Next Steps section
    steps_match = re.search(
        r'###\s*Next\s*Steps\s*\n(.*?)(?=###|\Z)',
        message,
        re.DOTALL | re.IGNORECASE
    )
    if steps_match:
        sections.append(f"### Next Steps\n{steps_match.group(1).strip()}")

    if sections:
        return "\n\n".join(sections)

    # Fallback to full message if no sections found
    return message


def build_correction_summary_sources(
    prior_paths: list[Path] | tuple[Path, ...],
    current_phase_dir: Path,
) -> list[Path]:
    """Build correction-summary sources with the latest same-phase discussion.

    Ensures correction meetings receive summaries from prior phases plus the
    current phase's latest canonical discussion when present.
    """
    sources: list[Path] = []
    seen: set[str] = set()

    for path in list(prior_paths) + [current_phase_dir / "discussion.json"]:
        if not path or not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        sources.append(path)

    return sources


def load_summaries(
    discussion_paths: list[Path],
    strip_thinking: bool = True,
    extract_recommendations_only: bool = False,
) -> tuple[str, ...]:
    """Load summaries from a list of discussion paths.

    :param discussion_paths: The paths to the discussion JSON files. The summary is the last entry in the discussion.
    :param strip_thinking: If True (default), remove thinking process sections from summaries.
                          This is recommended for downstream agents who only need the response content.
    :param extract_recommendations_only: If True, extract only Recommendation, Answers, and Next Steps
                                        sections for maximum token efficiency. Default False.
    :return: A tuple of summaries.
    """
    summaries = []
    for discussion_path in discussion_paths:
        with open(discussion_path, "r") as file:
            discussion = json.load(file)
        summary = get_summary(discussion)

        # Strip thinking process if requested (default behavior)
        if strip_thinking:
            summary = strip_thinking_process(summary)

        # Extract only key sections if requested (for MINIMAL verbosity)
        if extract_recommendations_only:
            summary = extract_recommendations_section(summary)

        summaries.append(summary)

    return tuple(summaries)
