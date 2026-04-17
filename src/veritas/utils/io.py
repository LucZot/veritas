"""File I/O utilities for saving and loading meeting data."""

import json
from pathlib import Path
from shutil import copy2


def print_cost_and_time(
    token_counts: dict[str, int],
    model: str,
    elapsed_time: float,
) -> None:
    """Print token counts and elapsed time for a meeting.

    :param token_counts: Dictionary with 'input', 'output', 'tool', 'max' counts
    :param model: Model name (informational only)
    :param elapsed_time: Elapsed time in seconds
    """
    print(f"Input token count: {token_counts['input']:,}")
    print(f"Output token count: {token_counts['output']:,}")
    print(f"Tool token count: {token_counts['tool']:,}")
    print(f"Max token length: {token_counts['max']:,}")
    print(f"Model: {model}")
    print(f"Time: {int(elapsed_time // 60)}:{int(elapsed_time % 60):02d}")


def archive_meeting_artifacts(save_dir: Path, save_name: str, archive_name: str) -> None:
    """Archive existing meeting artifacts under a versioned name before overwrite.

    Copies the canonical JSON/Markdown discussion and execution metadata, when present,
    so correction passes do not destroy provenance of earlier attempts.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    suffixes = [('.json', '.json'), ('.md', '.md'), ('_execution_metadata.json', '_execution_metadata.json')]
    for source_suffix, archive_suffix in suffixes:
        source = save_dir / f"{save_name}{source_suffix}"
        if not source.exists():
            continue
        target = save_dir / f"{archive_name}{archive_suffix}"
        copy2(source, target)


def save_meeting(
    save_dir: Path, save_name: str, discussion: list[dict[str, str]]
) -> None:
    """Save a meeting discussion to JSON and Markdown files.

    :param save_dir: The directory to save the discussion.
    :param save_name: The name of the discussion file that will be saved.
    :param discussion: The discussion to save.
    """
    # Create the save directory if it does not exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the discussion as JSON
    with open(save_dir / f"{save_name}.json", "w") as f:
        json.dump(discussion, f, indent=4)

    # Save the discussion as Markdown
    with open(save_dir / f"{save_name}.md", "w") as file:
        for i, turn in enumerate(discussion):
            # Add double separator line before each speaker (except first)
            # This distinguishes speaker boundaries from agents' own formatting
            if i > 0:
                file.write("---\n\n---\n\n")
            file.write(f"## {turn['agent']}\n\n{turn['message']}\n\n")
