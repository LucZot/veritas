"""Prompt verbosity configuration for multi-agent workflows.

Controls token efficiency vs. context redundancy tradeoffs when working
with different LLM context window sizes. This setting controls the verbosity
of INPUT prompts (reminders, context re-injection), NOT output verbosity.

- MINIMAL: Maximum token efficiency for large context windows (64K+)
- STANDARD: Balanced approach for medium context windows (8K-32K)
- VERBOSE: Current behavior, defensive patterns for small context (2K-4K)

Usage:
    from veritas.verbosity import PromptVerbosityLevel, PromptVerbosityConfig

    # Get config from level
    config = PromptVerbosityConfig.from_level(PromptVerbosityLevel.MINIMAL)

    # Or use convenience function
    config = get_prompt_verbosity_config("minimal")

    # Check specific settings
    if config.include_expertise_reminder:
        prompt += f"Based on your expertise in {agent.expertise}..."
"""

from enum import Enum
from dataclasses import dataclass


class PromptVerbosityLevel(Enum):
    """Prompt verbosity levels controlling input prompt redundancy.

    This controls how much context is re-injected into prompts, NOT output verbosity.

    MINIMAL (64K+ context): Trust large context window, maximum efficiency
        - No per-turn role/expertise reminders
        - No agenda re-injection at final round
        - Summaries: only Recommendation + Answers + Next Steps

    STANDARD (8K-32K context): Balanced approach
        - Brief expertise reminder per turn
        - Brief agenda reference at final round
        - Summaries: full content, thinking stripped

    VERBOSE (2K-4K context): Current behavior, backward compatible
        - Full role + expertise reminders per turn
        - Full agenda re-injection at final round
        - Summaries: full content, thinking stripped
    """

    MINIMAL = "minimal"
    STANDARD = "standard"
    VERBOSE = "verbose"

    @classmethod
    def from_string(cls, value: str) -> "PromptVerbosityLevel":
        """Parse prompt verbosity level from string (case-insensitive).

        Args:
            value: String value like "minimal", "STANDARD", etc.

        Returns:
            Corresponding PromptVerbosityLevel enum member.

        Raises:
            ValueError: If value is not a valid prompt verbosity level.
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid = [v.value for v in cls]
            raise ValueError(f"Invalid prompt_verbosity '{value}'. Must be one of: {valid}")


# Backward compatibility alias
VerbosityLevel = PromptVerbosityLevel


@dataclass(frozen=True)
class PromptVerbosityConfig:
    """Configuration derived from prompt verbosity level.

    This dataclass holds all the specific settings controlled by prompt_verbosity.
    Use PromptVerbosityConfig.from_level() to create instances.

    Attributes:
        level: The prompt verbosity level this config represents.
        include_expertise_reminder: Include "Based on your expertise in X" in prompts.
        include_round_counter: Include "(round N of M)" in prompts.
        include_turn_boundary: Include "Respond ONLY as {title}" warning.
        include_goal_reminder: Include agent's goal in per-turn prompts.
        reinject_agenda: Re-inject full agenda at final round.
        reinject_questions_rules: Re-inject questions/rules at final round.
        strip_thinking: Strip thinking process from loaded summaries.
        extract_recommendations_only: Extract only Recommendation+Answers+NextSteps.
        use_minimal_agenda: Use minimal agenda (True for MINIMAL/STANDARD, False for VERBOSE).
    """

    level: PromptVerbosityLevel

    # Per-turn prompt settings
    include_expertise_reminder: bool
    include_round_counter: bool
    include_turn_boundary: bool
    include_goal_reminder: bool

    # Final round re-injection
    reinject_agenda: bool
    reinject_questions_rules: bool

    # Summary handling
    strip_thinking: bool
    extract_recommendations_only: bool

    # Agenda verbosity
    use_minimal_agenda: bool

    @classmethod
    def from_level(cls, level: PromptVerbosityLevel) -> "PromptVerbosityConfig":
        """Create configuration from prompt verbosity level.

        Args:
            level: The prompt verbosity level to create config for.

        Returns:
            PromptVerbosityConfig with appropriate settings for the level.
        """
        configs = {
            PromptVerbosityLevel.MINIMAL: cls(
                level=level,
                include_expertise_reminder=False,
                include_round_counter=False,
                include_turn_boundary=False,
                include_goal_reminder=False,
                reinject_agenda=False,
                reinject_questions_rules=False,
                strip_thinking=True,
                extract_recommendations_only=True,
                use_minimal_agenda=True,
            ),
            PromptVerbosityLevel.STANDARD: cls(
                level=level,
                include_expertise_reminder=True,
                include_round_counter=True,
                include_turn_boundary=True,
                include_goal_reminder=False,
                reinject_agenda=False,
                reinject_questions_rules=False,
                strip_thinking=True,
                extract_recommendations_only=False,
                use_minimal_agenda=True,
            ),
            PromptVerbosityLevel.VERBOSE: cls(
                level=level,
                include_expertise_reminder=True,
                include_round_counter=True,
                include_turn_boundary=True,
                include_goal_reminder=True,
                reinject_agenda=True,
                reinject_questions_rules=True,
                strip_thinking=True,
                extract_recommendations_only=False,
                use_minimal_agenda=False,
            ),
        }
        return configs[level]


# Backward compatibility alias
VerbosityConfig = PromptVerbosityConfig

# Default prompt verbosity level (VERBOSE for backward compatibility)
DEFAULT_PROMPT_VERBOSITY = PromptVerbosityLevel.VERBOSE
DEFAULT_VERBOSITY = DEFAULT_PROMPT_VERBOSITY  # Backward compatibility


def get_prompt_verbosity_config(
    prompt_verbosity: str | PromptVerbosityLevel | None = None,
) -> PromptVerbosityConfig:
    """Get prompt verbosity config from various input types.

    Convenience function that handles string, enum, or None input.

    Args:
        prompt_verbosity: Prompt verbosity as string, enum, or None (uses default).

    Returns:
        PromptVerbosityConfig for the specified level.
    """
    if prompt_verbosity is None:
        return PromptVerbosityConfig.from_level(DEFAULT_PROMPT_VERBOSITY)
    if isinstance(prompt_verbosity, str):
        return PromptVerbosityConfig.from_level(
            PromptVerbosityLevel.from_string(prompt_verbosity)
        )
    return PromptVerbosityConfig.from_level(prompt_verbosity)


# Backward compatibility alias
def get_verbosity_config(
    verbosity: str | PromptVerbosityLevel | None = None,
) -> PromptVerbosityConfig:
    """Deprecated: Use get_prompt_verbosity_config instead."""
    return get_prompt_verbosity_config(verbosity)