"""Meeting validation utilities.

Shared validation logic for all meeting orchestrators.
"""

from typing import Literal

from veritas.agent import Agent


def validate_meeting_args(
    meeting_type: Literal["team", "individual"],
    team_lead: Agent | None = None,
    team_members: tuple[Agent, ...] | None = None,
    team_member: Agent | None = None,
) -> None:
    """Validate meeting arguments based on meeting type.

    :param meeting_type: The type of meeting ("team" or "individual")
    :param team_lead: The team lead (required for team meetings)
    :param team_members: The team members (required for team meetings)
    :param team_member: The individual agent (required for individual meetings)
    :raises ValueError: If arguments are invalid for the meeting type
    """
    if meeting_type == "team":
        if team_lead is None or team_members is None or len(team_members) == 0:
            raise ValueError("Team meeting requires team lead and team members")
        if team_member is not None:
            raise ValueError("Team meeting does not require individual team member")
        if team_lead in team_members:
            raise ValueError("Team lead must be separate from team members")
        if len(set(team_members)) != len(team_members):
            raise ValueError("Team members must be unique")
    elif meeting_type == "individual":
        if team_member is None:
            raise ValueError("Individual meeting requires individual team member")
        if team_lead is not None or team_members is not None:
            raise ValueError(
                "Individual meeting does not require team lead or team members"
            )
    else:
        raise ValueError(f"Invalid meeting type: {meeting_type}")
