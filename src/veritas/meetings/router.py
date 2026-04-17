"""Meeting orchestration entry point.

Thin wrapper that dispatches to the LangGraph meeting implementation. Kept
as a separate module so callers can import ``run_meeting`` without pulling
in the graph machinery until it's actually needed.
"""

from pathlib import Path
from typing import Literal

from veritas.agent import Agent
from veritas.constants import CONSISTENT_TEMPERATURE


def run_meeting(
    meeting_type: Literal["team", "individual"],
    agenda: str,
    save_dir: Path,
    save_name: str = "discussion",
    team_lead: Agent | None = None,
    team_members: tuple[Agent, ...] | None = None,
    team_member: Agent | None = None,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
    summaries: tuple[str, ...] = (),
    contexts: tuple[str, ...] = (),
    num_rounds: int = 0,
    temperature: float = CONSISTENT_TEMPERATURE,
    top_p: float | None = None,
    prompt_verbosity: str = "verbose",
    return_summary: bool = False,
    critic: Agent | None = None,
    enable_critic: bool = True,
    workflow_instruction: str = "",
    summary_instructions: str = "",
) -> str:
    """Runs a meeting with LLM agents via the LangGraph implementation.

    :param meeting_type: The type of meeting.
    :param agenda: The agenda for the meeting.
    :param save_dir: The directory to save the discussion.
    :param save_name: The name of the discussion file that will be saved.
    :param team_lead: The team lead for a team meeting (None for individual meeting).
    :param team_members: The team members for a team meeting (None for individual meeting).
    :param team_member: The team member for an individual meeting (None for team meeting).
    :param agenda_questions: The agenda questions to answer by the end of the meeting.
    :param agenda_rules: The rules for the meeting.
    :param summaries: The summaries of previous meetings.
    :param contexts: The contexts for the meeting.
    :param num_rounds: The number of rounds of discussion.
    :param temperature: The sampling temperature.
    :param prompt_verbosity: Controls INPUT prompt redundancy ("minimal", "standard", or "verbose").
    :param return_summary: Whether to return the summary of the meeting.
    :param critic: Optional custom critic agent (defaults to SCIENTIFIC_CRITIC if enable_critic=True).
    :param enable_critic: Whether to enable critic agent in individual meetings.
    :param workflow_instruction: Optional instruction describing this phase's role in a sequential workflow.
    :return: The summary of the meeting (i.e., the last message) if return_summary is True, else None.
    """
    from veritas.meetings.langgraph import run_meeting_langgraph

    return run_meeting_langgraph(
        meeting_type=meeting_type,
        agenda=agenda,
        save_dir=save_dir,
        save_name=save_name,
        team_lead=team_lead,
        team_members=team_members,
        team_member=team_member,
        agenda_questions=agenda_questions,
        agenda_rules=agenda_rules,
        summaries=summaries,
        contexts=contexts,
        num_rounds=num_rounds,
        temperature=temperature,
        top_p=top_p,
        prompt_verbosity=prompt_verbosity,
        return_summary=return_summary,
        critic=critic,
        enable_critic=enable_critic,
        workflow_instruction=workflow_instruction,
        summary_instructions=summary_instructions,
    )
