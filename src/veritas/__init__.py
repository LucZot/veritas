"""VERITAS package."""

from veritas.__about__ import __version__
from veritas.agent import Agent
from veritas.meetings import run_meeting


__all__ = [
    "__version__",
    "Agent",
    "run_meeting",
]
