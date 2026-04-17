"""The LLM agent class."""


class Agent:
    """An LLM agent."""

    def __init__(
        self,
        title: str,
        expertise: str,
        goal: str,
        role: str,
        model: str,
        mcp_servers: tuple[str, ...] | None = None,
        available_tools: list[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> None:
        """Initializes the agent.

        :param title: The title of the agent.
        :param expertise: The expertise of the agent.
        :param goal: The goal of the agent.
        :param role: The role of the agent.
        :param model: The model to use for this agent.
        :param mcp_servers: Optional tuple of MCP server names this agent should have access to.
                           These servers must be configured in mcp_servers.json.
                           Example: ("vlm", "code_executor")
        :param available_tools: List of tool names this agent can use.
        :param temperature: Optional per-agent sampling temperature.
                           If None, falls back to the global meeting/phase temperature.
        :param top_p: Optional per-agent nucleus sampling parameter.
                     If None, falls back to the global meeting/phase top_p.
        """
        self.title = title
        self.expertise = expertise
        self.goal = goal
        self.role = role
        self.model = model
        self.mcp_servers = mcp_servers or ()
        self.available_tools = available_tools or []
        self.temperature = temperature
        self.top_p = top_p

    @property
    def prompt(self) -> str:
        """Returns the prompt for the agent."""
        return (
            f"You are a {self.title}. "
            f"Your expertise is in {self.expertise}. "
            f"Your goal is to {self.goal}. "
            f"Your role is to {self.role}."
        )

    @property
    def message(self) -> dict[str, str]:
        """Returns the message for the agent in OpenAI API form."""
        return {
            "role": "system",
            "content": self.prompt,
        }

    def __hash__(self) -> int:
        """Returns the hash of the agent."""
        return hash(self.title)

    def __eq__(self, other: object) -> bool:
        """Checks if the agent is equal to another agent (based on title)."""
        if not isinstance(other, Agent):
            return False

        return (
            self.title == other.title
            and self.expertise == other.expertise
            and self.goal == other.goal
            and self.role == other.role
            and self.model == other.model
            and self.mcp_servers == other.mcp_servers
            and self.available_tools == other.available_tools
            and self.temperature == other.temperature
            and self.top_p == other.top_p
        )

    def __str__(self) -> str:
        """Returns the string representation of the agent (i.e., the agent's title)."""
        return self.title

    def __repr__(self) -> str:
        """Returns the string representation of the agent (i.e., the agent's title)."""
        return self.title


def resolve_temperature(agent: Agent, global_temperature: float) -> float:
    """Resolve effective temperature: agent.temperature if set, else global_temperature."""
    if agent.temperature is not None:
        return agent.temperature
    return global_temperature


def resolve_top_p(agent: Agent, global_top_p: float | None) -> float | None:
    """Resolve effective top_p: agent.top_p if set, else global_top_p."""
    if agent.top_p is not None:
        return agent.top_p
    return global_top_p
