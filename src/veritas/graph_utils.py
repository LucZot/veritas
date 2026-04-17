"""Utility functions for building and managing LangGraph workflows.

This module provides helper functions for constructing, visualizing, and
debugging LangGraph-based meeting workflows.
"""

from typing import Any
from pathlib import Path
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from veritas.graph_state import MeetingState, IndividualMeetingState
from veritas.utils import extract_message_with_reasoning


def messages_to_discussion(
    messages: list[BaseMessage],
    agent_name_map: dict[str, str] = None,
) -> list[dict[str, str]]:
    """Convert LangChain messages to VERITAS discussion format.

    Args:
        messages: List of LangChain BaseMessage objects.
        agent_name_map: Optional mapping of message IDs/names to agent titles.

    Returns:
        Discussion format: [{"agent": "...", "message": "..."}, ...]

    Example:
        >>> msgs = [HumanMessage(content="Hi"), AIMessage(content="Hello")]
        >>> discussion = messages_to_discussion(msgs)
        >>> discussion[0]
        {'agent': 'Human User', 'message': 'Hi'}
    """
    discussion = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            # Skip system messages in discussion output
            continue
        elif isinstance(msg, HumanMessage):
            # User messages (prompts, tool outputs, etc.)
            discussion.append({"agent": "Human User", "message": msg.content})
        elif isinstance(msg, AIMessage):
            # Agent responses
            agent_name = "Assistant"
            if hasattr(msg, "name") and msg.name:
                agent_name = msg.name
            elif agent_name_map and hasattr(msg, "id"):
                agent_name = agent_name_map.get(msg.id, "Assistant")

            # Extract content including thinking/reasoning and tool calls
            content = extract_message_with_reasoning(msg)

            # Only add if there's actual content
            if content:
                discussion.append({"agent": agent_name, "message": content})

    return discussion


def extract_final_state_discussion(
    state: MeetingState | IndividualMeetingState,
) -> list[dict[str, str]]:
    """Extract discussion from final meeting state.

    Args:
        state: Final state from graph execution.

    Returns:
        Discussion in VERITAS format.
    """
    # Build agent name mapping if available
    agent_name_map = {}

    return messages_to_discussion(state["messages"], agent_name_map)


def add_references_to_discussion(
    discussion: list[dict[str, str]],
    state: MeetingState | IndividualMeetingState,
) -> list[dict[str, str]]:
    """No-op: PubMed reference tracking is not enabled in the default workflow."""
    return discussion


def visualize_graph(graph: Any, output_path: Path = None) -> None:
    """Visualize a LangGraph StateGraph as a diagram.

    Requires graphviz and pygraphviz to be installed.

    Args:
        graph: Compiled LangGraph StateGraph.
        output_path: Optional path to save diagram (PNG format).

    Example:
        >>> from langgraph.graph import StateGraph
        >>> graph = StateGraph(MeetingState)
        >>> # ... add nodes and edges ...
        >>> app = graph.compile()
        >>> visualize_graph(app, Path("./meeting_graph.png"))
    """
    try:
        from langchain_core.runnables.graph import MermaidDrawMethod

        # Generate mermaid diagram
        mermaid_png = graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(mermaid_png)
            print(f"Graph visualization saved to: {output_path}")
        else:
            print("Graph visualization (requires display):")
            print(graph.get_graph().draw_mermaid())

    except ImportError:
        print("Warning: Could not visualize graph. Install graphviz and pygraphviz:")
        print("  pip install pygraphviz")
        print("  Or use Mermaid: graph.get_graph().draw_mermaid()")
    except Exception as e:
        print(f"Warning: Could not create graph visualization: {e}")
        print("Graph structure:")
        print(graph.get_graph())


def print_graph_structure(graph: Any) -> None:
    """Print the structure of a LangGraph for debugging.

    Args:
        graph: Compiled LangGraph StateGraph.
    """
    print("\n" + "=" * 70)
    print("LangGraph Structure")
    print("=" * 70)

    graph_def = graph.get_graph()

    print("\nNodes:")
    for node in graph_def.nodes:
        print(f"  - {node}")

    print("\nEdges:")
    for edge in graph_def.edges:
        print(f"  - {edge}")

    print("=" * 70 + "\n")


def estimate_state_size(state: MeetingState | IndividualMeetingState) -> dict[str, int]:
    """Estimate memory usage of state components.

    Useful for optimizing long-running meetings.

    Args:
        state: Meeting state to analyze.

    Returns:
        Dictionary with size estimates in bytes for each component.
    """
    import sys

    size_info = {}

    # Estimate message history size
    messages_size = sum(
        sys.getsizeof(msg.content) if hasattr(msg, "content") else 0
        for msg in state.get("messages", [])
    )
    size_info["messages_bytes"] = messages_size

    # Estimate tool outputs size
    tool_outputs_size = sum(
        sys.getsizeof(str(output)) for output in state.get("tool_outputs", [])
    )
    size_info["tool_outputs_bytes"] = tool_outputs_size

    # Estimate reference registry size

    # Total estimate
    size_info["total_bytes"] = sum(size_info.values())
    size_info["total_kb"] = size_info["total_bytes"] / 1024
    size_info["total_mb"] = size_info["total_kb"] / 1024

    return size_info


