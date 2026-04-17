"""LangGraph-based meeting orchestrator for VERITAS.

This module implements graph-based meeting workflows using LangGraph.
It provides the same interface as run_meeting.py and run_meeting_langchain.py
but uses state graphs for more flexible orchestration.

Key advantages over LangChain implementation:
- Graph-based routing (vs linear loops)
- Parallel agent execution capability
- Built-in checkpointing and state management
- Conditional routing based on discussion quality
- Visual graph debugging
"""

import time
from pathlib import Path
from typing import Literal

from tqdm import tqdm
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from veritas.agent import Agent
from veritas.config import get_ollama_config
from veritas.constants import CONSISTENT_TEMPERATURE
from veritas.graph_state import (
    MeetingState,
    IndividualMeetingState,
    create_initial_team_meeting_state,
    create_initial_individual_meeting_state,
)
from veritas.graph import (
    create_team_lead_node,
    create_team_member_node,
    create_agent_node,
    create_individual_agent_node,
    create_critic_node,
    create_team_critic_node,
    create_tool_execution_node,
    increment_round_node,
    should_continue_discussion,
    should_critic_respond,
    should_critic_review_team,
    should_execute_tools,
    route_after_tools,
    route_after_team_lead,
)
from veritas.graph_utils import (
    extract_final_state_discussion,
    add_references_to_discussion,
    print_graph_structure,
)
from veritas.langgraph import (
    build_team_meeting_graph,
    build_individual_meeting_graph,
)
from veritas.prompts import (
    team_meeting_start_prompt,
    individual_meeting_start_prompt,
    SCIENTIFIC_CRITIC,
)
from veritas.utils import (
    count_discussion_tokens,
    print_cost_and_time,
    save_meeting,
    get_summary,
)
from veritas.meetings.validation import validate_meeting_args


def run_meeting_langgraph(
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
    enable_checkpointing: bool = False,
    debug_graph: bool = False,
    critic: Agent | None = None,
    enable_critic: bool = True,
    workflow_instruction: str = "",
    summary_instructions: str = "",
) -> str:
    """Run a meeting using LangGraph state machine.

    This function has the same signature as run_meeting() and run_meeting_langchain()
    but uses LangGraph for orchestration.

    Args:
        meeting_type: "team" or "individual"
        agenda: The meeting agenda/topic
        save_dir: Directory to save discussion
        save_name: Name for saved files
        team_lead: Leader agent (for team meetings)
        team_members: Member agents (for team meetings)
        team_member: Single agent (for individual meetings)
        agenda_questions: Questions to answer
        agenda_rules: Rules to follow
        summaries: Previous meeting summaries
        contexts: Additional context
        num_rounds: Number of discussion rounds
        temperature: Sampling temperature (0-1)
        return_summary: Return summary string
        enable_checkpointing: Enable state checkpointing for resume capability
        debug_graph: Print graph structure for debugging
        enable_critic: Enable critic agent in individual meetings (default True)

    Returns:
        Summary string if return_summary=True, else empty string

    Example:
        >>> from pathlib import Path
        >>> from veritas.agent import Agent
        >>>
        >>> agent = Agent("Scientist", "research", "discover", "lead", "llama3.1")
        >>> run_meeting_langgraph(
        ...     meeting_type="individual",
        ...     team_member=agent,
        ...     agenda="Propose research ideas",
        ...     save_dir=Path("./meetings"),
        ...     num_rounds=1
        ... )
    """
    # Validate meeting arguments
    validate_meeting_args(meeting_type, team_lead, team_members, team_member)

    # Start timing
    start_time = time.time()

    # Get Ollama configuration
    ollama_config = get_ollama_config()

    # Build the appropriate graph based on meeting type
    if meeting_type == "team":
        # Use custom critic if provided, otherwise use default SCIENTIFIC_CRITIC
        if critic is None and enable_critic:
            critic = SCIENTIFIC_CRITIC

        graph_app = build_team_meeting_graph(
            team_lead=team_lead,
            team_members=team_members,
            ollama_config=ollama_config,
            temperature=temperature,
            top_p=top_p,
            enable_checkpointing=enable_checkpointing,
            critic=critic,
            enable_critic=enable_critic,
        )

        # Create initial state
        initial_state = create_initial_team_meeting_state(
            team_lead_name=team_lead.title,
            team_member_names=[m.title for m in team_members],
            critic_name=critic.title if (enable_critic and critic) else "",
            agenda=agenda,
            max_rounds=num_rounds,
            agenda_questions=agenda_questions,
            agenda_rules=agenda_rules,
            summaries=summaries,
            contexts=contexts,
            temperature=temperature,
            prompt_verbosity=prompt_verbosity,
            summary_instructions=summary_instructions,
        )

        # Add initial meeting prompt to state
        initial_prompt = team_meeting_start_prompt(
            team_lead=team_lead,
            team_members=team_members,
            agenda=agenda,
            agenda_questions=agenda_questions,
            agenda_rules=agenda_rules,
            summaries=summaries,
            contexts=contexts,
            num_rounds=num_rounds,
            workflow_instruction=workflow_instruction,
        )
        initial_state["messages"] = [HumanMessage(content=initial_prompt)]

    else:  # individual meeting
        # Respect caller-provided critic (e.g., phase-aware critic); fall back to default.
        if critic is None:
            critic = SCIENTIFIC_CRITIC
        graph_app = build_individual_meeting_graph(
            agent=team_member,
            critic=critic,
            ollama_config=ollama_config,
            temperature=temperature,
            top_p=top_p,
            enable_checkpointing=enable_checkpointing,
            enable_critic=enable_critic,
        )

        # Create initial state
        initial_state = create_initial_individual_meeting_state(
            agent_name=team_member.title,
            critic_name=critic.title,
            agenda=agenda,
            max_rounds=num_rounds,
            agenda_questions=agenda_questions,
            agenda_rules=agenda_rules,
            summaries=summaries,
            contexts=contexts,
            temperature=temperature,
            prompt_verbosity=prompt_verbosity,
            summary_instructions=summary_instructions,
        )

        # Add initial prompt to state
        initial_prompt = individual_meeting_start_prompt(
            team_member=team_member,
            agenda=agenda,
            agenda_questions=agenda_questions,
            agenda_rules=agenda_rules,
            summaries=summaries,
            contexts=contexts,
            workflow_instruction=workflow_instruction,
        )
        initial_state["messages"] = [HumanMessage(content=initial_prompt)]

    # Debug: Print graph structure if requested
    if debug_graph:
        print_graph_structure(graph_app)

    # Execute the graph
    from veritas.graph.utils import _print_round_header
    print("🚀 Running LangGraph meeting workflow...")
    print(f"   Max rounds: {num_rounds}")
    print(f"   Checkpointing: {'enabled' if enable_checkpointing else 'disabled'}")
    print()
    _print_round_header(1, num_rounds)

    # Run the graph with progress tracking
    config = {"recursion_limit": 100}
    if enable_checkpointing:
        config["thread_id"] = f"meeting_{save_name}_{int(time.time())}"

    # Execute the graph and get the full final state
    # Two approaches:
    # 1. Use invoke() - simple but no progress tracking
    # 2. Use stream() + checkpointer - allows progress tracking + get_state()

    try:
        if enable_checkpointing:
            # With checkpointing: stream for progress, then get_state() for full state
            for state_snapshot in tqdm(
                graph_app.stream(initial_state, config=config),
                desc="Graph execution",
                unit=" nodes"
            ):
                pass  # Just iterate through for progress tracking

            # Get full accumulated state from checkpointer
            final_state_value = graph_app.get_state(config).values
        else:
            # Without checkpointing: use invoke() which returns full final state
            final_state_value = graph_app.invoke(initial_state, config=config)

    except Exception as e:
        print(f"\n❌ Error during graph execution: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise

    if not final_state_value:
        raise RuntimeError("Graph execution produced no output")

    # Convert graph state to discussion format
    discussion = extract_final_state_discussion(final_state_value)

    # Add references if any were found
    discussion = add_references_to_discussion(discussion, final_state_value)

    # Count tokens
    token_counts = count_discussion_tokens(discussion)

    # Add tool token count if available (default to 0)
    if final_state_value.get("tool_outputs"):
        from veritas.utils import count_tokens
        import json

        tool_token_count = 0
        for output in final_state_value["tool_outputs"]:
            output_val = output.get("output", "")
            # Convert to string if not already
            if isinstance(output_val, str):
                output_str = output_val
            elif output_val is None:
                output_str = ""
            else:
                # Convert dict/list/other to JSON string
                output_str = json.dumps(output_val)
            tool_token_count += count_tokens(output_str)

        token_counts["tool"] = tool_token_count
    else:
        token_counts["tool"] = 0

    # Print cost and time
    model = team_lead.model if meeting_type == "team" else team_member.model
    print_cost_and_time(
        token_counts=token_counts,
        model=model,
        elapsed_time=time.time() - start_time,
    )

    # Save the discussion
    save_meeting(
        save_dir=save_dir,
        save_name=save_name,
        discussion=discussion,
    )

    # Persist code execution metadata for auditability
    code_execs = final_state_value.get("code_executions", [])
    if code_execs:
        import json as _json
        exec_meta = {
            "total_executions": len(code_execs),
            "successful_executions": sum(1 for e in code_execs if e.get("success")),
            "failed_executions": sum(1 for e in code_execs if not e.get("success")),
            "executions": [
                {
                    "index": i,
                    "agent": e.get("agent", ""),
                    "filename": e.get("filename", ""),
                    "success": e.get("success", False),
                    "block_number": e.get("block_number", 1),
                    "total_blocks": e.get("total_blocks", 1),
                }
                for i, e in enumerate(code_execs)
            ],
        }
        meta_path = save_dir / f"{save_name}_execution_metadata.json"
        with open(meta_path, "w") as _f:
            _json.dump(exec_meta, _f, indent=2)
        print(f"   📋 Execution metadata: {len(code_execs)} trial(s), "
              f"{exec_meta['successful_executions']} succeeded → {meta_path.name}")

    # Return summary if requested
    if return_summary:
        return get_summary(discussion)
    return ""
