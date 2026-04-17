"""Code extraction and execution node for LangGraph workflows.

Handles extraction of Python code from agent messages and execution via MCP backend.
This enables agents to write code directly in their discussion messages instead of
using tool calls.

State persistence works at the phase level: variables survive across all code
executions within a single phase run, stored in the workspace's .python_state.dill.
"""

import re
import os
import asyncio
import time
import textwrap
from typing import Any
from dataclasses import dataclass
from pathlib import Path
from langchain_core.messages import HumanMessage
from veritas.agent import Agent
from veritas.constants import MAX_CODE_EXECUTIONS_PER_ROUND
from veritas.graph_state import MeetingState, IndividualMeetingState
from veritas.utils.model_detection import detect_model_type


@dataclass
class CodeBlock:
    """Represents a single code block extracted from a message."""
    code: str
    filename: str
    block_number: int
    total_blocks: int


@dataclass
class ExecutionResult:
    """Represents the result of executing a code block."""
    agent: str
    filename: str
    code: str
    result: str
    success: bool
    block_number: int
    total_blocks: int


def _default_filename(agent_name: str) -> str | None:
    agent = agent_name.lower()
    if "coding ml statistician" in agent:
        return "code/analysis.py"
    if agent == "ml statistician":
        return "code/power_analysis.py"
    if "coding medical imaging specialist" in agent:
        return "code/run_segmentation.py"
    return None


def _extract_code_blocks(content: str, agent_name: str) -> list[CodeBlock]:
    """Extract Python code blocks from message content.

    Args:
        content: Message content that may contain code blocks
        agent_name: Name of agent for auto-generating filenames

    Returns:
        List of CodeBlock objects, empty if no code blocks found
    """
    # Pattern to match Python code blocks
    pattern = r'```python\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        return []

    blocks = []
    num_blocks = len(matches)

    for block_idx, code_text in enumerate(matches):
        block_num = block_idx + 1
        code_text = textwrap.dedent(code_text).strip("\n")

        # Extract filename from code comment
        filename_match = re.search(r'#\s*filename:\s*(\S+\.py)', code_text)
        if filename_match:
            filename = filename_match.group(1)
            if not filename.startswith('code/'):
                filename = f"code/{filename}"
        else:
            filename = _default_filename(agent_name)
            if not filename:
                timestamp_ms = int(time.time() * 1000)
                filename = f"code/{timestamp_ms}_code.py"

        blocks.append(CodeBlock(
            code=code_text,
            filename=filename,
            block_number=block_num,
            total_blocks=num_blocks
        ))

    return blocks


def _summarize_outputs(outputs: dict) -> str:
    """Summarize the outputs dictionary for concise model feedback.

    Truncates large collections to show previews instead of overwhelming models
    with massive JSON output. Models work better with summaries than raw data.

    Args:
        outputs: Dictionary from data/outputs.json or similar JSON files

    Returns:
        Human-readable summary string with truncated collections
    """
    import pandas as pd
    import numpy as np

    if not outputs:
        return ""

    summaries = []
    for key, value in outputs.items():
        # Handle pandas DataFrames
        if isinstance(value, dict) and all(k in value for k in ['Patient_ID', 'Group', 'Biomarker_Level']):
            # This looks like a DataFrame serialized as dict
            try:
                df = pd.DataFrame(value)
                summaries.append(f"  - {key}: DataFrame ({len(df)} rows, {len(df.columns)} columns: {', '.join(df.columns.tolist())})")
            except (ValueError, KeyError, TypeError):
                summaries.append(f"  - {key}: dict with {len(value)} keys")
        # Handle lists/arrays - TRUNCATE if large
        elif isinstance(value, (list, tuple)):
            if len(value) > 10:
                # Show first 3, last 2, with ellipsis
                preview = [value[0], value[1], value[2], '...', value[-2], value[-1]]
                preview_str = ', '.join(str(v) if not isinstance(v, float) else f'{v:.2f}' for v in preview)
                summaries.append(f"  - {key}: [{preview_str}] (length {len(value)})")
            else:
                # Format floats to 2 decimals
                formatted = [v if not isinstance(v, float) else f'{v:.2f}' for v in value]
                summaries.append(f"  - {key}: {formatted}")
        # Handle numpy arrays
        elif hasattr(value, 'shape'):
            summaries.append(f"  - {key}: numpy array (shape {value.shape})")
        # Handle dicts - TRUNCATE if large
        elif isinstance(value, dict):
            if len(value) > 5:
                # Show first 3 keys, last 2 keys, with ellipsis
                keys_list = list(value.keys())
                sample_keys = keys_list[:3] + ['...'] + keys_list[-2:]
                summaries.append(f"  - {key}: dict ({len(value)} keys: {sample_keys})")
            else:
                # Format float values to 2 decimals
                formatted_dict = {k: (v if not isinstance(v, float) else f'{v:.2f}') for k, v in value.items()}
                summaries.append(f"  - {key}: {formatted_dict}")
        # Handle scalars - format floats
        else:
            if isinstance(value, float):
                summaries.append(f"  - {key}: {value:.4f}")
            else:
                summaries.append(f"  - {key}: {value}")

    return "\n".join(summaries) if summaries else ""


def _get_versioned_filename(base_path: Path, filename: str) -> tuple[str, str]:
    """Get a versioned filename if the original already exists.

    Args:
        base_path: Directory where the file will be saved
        filename: Original filename (e.g., "analysis.py")

    Returns:
        Tuple of (actual_filename, version_note)
        - actual_filename: Versioned filename if collision exists, otherwise original
        - version_note: Empty string or note about versioning for the model
    """
    from pathlib import Path

    original_path = base_path / filename

    # If file doesn't exist, use original name
    if not original_path.exists():
        return filename, ""

    # File exists - need to version it
    stem = original_path.stem  # e.g., "analysis"
    suffix = original_path.suffix  # e.g., ".py"
    parent = original_path.parent

    # Find next available version number
    version = 2
    while True:
        versioned_name = f"{stem}_v{version}{suffix}"
        versioned_path = parent / versioned_name
        if not versioned_path.exists():
            version_note = (
                f"ℹ️  Auto-versioned: '{filename}' → '{versioned_name}' "
                f"(original file already exists)"
            )
            return versioned_name, version_note
        version += 1


async def _execute_code_block(block: CodeBlock, workspace_base: str) -> str:
    """Execute a single code block via MCP backend.

    Args:
        block: CodeBlock to execute
        workspace_base: Workspace directory for this phase

    Returns:
        Execution result as string
    """
    from veritas.mcp import MCPClientManager, load_mcp_config
    from pathlib import Path

    mcp_manager = MCPClientManager()

    # Load MCP config and register code_execution server
    mcp_config = load_mcp_config()
    server_lookup = {server['name']: server for server in mcp_config.get('servers', [])}

    if 'code_execution' not in server_lookup:
        raise Exception("code_execution server not found in mcp_servers.json")

    server_config = server_lookup['code_execution']
    mcp_manager.register_server('code_execution', server_config)

    # Step 1: Write code file directly to workspace with auto-versioning
    workspace_path = Path(workspace_base)
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Get versioned filename if needed (block.filename already includes "code/" prefix)
    file_path_parts = Path(block.filename)
    just_filename = file_path_parts.name  # e.g., "analysis.py"
    code_dir = workspace_path / file_path_parts.parent  # e.g., workspace/code
    code_dir.mkdir(parents=True, exist_ok=True)

    actual_filename, version_note = _get_versioned_filename(code_dir, just_filename)
    code_file_path = code_dir / actual_filename

    if version_note:
        print(version_note)

    print(f"📝 [Write File] {code_file_path}")
    code_file_path.write_text(block.code)

    # Step 2: Execute the exact file that was written (including auto-versioned name)
    exec_file_path = (
        str(file_path_parts.parent / actual_filename)
        if str(file_path_parts.parent) not in {"", "."}
        else actual_filename
    )
    exec_args = {
        "file_path": exec_file_path,
        "workspace_base_dir": workspace_base
    }

    print(f"▶️  [Execute Code] Running code...")
    result = await mcp_manager.call_tool('code_execution', 'execute_code_file', exec_args)

    # Step 3: Format result for model consumption (simplified, less verbose)
    if isinstance(result, dict):
        # Extract key fields
        success = result.get('success', False)
        stdout = result.get('stdout', '')
        stderr = result.get('stderr', '')
        error = result.get('error', '')  # Explicit error message from validation/execution
        plots = result.get('plots', [])
        file_path = result.get('file_path', '')
        outputs = result.get('outputs', {})

        cleanup_failed = os.getenv("CODE_EXEC_CLEANUP_FAILED", "true").lower() == "true"

        # Step 3a: Keep or clean failed code files for debugging
        if not success and code_file_path.exists():
            if cleanup_failed:
                code_file_path.unlink(missing_ok=True)
                print(f"🗑️  [Cleanup] Removed crashed code file: {code_file_path.name} (execution failed)")
            else:
                print(f"💾 [Saved] Code file kept (execution failed): {code_file_path.name}")
        elif success and code_file_path.exists():
            print(f"💾 [Saved] Code file kept: {code_file_path.name}")

        if not success:
            if error:
                print(f"❌ [Code Error] {error}")
            if stderr and stderr != error:
                print(f"❌ [Code Stderr] {stderr}")

        # Build simplified result string
        parts = [f"{{'success': {success}"]

        if file_path:
            parts.append(f", 'file_path': '{file_path}'")

        # Include error message (from validation errors, execution errors, etc.)
        if error:
            parts.append(f", 'error': {repr(error)}")

        if stdout:
            parts.append(f", 'stdout': {repr(stdout)}")

        if stderr and stderr != error:  # Avoid duplicating if stderr == error
            parts.append(f", 'stderr': {repr(stderr)}")

        if plots:
            parts.append(f", 'plots': {plots}")

        # Summarize outputs instead of including raw data
        if outputs:
            outputs_summary = _summarize_outputs(outputs)
            if outputs_summary:
                parts.append(f", 'outputs_summary': '''\n{outputs_summary}\n'''")

        parts.append("}")
        return "".join(parts)

    return str(result)


def _format_success_feedback_coding(
    combined_result: str,
    agent: Agent,
    is_first_execution: bool
) -> str:
    """Format execution success feedback for coding models (terse, technical)."""
    persistence_note = ""
    if not is_first_execution:
        persistence_note = "\n💡 Variables from previous code blocks persist in this phase."

    return (
        f"✅ Code Executed Successfully:\n\n{combined_result}\n\n"
        f"Analyze output. If task complete, respond in plain text only (no code blocks). "
        f"Only write another code block if there's missing work or errors.{persistence_note}"
    )


def _format_success_feedback_general(
    combined_result: str,
    agent: Agent,
    is_first_execution: bool
) -> str:
    """Format execution success feedback for general models (narrative, descriptive)."""
    # For discussion agents (Phase 1/3), code is optional - don't encourage more
    agent_title = agent.title.lower() if agent and hasattr(agent, 'title') else ""
    is_discussion_agent = "discussion" in agent_title or "statistician" in agent_title

    if is_discussion_agent:
        # Discussion mode: code is OPTIONAL, don't push for more
        return (
            f"✅ Code Executed Successfully:\n\n{combined_result}\n\n"
            f"Code output shown above. Return to text-based discussion of the agenda items. "
            f"Only write more code if absolutely necessary for a calculation."
        )
    else:
        # Standard feedback for coding-focused agents
        return (
            f"✅ Code Executed Successfully:\n\n{combined_result}\n\n"
            f"Review the output above. If all required outputs (data files, plots, results) have been created, "
            f"provide your final summary. Only write another code block if there's missing work or errors. "
            f"Variables persist between code blocks in this phase."
        )


def _format_error_feedback_coding(
    combined_result: str,
    agent: Agent,
    is_first_execution: bool
) -> str:
    """Format error feedback for coding models (terse, technical)."""
    persistence_note = "" if is_first_execution else " Variables from previous blocks are in memory."

    return (
        f"❌ Code Execution Failed:\n\n{combined_result}\n\n"
        f"Fix the error and write corrected code.{persistence_note}"
    )


def _format_error_feedback_general(
    combined_result: str,
    agent: Agent,
    is_first_execution: bool
) -> str:
    """Format error feedback for general models (narrative, descriptive)."""
    # For discussion agents, code is optional - suggest moving on after error
    agent_title = agent.title.lower() if agent and hasattr(agent, 'title') else ""
    is_discussion_agent = "discussion" in agent_title or ("statistician" in agent_title and "coding" not in agent_title)

    if is_discussion_agent:
        # Discussion mode: code is optional, suggest continuing without it
        return (
            f"❌ Code Execution Failed:\n\n{combined_result}\n\n"
            f"Code encountered an error. Since code is optional in discussion phase, "
            f"continue with text-based analysis. Only retry code if the calculation is critical."
        )
    else:
        # Standard feedback for coding-focused agents
        return (
            f"❌ Code Execution Failed:\n\n{combined_result}\n\n"
            f"Analyze the error and write corrected code in a new block. "
            f"Variables from previous successful blocks are still in memory."
        )


def _check_misplaced_plots(workspace_base: str) -> str:
    """Check if any PNG files were saved to workspace root instead of plots/ folder.

    Args:
        workspace_base: Workspace directory for this phase

    Returns:
        Warning message if misplaced plots found, empty string otherwise
    """
    workspace_path = Path(workspace_base)

    # Look for PNG files in workspace root (not in plots/ subdirectory)
    misplaced_pngs = list(workspace_path.glob("*.png"))

    if misplaced_pngs:
        return (
            f"\n⚠️  Reminder: Use plt.savefig('plots/name.png') to save plots to plots/ folder "
            f"(found: {', '.join(f.name for f in misplaced_pngs)})"
        )

    return ""


def _format_execution_results(
    results: list[ExecutionResult],
    agent: Agent,
    agenda: str,
    is_first_execution: bool,
    workspace_base: str = ""
) -> str:
    """Format execution results into a context-aware feedback message.

    Uses model-specific formatting strategies:
    - Coding models: Terse, technical, with explicit turn boundaries
    - General models: Narrative, descriptive, encouraging

    Args:
        results: List of execution results
        agent: The agent that wrote the code
        agenda: Meeting agenda (kept for compatibility, not used - agenda is in conversation context)
        is_first_execution: Whether this is the first execution this round

    Returns:
        Formatted feedback message with model-specific prompting
    """
    if not results:
        return ""

    # Compile all result texts
    result_texts = []
    for result in results:
        if result.total_blocks > 1:
            header = f"--- Code Block {result.block_number}/{result.total_blocks} Results ---"
            result_texts.append(f"{header}\n{result.result}")
        else:
            result_texts.append(result.result)

    combined_result = "\n\n".join(result_texts)

    # Check if any execution had errors
    any_errors = any(not result.success for result in results)

    # Detect model type for formatting
    model_type = detect_model_type(agent)

    # Generate context-aware feedback based on model type
    if any_errors:
        # Error feedback
        if model_type == "coding":
            return _format_error_feedback_coding(
                combined_result, agent, is_first_execution
            )
        else:
            return _format_error_feedback_general(
                combined_result, agent, is_first_execution
            )
    else:
        # Success feedback - check for misplaced plots
        plot_warning = ""
        if workspace_base:
            plot_warning = _check_misplaced_plots(workspace_base)

        success_feedback = ""
        if model_type == "coding":
            success_feedback = _format_success_feedback_coding(
                combined_result, agent, is_first_execution
            )
        else:
            success_feedback = _format_success_feedback_general(
                combined_result, agent, is_first_execution
            )

        return success_feedback + plot_warning


def create_code_execution_node(agent: Agent = None, agent_map: dict[str, Agent] = None) -> callable:
    """Create a node that extracts and executes Python code from agent messages.

    This node looks for Python code blocks in the last message, extracts them,
    and executes them via the MCP code_execution backend. Results are added
    back to the conversation with context-aware prompting.

    Code Block Format:
        ```python
        # filename: analysis.py  (optional, auto-generated if missing)
        import numpy as np
        result = np.mean([1, 2, 3])
        print(result)
        ```

    State persistence: Variables survive across all code blocks within a phase,
    stored in workspace/.python_state.dill. No session header needed.

    Args:
        agent: The agent for individual meetings.
        agent_map: Dictionary mapping agent names to Agent objects (for team meetings).

    Returns:
        A node function that executes code from messages.
    """

    def code_execution_node(state: MeetingState | IndividualMeetingState) -> dict[str, Any]:
        """Extract and execute code from the last message."""
        from veritas.config import CODE_EXEC_WORKSPACE_DIR, DEFAULT_WORKSPACE_BASE
        import os

        messages = list(state["messages"])
        if not messages:
            return {}

        last_message = messages[-1]

        # Extract content from message
        content = last_message.content if hasattr(last_message, 'content') else ""
        if not content:
            return {}

        executions_this_round = state.get("code_executions_this_round", 0)
        if executions_this_round >= MAX_CODE_EXECUTIONS_PER_ROUND:
            # Limit reached - inject guidance message
            limit_message = HumanMessage(content=(
                f"Code execution limit reached ({MAX_CODE_EXECUTIONS_PER_ROUND} executions this round). "
                "Please provide your final interpretation and conclusions based on the results obtained so far, "
                "without writing more code. Use 'Final Answer:' to signal completion."
            ))
            return {
                "messages": [limit_message],
                "tools_just_executed": False,
            }

        # Determine which agent wrote the code
        current_agent = agent
        if agent_map and hasattr(last_message, 'name'):
            current_agent = agent_map.get(last_message.name)
        agent_name = current_agent.title if current_agent else "Agent"

        # Step 1: Extract code blocks
        blocks = _extract_code_blocks(content, agent_name)
        if not blocks:
            return {}  # No code blocks found

        # Log multi-block execution
        if len(blocks) > 1:
            print(f"📋 [Code Execution] Detected {len(blocks)} code blocks - will execute sequentially")

        # Determine workspace directory
        workspace_base = os.getenv('CODE_EXEC_WORKSPACE_DIR')
        if not workspace_base:
            workspace_base = str(CODE_EXEC_WORKSPACE_DIR if CODE_EXEC_WORKSPACE_DIR else DEFAULT_WORKSPACE_BASE)

        # Step 2: Execute all code blocks sequentially
        execution_results = []
        should_continue = True

        for block in blocks:
            if not should_continue:
                break

            print(f"🔧 [Code Block {block.block_number}/{block.total_blocks}] {agent_name} - file: {block.filename}")

            # MCP setup (only once on first block)
            if block.block_number == 1:
                print(f"🔌 [MCP Setup] Registering code_execution server")

            try:
                # Use existing event loop if available (LangGraph runs async),
                # fall back to asyncio.run() for standalone usage
                try:
                    loop = asyncio.get_running_loop()
                    import nest_asyncio
                    nest_asyncio.apply(loop)
                    result_str = loop.run_until_complete(_execute_code_block(block, workspace_base))
                except RuntimeError:
                    result_str = asyncio.run(_execute_code_block(block, workspace_base))

                # Check execution status with explicit success flag precedence.
                # Fallback to keyword heuristics only when no explicit status exists.
                result_lower = result_str.lower()
                has_explicit_failure = (
                    "'success': false" in result_lower
                    or '"success": false' in result_lower
                    or '"success":false' in result_lower
                )
                has_explicit_success = (
                    "'success': true" in result_lower
                    or '"success": true' in result_lower
                    or '"success":true' in result_lower
                )
                if has_explicit_failure:
                    is_error = True
                elif has_explicit_success:
                    is_error = False
                else:
                    is_error = any(
                        keyword in result_lower
                        for keyword in ["error", "failed", "exception", "traceback"]
                    )

                execution_results.append(ExecutionResult(
                    agent=agent_name,
                    filename=block.filename,
                    code=block.code,
                    result=result_str,
                    success=not is_error,
                    block_number=block.block_number,
                    total_blocks=block.total_blocks
                ))

                if is_error:
                    print(f"❌ [Code Block {block.block_number}] Failed - stopping execution")
                    if block.block_number < block.total_blocks:
                        remaining = block.total_blocks - block.block_number
                        error_note = (
                            f"\n\n⚠️  Note: Execution stopped at block {block.block_number} due to error.\n"
                            f"   Remaining {remaining} block(s) were NOT executed.\n"
                            f"   Please fix the error and try again."
                        )
                        # Update the result field by creating new ExecutionResult
                        prev = execution_results[-1]
                        execution_results[-1] = ExecutionResult(
                            agent=prev.agent,
                            filename=prev.filename,
                            code=prev.code,
                            result=prev.result + error_note,
                            success=prev.success,
                            block_number=prev.block_number,
                            total_blocks=prev.total_blocks
                        )
                    should_continue = False
                else:
                    print(f"✅ [Code Block {block.block_number}] Success")

            except Exception as e:
                print(f"❌ [Code Block {block.block_number}] Execution Error: {e}")
                error_result = f"Execution error: {e}"

                if block.block_number < block.total_blocks:
                    remaining = block.total_blocks - block.block_number
                    error_result += (
                        f"\n\n⚠️  Execution stopped at block {block.block_number} due to exception.\n"
                        f"   Remaining {remaining} block(s) were NOT executed."
                    )

                execution_results.append(ExecutionResult(
                    agent=agent_name,
                    filename=block.filename,
                    code=block.code,
                    result=error_result,
                    success=False,
                    block_number=block.block_number,
                    total_blocks=block.total_blocks
                ))
                should_continue = False

        # Step 3: Format results
        is_first_execution = state.get("code_executions_this_round", 0) == 0
        agenda = state.get("agenda", "")
        result_context = _format_execution_results(
            execution_results,
            current_agent,
            agenda,
            is_first_execution,
            workspace_base
        )

        # Convert ExecutionResult dataclasses to dicts for state
        execution_records = [
            {
                "agent": r.agent,
                "filename": r.filename,
                "code": r.code,
                "result": r.result,
                "success": r.success,
                "block_number": r.block_number,
                "total_blocks": r.total_blocks
            }
            for r in execution_results
        ]

        return {
            "messages": [HumanMessage(content=result_context)],
            "code_executions": execution_records,
            "last_code_caller": agent_name,
            "tools_just_executed": True,
            "code_executions_this_round": state.get("code_executions_this_round", 0) + len(execution_results),
        }

    return code_execution_node
