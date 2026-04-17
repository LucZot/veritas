"""LangChain tool definitions for VERITAS.

This module defines tools that agents can use during meetings,
formatted for LangChain's tool calling interface.
"""

from langchain_core.tools import tool
from pathlib import Path
import os
from typing import Optional, List
import re


# ============================================================================
# File Manipulation Tools for Code Execution
# ============================================================================

# Import shared configuration
try:
    from veritas.config import CODE_EXEC_WORKSPACE_DIR, DEFAULT_WORKSPACE_BASE
    # Use configured workspace directory or fall back to default
    OUTPUT_DIR = CODE_EXEC_WORKSPACE_DIR if CODE_EXEC_WORKSPACE_DIR else str(DEFAULT_WORKSPACE_BASE)
except ImportError:
    # Fallback if config module not available (backwards compatibility)
    _project_root = Path(__file__).parent.parent.parent
    OUTPUT_DIR = os.getenv("CODE_EXEC_OUTPUT_DIR", str(_project_root / ".tmp" / "code_execution_outputs"))


def _get_session_workspace(session_id: str, workspace_base_dir: Optional[str] = None) -> Path:
    """Get workspace directory for a session.

    Args:
        session_id: Session identifier
        workspace_base_dir: Optional base directory override

    Returns:
        Path to session workspace
    """
    if workspace_base_dir:
        base_dir = Path(workspace_base_dir)
    else:
        base_dir = Path(OUTPUT_DIR)
    return base_dir / f'session_{session_id}'


def _validate_file_path(file_path: str, workspace_dir: Path) -> Path:
    """Validate and resolve file path within workspace.

    Args:
        file_path: Relative file path
        workspace_dir: Workspace root directory

    Returns:
        Resolved absolute path

    Raises:
        ValueError: If path is invalid or outside workspace
    """
    # Remove any directory traversal attempts
    if '..' in Path(file_path).parts:
        raise ValueError(f"Path traversal not allowed: {file_path}")

    # Resolve relative to workspace
    full_path = (workspace_dir / file_path).resolve()

    # Ensure path is within workspace
    try:
        full_path.relative_to(workspace_dir.resolve())
    except ValueError:
        raise ValueError(f"Path outside workspace: {file_path}")

    return full_path



@tool
def write_file(session_id: str, file_path: str, content: str, subdirectory: str = "code", workspace_base_dir: Optional[str] = None) -> str:
    """Write Python code to workspace. Code is validated before execution.

    SECURITY (blacklist model):
    ✗ subprocess, socket, urllib, requests - FORBIDDEN (network/shell)
    ✗ eval(), exec(), os.system() - FORBIDDEN (code injection)
    ✓ open() - ALLOWED (read anywhere, write only in workspace)
    ✓ os.path, os.makedirs - ALLOWED (file operations)

    FILE I/O (use standard Python):
    ✓ open("data/results.json", "w") with json.dump() - save JSON
    ✓ df.to_csv("data/results.csv") - save DataFrames
    ✓ plt.savefig("plots/figure.png") - save plots

    SAT DOMAIN API (auto-available, no import needed):
    Data Access:
    ✓ load_sat_result(db_path, patient_id) - load segmentation masks
    ✓ list_sat_patients(db_path) - list available patients

    Basic Metrics:
    ✓ calculate_volume(mask, spacing, label_id=1) - volume in mL
    ✓ calculate_mass(mask, spacing, label_id=1, density_g_ml=1.05) - mass in grams

    Advanced Metrics (domain-specific):
    ✓ calculate_surface_area(mask, spacing, label_id=1) - surface area (mm²)
    ✓ calculate_sphericity_index(mask, spacing, label_id=1) - shape remodeling metric
    ✓ calculate_wall_thickness(inner_mask, outer_mask, spacing, label_id=1) - wall stats
    ✓ calculate_ejection_fraction(ed_mask, es_mask, spacing, label_id=1) - LVEF (%)

    Example:
        content = '''
        import numpy as np
        import json
        from scipy import stats

        t_stat, p_val = stats.ttest_ind(group1, group2)
        with open("data/results.json", "w") as f:
            json.dump({"t_stat": float(t_stat), "p_value": float(p_val)}, f)
        '''

    Style: Use double quotes "text" not 'text' (avoids issues with "Cohen's d").

    Args:
        session_id: Session ID for workspace isolation
        file_path: Relative file path (e.g., "analysis.py", "utils/helpers.py")
        content: File content to write
        subdirectory: Workspace subdirectory (default: "code", options: "code", "data", "plots")
        workspace_base_dir: Optional base directory for workspace

    Returns:
        Success message with absolute file path
    """
    # DEAD SIMPLE APPROACH: Fix the most common issue only
    # If content looks like JSON-escaped (single line with many \n), decode it
    if ('\\n' in content and 
        len(content.splitlines()) <= 3 and 
        content.count('\\n') > 5):
        try:
            content = content.encode('utf-8').decode('unicode_escape')
        except (UnicodeDecodeError, ValueError):
            pass  # If decoding fails, use original content
    
    # File size limits
    MAX_FILE_SIZE = {
        'code': 100 * 1024,           # 100 KB
        'data': 10 * 1024 * 1024,     # 10 MB
        'plots': 5 * 1024 * 1024      # 5 MB
    }

    # Check file size
    content_bytes = len(content.encode('utf-8'))
    max_size = MAX_FILE_SIZE.get(subdirectory, 10 * 1024 * 1024)

    if content_bytes > max_size:
        raise ValueError(
            f"File too large: {content_bytes} bytes (max: {max_size} bytes for '{subdirectory}' directory)"
        )

    workspace_dir = _get_session_workspace(session_id, workspace_base_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectory
    subdir_path = workspace_dir / subdirectory
    subdir_path.mkdir(parents=True, exist_ok=True)

    # Handle case where file_path already includes subdirectory prefix
    if file_path.startswith(f"{subdirectory}/"):
        file_path = file_path[len(subdirectory) + 1:]

    # Validate and resolve path
    full_path = _validate_file_path(f"{subdirectory}/{file_path}", workspace_dir)

    # Create parent directories if needed
    full_path.parent.mkdir(parents=True, exist_ok=True)

    # For Python files, try to fix common LLM-generated issues before validation
    if file_path.endswith('.py'):
        original_content = content
        content = _fix_content_with_validation(content)

        # Validate after fixes
        try:
            import ast
            ast.parse(content)
        except SyntaxError as e:
            # Save raw content for debugging
            debug_path = full_path.parent / f".debug_{file_path}.txt"
            debug_path.write_text(f"=== ORIGINAL ===\n{original_content}\n\n=== AFTER FIXES ===\n{content}", encoding='utf-8')
            raise ValueError(f"Generated Python code has syntax error: {e}. Debug saved to: {debug_path}")

    # Write file with clean content
    full_path.write_text(content, encoding='utf-8')

    # Verify file was actually written
    if not full_path.exists():
        raise RuntimeError(f"File write appeared to succeed but file does not exist: {full_path}")

    return f"File written: {full_path} ({content_bytes} bytes)"


def _is_valid_python(content: str) -> bool:
    """Check if content is valid Python syntax."""
    try:
        import ast
        ast.parse(content)
        return True
    except SyntaxError:
        return False


def _fix_content_with_validation(content: str) -> str:
    """Bulletproof content fixing based on industry best practices."""

    # Step 1: If content is already valid, return as-is (most common case)
    if _is_valid_python(content):
        return content

    # Step 2: Apply fixes in order of likelihood (based on research)
    fixes_to_try = [
        ("JSON-escaped content", _fix_json_escaping),
        ("Markdown artifacts", _fix_markdown_artifacts),
        ("Broken string literals", _fix_broken_string_literals),
        ("Auto-format with autopep8", _autoformat_python),  # Handles whitespace/indentation
    ]

    for fix_name, fix_func in fixes_to_try:
        try:
            fixed_content = fix_func(content)
            if fixed_content != content and _is_valid_python(fixed_content):
                return fixed_content
        except Exception:
            continue

    # Step 3: If all targeted fixes fail, return original with validation info
    return content


def _looks_like_json_escaped(content: str) -> bool:
    """Detect JSON-escaped content - the most common issue."""
    has_literal_backslash_n = '\\n' in content
    actual_line_count = len(content.splitlines())
    literal_newline_count = content.count('\\n')
    
    return (
        has_literal_backslash_n and              # Has literal \n
        actual_line_count < 5 and                # Very few actual lines
        literal_newline_count > 3                # Many escape sequences
    )


def _fix_json_escaping(content: str) -> str:
    """Fix JSON-escaped content (most common issue)."""
    if _looks_like_json_escaped(content):
        return content.encode('utf-8').decode('unicode_escape')
    return content


def _fix_markdown_artifacts(content: str) -> str:
    """Remove common markdown artifacts from LLM responses."""
    # Remove ```python and ``` blocks
    import re
    content = re.sub(r'^```python\s*\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'\n```\s*$', '', content, flags=re.MULTILINE) 
    content = re.sub(r'^```\s*$', '', content, flags=re.MULTILINE)
    return content.strip()


def _fix_broken_string_literals(content: str) -> str:
    """Fix string literals that have literal newlines instead of \\n."""
    import re

    # Fix single-quoted strings with literal newlines
    content = re.sub(r"'([^']*)\n([^']*)'", r"'\1\\n\2'", content)

    # Fix double-quoted strings with literal newlines
    content = re.sub(r'"([^"]*)\n([^"]*)"', r'"\1\\n\2"', content)

    return content


def _autoformat_python(content: str) -> str:
    """Try to auto-format Python code using autopep8 (if available).

    This is a last-resort fix for badly formatted code from LLMs.
    Only fixes basic formatting issues (indentation, whitespace).
    """
    try:
        import autopep8
        # Only apply safe, non-aggressive fixes
        return autopep8.fix_code(content, options={'aggressive': 0})
    except ImportError:
        # autopep8 not available, return unchanged
        return content
    except Exception:
        # Formatting failed, return unchanged
        return content


@tool
def read_file(session_id: str, file_path: str, subdirectory: str = "code", workspace_base_dir: Optional[str] = None) -> str:
    """Read content from a file in the session workspace.

    Args:
        session_id: Session ID for workspace
        file_path: Relative file path (e.g., "analysis.py")
        subdirectory: Workspace subdirectory (default: "code", options: "code", "data", "plots")
        workspace_base_dir: Optional base directory for workspace (for integration with meeting outputs)

    Returns:
        File content as string

    Raises:
        ValueError: If path is invalid or file doesn't exist

    Example:
        >>> content = read_file(
        ...     session_id="analysis_001",
        ...     file_path="stats_utils.py"
        ... )
    """
    workspace_dir = _get_session_workspace(session_id, workspace_base_dir)

    if not workspace_dir.exists():
        raise ValueError(f"Session workspace not found: {session_id}")

    # Validate and resolve path
    full_path = _validate_file_path(f"{subdirectory}/{file_path}", workspace_dir)

    if not full_path.exists():
        raise ValueError(f"File not found: {file_path} in {subdirectory}/")

    content = full_path.read_text(encoding='utf-8')
    return content


@tool
def list_files(session_id: str, subdirectory: Optional[str] = None, pattern: str = "*", workspace_base_dir: Optional[str] = None) -> str:
    """List files in the session workspace.

    Args:
        session_id: Session ID for workspace
        subdirectory: Optional subdirectory to list (e.g., "code", "data", "plots")
        pattern: Glob pattern for filtering (default: "*", examples: "*.py", "**/*.json")
        workspace_base_dir: Optional base directory for workspace (for integration with meeting outputs)

    Returns:
        Formatted list of files with sizes

    Example:
        >>> list_files(session_id="analysis_001", subdirectory="code", pattern="*.py")
        "Files in session_analysis_001/code:
         - analysis.py (1234 bytes)
         - utils/helpers.py (567 bytes)"
    """
    workspace_dir = _get_session_workspace(session_id, workspace_base_dir)

    if not workspace_dir.exists():
        return f"Session workspace not found: {session_id}"

    # Determine search directory
    if subdirectory:
        search_dir = workspace_dir / subdirectory
        if not search_dir.exists():
            return f"Subdirectory not found: {subdirectory}"
    else:
        search_dir = workspace_dir

    # Find matching files
    files = []
    for file_path in search_dir.rglob(pattern):
        if file_path.is_file():
            relative_path = file_path.relative_to(workspace_dir)
            size = file_path.stat().st_size
            files.append(f"  - {relative_path} ({size} bytes)")

    if not files:
        return f"No files found matching '{pattern}' in {search_dir.relative_to(workspace_dir.parent)}"

    header = f"Files in {workspace_dir.name}/{subdirectory or ''}:"
    return header + "\n" + "\n".join(sorted(files))


