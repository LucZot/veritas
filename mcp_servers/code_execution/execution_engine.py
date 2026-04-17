"""Safe Python code execution engine with resource limits and validation.

This module provides subprocess-based code execution with:
- Module blacklisting (block dangerous modules like subprocess, socket)
- Function blacklisting (no eval, exec, os.system, etc.)
- Workspace-restricted file operations (read anywhere, write only in workspace)
- Resource limits (timeout, memory)
- Output capture and structured results
"""

import ast
import json
import logging
import os
import resource
import subprocess
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


logger = logging.getLogger(__name__)

# Add project src to path to import shared config
import sys
_server_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_server_root / "src"))

# Import shared configuration
try:
    from veritas.config import CODE_EXEC_WORKSPACE_DIR, DEFAULT_WORKSPACE_BASE
    # Use configured workspace directory or fall back to default
    OUTPUT_DIR = CODE_EXEC_WORKSPACE_DIR if CODE_EXEC_WORKSPACE_DIR else str(DEFAULT_WORKSPACE_BASE)
except ImportError:
    # Fallback if config module not available (backwards compatibility)
    logger.warning("Could not import veritas.config, using environment variable fallback")
    OUTPUT_DIR = os.getenv("CODE_EXEC_OUTPUT_DIR", str(_server_root / ".tmp" / "code_execution_outputs"))


# =============================================================================
# BLACKLIST-BASED SECURITY MODEL
# =============================================================================
# We use a blacklist approach: allow most Python, block dangerous operations.
# This is simpler and more flexible than the previous whitelist approach.


# Forbidden function calls (code injection, shell execution)
FORBIDDEN_FUNCTIONS = {
    'eval',
    'exec',
    'compile',
    '__import__',
    'input',
    'raw_input',
}


# Forbidden modules (network, shell access)
FORBIDDEN_MODULES = {
    'subprocess',  # Shell command execution
    'socket',      # Network sockets
    'urllib',      # URL/network access
    'requests',    # HTTP requests
    'http',        # HTTP server/client
}


# Forbidden os module functions (allow os for paths, block shell execution)
# These are checked as os.X() attribute calls
FORBIDDEN_OS_FUNCTIONS = {
    'system',      # os.system() - shell execution
    'popen',       # os.popen() - shell execution
    'spawn',       # os.spawn*() - process spawning
    'spawnl', 'spawnle', 'spawnlp', 'spawnlpe',
    'spawnv', 'spawnve', 'spawnvp', 'spawnvpe',
    'exec',        # os.exec*() - replace process
    'execl', 'execle', 'execlp', 'execlpe',
    'execv', 'execve', 'execvp', 'execvpe',
    'fork',        # os.fork() - process forking
    'forkpty',     # os.forkpty()
    'kill',        # os.kill() - send signals
    'killpg',      # os.killpg()
}


def create_restricted_open(workspace_dir: str):
    """Create a restricted open() function with read-anywhere, write-only-in-workspace.

    This wraps the built-in open() to:
    - Allow READ access to any path (needed for loading external data like Phase 2A results)
    - Restrict WRITE access to paths within the workspace directory only

    This prevents agents from writing files outside the workspace while still
    allowing them to read external data sources (segmentation results, datasets, etc.).

    Args:
        workspace_dir: Absolute path to the workspace directory

    Returns:
        A restricted open() function that validates write paths

    Example:
        >>> restricted_open = create_restricted_open("/tmp/workspace")
        >>> f = restricted_open("data.json", "w")  # OK - write in workspace
        >>> f = restricted_open("/external/data.json", "r")  # OK - read anywhere
        >>> f = restricted_open("../secret.txt", "w")  # Raises PermissionError
    """
    import builtins
    workspace_path = Path(workspace_dir).resolve()

    # Modes that involve writing
    WRITE_MODES = {'w', 'wb', 'a', 'ab', 'x', 'xb', 'w+', 'wb+', 'a+', 'ab+', 'x+', 'xb+', 'r+', 'rb+'}

    def restricted_open(file, mode='r', *args, **kwargs):
        # Convert to Path and resolve
        file_path = Path(file)

        # Handle relative paths - resolve relative to workspace
        if not file_path.is_absolute():
            resolved = (workspace_path / file_path).resolve()
        else:
            resolved = file_path.resolve()

        # Check if this is a write operation
        is_write = mode in WRITE_MODES

        # For write operations, enforce workspace restriction
        if is_write:
            try:
                resolved.relative_to(workspace_path)
            except ValueError:
                raise PermissionError(
                    f"Write access denied: '{file}' is outside the workspace directory. "
                    f"Write operations must be within the session workspace. "
                    f"Read access is allowed for external files."
                )

        # Call the real open()
        return builtins.open(resolved, mode, *args, **kwargs)

    return restricted_open


class CodeValidationError(Exception):
    """Raised when code fails safety validation."""
    pass


class CodeExecutionError(Exception):
    """Raised when code execution fails."""
    pass


def validate_imports(code: str) -> Tuple[bool, Optional[str]]:
    """Validate that no forbidden modules are imported.

    Uses a blacklist approach - most imports are allowed, only dangerous
    modules (subprocess, socket, network libraries) are blocked.

    Args:
        code: Python code to validate

    Returns:
        (valid, error_message): True if valid, False with error message if not

    Example:
        >>> validate_imports("import numpy as np")
        (True, None)
        >>> validate_imports("import subprocess")
        (False, "Forbidden module: subprocess...")
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        # Check "import X" statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split('.')[0]
                if module_name in FORBIDDEN_MODULES:
                    return False, (
                        f"Forbidden module: {module_name}. "
                        f"Security restriction - network and shell access is not allowed."
                    )

        # Check "from X import Y" statements
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split('.')[0]
                if module_name in FORBIDDEN_MODULES:
                    return False, (
                        f"Forbidden module: {module_name}. "
                        f"Security restriction - network and shell access is not allowed."
                    )

                # Check for "from os import system/popen/fork/etc."
                if module_name == 'os':
                    for alias in node.names:
                        if alias.name in FORBIDDEN_OS_FUNCTIONS:
                            return False, (
                                f"Forbidden os function: os.{alias.name}. "
                                f"Security restriction - shell execution is not allowed."
                            )

    return True, None


def validate_function_calls(code: str) -> Tuple[bool, Optional[str]]:
    """Validate that no forbidden functions are called.

    Checks for:
    - Direct forbidden functions: eval(), exec(), compile(), etc.
    - Dangerous os module functions: os.system(), os.popen(), os.fork(), etc.

    Args:
        code: Python code to validate

    Returns:
        (valid, error_message): True if valid, False with error message if not

    Example:
        >>> validate_function_calls("print('hello')")
        (True, None)
        >>> validate_function_calls("eval('1+1')")
        (False, "Forbidden function: eval")
        >>> validate_function_calls("os.system('ls')")
        (False, "Forbidden os function: system...")
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Direct function call: eval(), exec(), etc.
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_FUNCTIONS:
                    return False, (
                        f"Forbidden function: {node.func.id}(). "
                        f"Security restriction - code injection is not allowed."
                    )

            # Attribute call: os.system(), os.popen(), etc.
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

                # Check if it's a forbidden direct function
                if func_name in FORBIDDEN_FUNCTIONS:
                    return False, (
                        f"Forbidden function: {func_name}(). "
                        f"Security restriction - code injection is not allowed."
                    )

                # Check if it's a forbidden os.X() call
                if func_name in FORBIDDEN_OS_FUNCTIONS:
                    # Try to determine if it's called on 'os' module
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'os':
                        return False, (
                            f"Forbidden os function: os.{func_name}(). "
                            f"Security restriction - shell execution is not allowed. "
                            f"Use standard file I/O (open, os.path, os.makedirs) instead."
                        )
                    # Also catch os module from imports like "from os import system"
                    # This is a heuristic - the function name alone is suspicious
                    if func_name in {'system', 'popen', 'fork'}:
                        return False, (
                            f"Forbidden function: {func_name}(). "
                            f"Security restriction - shell/process execution is not allowed."
                        )

    return True, None


def validate_code(code: str) -> Tuple[bool, Optional[str]]:
    """Run all safety validations on code.

    Args:
        code: Python code to validate

    Returns:
        (valid, error_message): True if valid, False with error message if not
    """
    # Check imports
    valid, error = validate_imports(code)
    if not valid:
        return False, error

    # Check function calls
    valid, error = validate_function_calls(code)
    if not valid:
        return False, error

    return True, None


def execute_code(
    code: str,
    timeout: int = 60,
    max_memory_mb: int = 4096,
    working_dir: Optional[str] = None,
    capture_plots: bool = True,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Execute Python code with safety checks and resource limits.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 60)
        max_memory_mb: Maximum memory in MB (default: 4096)
        working_dir: Optional working directory for execution
        capture_plots: Auto-save matplotlib figures (default: True)
        session_id: Optional session ID for state persistence across executions.
                   If provided, variables will persist between calls using the same session_id.

    Returns:
        Dictionary with:
        - success: bool
        - stdout: str (captured print output)
        - stderr: str (captured errors)
        - return_value: Any (if code ends with expression)
        - plots: list of saved plot paths
        - error: Optional error message
        - session_id: str (if session_id was provided)

    Raises:
        CodeValidationError: If code fails safety checks
        CodeExecutionError: If execution fails

    Example:
        >>> result = execute_code("import numpy as np; print(np.mean([1,2,3]))")
        >>> result['stdout']
        '2.0\\n'

        >>> # With session persistence
        >>> r1 = execute_code("x = 42", session_id="my_session")
        >>> r2 = execute_code("print(x)", session_id="my_session")  # x is available
        >>> r2['stdout']
        '42\\n'
    """
    # Validate code first
    valid, error = validate_code(code)
    if not valid:
        raise CodeValidationError(error)

    # Determine working directory based on session_id
    if session_id:
        # Use session-specific workspace for state persistence
        base_dir = Path(OUTPUT_DIR)
        base_dir.mkdir(parents=True, exist_ok=True)
        working_dir = str(base_dir / f'session_{session_id}')
        os.makedirs(working_dir, exist_ok=True)
    elif working_dir is None:
        # Create temporary workspace (existing behavior)
        working_dir = tempfile.mkdtemp(prefix='code_exec_')
    else:
        # Use provided working directory
        working_dir = str(Path(working_dir).absolute())
        os.makedirs(working_dir, exist_ok=True)

    # State file for session persistence
    state_file = Path(working_dir) / '.python_state.dill'

    # Create temporary file for code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Add workspace-restricted open() for session-based execution
        if session_id:
            f.write(f"""
# Set up workspace-restricted open() - read anywhere, write only in workspace
import builtins
from pathlib import Path

_workspace_path = Path(r'{working_dir}').resolve()
_builtin_open = builtins.open

# Modes that involve writing
_WRITE_MODES = {{'w', 'wb', 'a', 'ab', 'x', 'xb', 'w+', 'wb+', 'a+', 'ab+', 'x+', 'xb+', 'r+', 'rb+'}}

def open(file, mode='r', *args, **kwargs):
    '''Workspace-restricted open() - read anywhere, write only in workspace.'''
    file_path = Path(file)
    if not file_path.is_absolute():
        resolved = (_workspace_path / file_path).resolve()
    else:
        resolved = file_path.resolve()

    # Only restrict write operations to workspace
    if mode in _WRITE_MODES:
        try:
            resolved.relative_to(_workspace_path)
        except ValueError:
            raise PermissionError(
                f"Write access denied: '{{file}}' is outside the workspace directory. "
                f"Write operations must be within the session workspace. "
                f"Read access is allowed for external files."
            )
    return _builtin_open(resolved, mode, *args, **kwargs)

# Override the built-in open in the global namespace
builtins.open = open

""")

        # Add plot capture if requested
        if capture_plots:
            plot_capture_code = """
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os

# Override plt.savefig to track saved plots
_original_savefig = plt.savefig
_saved_plots = []

def _tracked_savefig(*args, **kwargs):
    _original_savefig(*args, **kwargs)
    if args:
        _saved_plots.append(str(args[0]))

plt.savefig = _tracked_savefig

"""
            f.write(plot_capture_code)

        # Add state loading for session persistence
        if session_id:
            state_loading_code = f"""
# Load session state
_state_file = None
try:
    import dill
    from pathlib import Path
    _state_file = Path(r'{state_file}')
    if _state_file.exists():
        with open(_state_file, 'rb') as _f:
            _state = dill.load(_f)
            globals().update(_state)
except Exception as _e:
    try:
        if _state_file and _state_file.exists():
            _state_file.unlink()
    except Exception:
        pass
    print(f"Warning: Could not load session state: {{_e}}", file=__import__('sys').stderr)

"""
            f.write(state_loading_code)

        # Write user code
        f.write("\n# User code\n")
        f.write(code)

        # Add state saving for session persistence
        if session_id:
            state_saving_code = f"""

# Save session state
try:
    import dill
    from pathlib import Path
    _state_file = Path(r'{state_file}')
    _excluded = {{'dill', 'Path', 'sys', 'json', 'pickle',
                 'registry', 'bio_api', 'sat', 'open',
                 'list_dataset_patients', 'get_patient_metadata',
                 'check_sat_status', 'segment_medical_structure',
                 'segment_structures_batch', 'list_available_structures',
                 'nib', 'nibabel'}}
    _state = {{k: v for k, v in globals().items()
             if not k.startswith('_') and k not in _excluded}}
    with open(_state_file, 'wb') as _f:
        dill.dump(_state, _f)
except Exception as _e:
    print(f"Warning: Could not save session state: {{_e}}", file=__import__('sys').stderr)
"""
            f.write(state_saving_code)

        # Add plot tracking output
        if capture_plots:
            f.write("\n\nif _saved_plots:\n    _plots_str = ','.join(_saved_plots)\n    print(f'__PLOTS__:{_plots_str}')\n")

        code_file = Path(f.name)

    try:
        # Prepare environment with memory limit + new process group
        def set_limits():
            """Set resource limits and create new process group for subprocess."""
            os.setpgrp()  # New process group so we can kill children on timeout
            try:
                max_memory_bytes = max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
            except Exception as e:
                logger.warning(f"Could not set memory limit: {e}")

        # Execute code with Popen for reliable process group cleanup on timeout
        import signal
        proc = subprocess.Popen(
            ['python3', str(code_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=working_dir,
            preexec_fn=set_limits,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the entire process group (including any child processes)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait()
            raise CodeExecutionError(f"Code execution timed out after {timeout} seconds")

        plots = []

        # Extract plot paths from output
        if capture_plots and '__PLOTS__:' in stdout:
            lines = stdout.split('\n')
            for line in lines:
                if line.startswith('__PLOTS__:'):
                    plot_list = line.replace('__PLOTS__:', '').strip()
                    if plot_list:
                        plots = plot_list.split(',')
            # Remove plot marker from stdout
            stdout = '\n'.join(line for line in lines if not line.startswith('__PLOTS__:'))

        # Determine success
        success = proc.returncode == 0

        response = {
            'success': success,
            'stdout': stdout,
            'stderr': stderr,
            'plots': plots,
            'working_dir': working_dir,
            'return_code': proc.returncode,
            'error': stderr if not success else None
        }

        # Add session_id to response if provided
        if session_id:
            response['session_id'] = session_id

        return response

    except CodeExecutionError:
        raise

    except Exception as e:
        raise CodeExecutionError(f"Execution failed: {str(e)}\n{traceback.format_exc()}")

    finally:
        # Clean up code file
        try:
            code_file.unlink()
        except Exception as e:
            logger.warning(f"Could not delete temp file {code_file}: {e}")


def execute_code_file(
    file_path: str,
    timeout: int = 60,
    max_memory_mb: int = 4096,
    capture_plots: bool = True,
    data_paths: Optional[Dict[str, str]] = None,
    workspace_base_dir: Optional[str] = None,
    session_id: Optional[str] = None  # Deprecated, ignored - kept for backwards compatibility
) -> Dict[str, Any]:
    """Execute a Python file from the phase workspace.

    This function executes a Python file that was previously written to the
    workspace (e.g., by code_execution.py). The file is validated, executed
    in an isolated subprocess, and results are returned.

    State persistence works at the phase level: variables survive across all
    code executions within a single phase, stored in workspace/.python_state.dill.

    Args:
        file_path: Relative path to .py file in workspace (e.g., "code/analysis.py")
        timeout: Maximum execution time in seconds (default: 60)
        max_memory_mb: Maximum memory in MB (default: 4096)
        capture_plots: Auto-save matplotlib figures (default: True)
        data_paths: Optional dictionary of data source names to paths
                   Example: {"results_database": "/path/to/db", "my_data": "/data/custom"}
        workspace_base_dir: Base directory for workspace. If not provided,
                           uses the globally configured OUTPUT_DIR.
        session_id: Deprecated. Ignored if provided.

    Returns:
        Dictionary with:
        - success: bool
        - stdout: str (captured print output)
        - stderr: str (captured errors)
        - plots: list of saved plot paths
        - working_dir: str (workspace directory)
        - return_code: int
        - error: Optional error message
        - file_path: str (executed file path)

    Raises:
        CodeValidationError: If file not found, outside workspace, or validation fails
        CodeExecutionError: If execution fails

    Example:
        >>> result = execute_code_file(
        ...     file_path="code/analysis.py",
        ...     workspace_base_dir="/path/to/workspace"
        ... )
        >>> print(result['stdout'])
    """
    # Get workspace directory (phase-level, no session subdirectory)
    if workspace_base_dir:
        workspace_dir = Path(workspace_base_dir)
    else:
        workspace_dir = Path(OUTPUT_DIR)

    # Create workspace if it doesn't exist
    if not workspace_dir.exists():
        workspace_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created workspace: {workspace_dir}")

    # Create standard subdirectories
    (workspace_dir / 'code').mkdir(exist_ok=True)
    (workspace_dir / 'data').mkdir(exist_ok=True)
    (workspace_dir / 'plots').mkdir(exist_ok=True)

    base_dir = workspace_dir

    # Validate file path (no traversal, must exist)
    if '..' in Path(file_path).parts:
        raise CodeValidationError(f"Path traversal not allowed: {file_path}")

    full_file_path = (workspace_dir / file_path).resolve()

    try:
        full_file_path.relative_to(workspace_dir.resolve())
    except ValueError:
        raise CodeValidationError(f"File path outside workspace: {file_path}")

    # If file doesn't exist, try looking in code/ subdirectory
    original_file_path = file_path
    if not full_file_path.exists():
        # Try code/ prefix if not already present
        if not file_path.startswith('code/'):
            code_file_path = workspace_dir / f'code/{file_path}'
            if code_file_path.exists():
                full_file_path = code_file_path.resolve()
                # Update file_path for logging/metadata
                file_path = f'code/{file_path}'
    
    if not full_file_path.exists():
        if original_file_path.startswith('code/'):
            raise CodeValidationError(f"File not found: {original_file_path}")
        else:
            raise CodeValidationError(f"File not found: {original_file_path} (also checked code/{original_file_path})")

    if not full_file_path.suffix == '.py':
        raise CodeValidationError(f"File must be .py: {file_path}")

    # Read code from file
    code = full_file_path.read_text(encoding='utf-8')

    # Validate code (AST-based security checks)
    valid, error = validate_code(code)
    if not valid:
        raise CodeValidationError(f"Code validation failed: {error}")

    # Create workspace subdirectories if they don't exist
    (workspace_dir / 'code').mkdir(exist_ok=True)
    (workspace_dir / 'data').mkdir(exist_ok=True)
    (workspace_dir / 'plots').mkdir(exist_ok=True)

    # Prepare workspace metadata (paths for data loading) if needed
    if data_paths:
        from data_helpers import prepare_workspace_metadata
        metadata_path = workspace_dir / "workspace_metadata.json"
        metadata = prepare_workspace_metadata(
            data_paths=data_paths,
            output_dir=str(workspace_dir)
        )
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    # State file
    state_file = workspace_dir / '.python_state.dill'

    # Create wrapper script that loads state, runs user file, saves state
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Plot capture
        if capture_plots:
            f.write("""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Track saved plots
_original_savefig = plt.savefig
_saved_plots = []

def _tracked_savefig(*args, **kwargs):
    _original_savefig(*args, **kwargs)
    if args:
        _saved_plots.append(str(args[0]))

plt.savefig = _tracked_savefig

""")

        # State loading
        f.write(f"""
# Load session state
_state_file = None
try:
    import dill
    from pathlib import Path
    _state_file = Path(r'{state_file}')
    if _state_file.exists():
        with open(_state_file, 'rb') as _f:
            _state = dill.load(_f)
            globals().update(_state)
except Exception as _e:
    try:
        if _state_file and _state_file.exists():
            _state_file.unlink()
    except Exception:
        pass
    print(f"Warning: Could not load session state: {{_e}}", file=__import__('sys').stderr)

""")

        # Import bio_api registry for discoverable API access
        # Get the path to the bio_api package (in src/)
        repo_root = Path(__file__).parent.parent.parent
        bio_api_path = repo_root / "src"

        f.write(f"""
# Set up bio_api for data access
import sys
import builtins
from pathlib import Path

# Add src directory to path for bio_api imports
_src_dir = Path(r'{bio_api_path}')
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Import SAT API - provides pre-loaded segmentation analysis tools
from bio_api.domains.sat import SATAPI
sat = SATAPI()

# Make 'sat' importable so agents can write 'import sat' naturally
# Create a fake module that just exposes the sat API object's methods
import types
_sat_module = types.ModuleType('sat')
_sat_module.__dict__.update({{
    'list_patients': sat.list_patients,
    'load_structure_mask': sat.load_structure_mask,
    'get_patient_metadata': sat.get_patient_metadata,
    'get_observation_identifiers': sat.get_observation_identifiers,
    'list_patient_source_images': sat.list_patient_source_images,
    'calculate_volume': sat.calculate_volume,
    'calculate_mass': sat.calculate_mass,
    'calculate_surface_area': sat.calculate_surface_area,
    'calculate_sphericity_index': sat.calculate_sphericity_index,
    'calculate_wall_thickness': sat.calculate_wall_thickness,
    'calculate_ejection_fraction': sat.calculate_ejection_fraction,
    'get_unique_labels': sat.get_unique_labels,
    '__doc__': 'SAT (Segment Anything for Medical Images) API - pre-loaded for analysis'
}})
sys.modules['sat'] = _sat_module

# Also keep registry for backwards compatibility
from bio_api import registry

# Dataset discovery APIs for Phase 2A flexibility
from veritas.vision.datasets.dataset_tools import (
    list_dataset_patients as _tool_list_dataset_patients,
    get_dataset_patient_info as _tool_get_dataset_patient_info,
)

def _invoke_tool(tool_fn, **kwargs):
    if hasattr(tool_fn, "invoke"):
        return tool_fn.invoke(kwargs)
    return tool_fn(**kwargs)

def list_dataset_patients(dataset: str, group: str = None, metadata_filters: dict = None) -> dict:
    '''List patients in dataset, optionally filtered by group and/or metadata.

    Args:
        dataset: Dataset name (e.g., "acdc")
        group: Optional group filter (e.g., "DCM", "GradeIV")
        metadata_filters: Optional dict to filter by metadata fields (e.g., {{"idh_status": "wildtype"}})

    Returns:
        dict with keys: dataset, total_count, filter, metadata_filters, patients (list)
    '''
    kwargs = {{"dataset": dataset, "include_patient_ids": True}}
    if group is not None:
        kwargs["group"] = group
    if metadata_filters is not None:
        kwargs["metadata_filters"] = metadata_filters
    return _invoke_tool(_tool_list_dataset_patients, **kwargs)

def get_patient_metadata(dataset: str, patient_id: str) -> dict:
    '''Get patient metadata including identifiers.

    Args:
        dataset: Dataset name (e.g., "acdc")
        patient_id: Patient ID (e.g., "patient001")

    Returns:
        dict with keys: patient_id, group, ed_identifier, es_identifier, ed_frame, es_frame, etc.
    '''
    return _invoke_tool(_tool_get_dataset_patient_info, dataset=dataset, patient_id=patient_id)

# Pre-inject results_db_path so agent code can use it directly without hardcoding paths
_results_db_env = r'{os.environ.get("PHASE2B_RESULTS_DB", "")}'
if _results_db_env:
    results_db_path = _results_db_env

# Set up workspace-restricted open() - read anywhere, write only in workspace root
# Use base_dir (workspace root) not workspace_dir (session subdir) to allow writing to workspace/data/
_workspace_path = Path(r'{base_dir}').resolve()
_builtin_open = builtins.open

# Modes that involve writing
_WRITE_MODES = {{'w', 'wb', 'a', 'ab', 'x', 'xb', 'w+', 'wb+', 'a+', 'ab+', 'x+', 'xb+', 'r+', 'rb+'}}

def open(file, mode='r', *args, **kwargs):
    '''Workspace-restricted open() - read anywhere, write only in workspace.'''
    file_path = Path(file)
    if not file_path.is_absolute():
        resolved = (_workspace_path / file_path).resolve()
    else:
        resolved = file_path.resolve()

    # Only restrict write operations to workspace
    if mode in _WRITE_MODES:
        try:
            resolved.relative_to(_workspace_path)
        except ValueError:
            raise PermissionError(
                f"Write access denied: '{{file}}' is outside the workspace directory. "
                f"Write operations must be within the session workspace. "
                f"Read access is allowed for external files."
            )
    return _builtin_open(resolved, mode, *args, **kwargs)

# Override the built-in open in the global namespace
builtins.open = open

""")

        # Execute user file (add workspace/code to PYTHONPATH for imports)
        f.write(f"""
# Add workspace code directory to Python path for imports
import sys
_code_dir = r'{workspace_dir / "code"}'
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

# Execute user file
with open(r'{full_file_path}', 'r', encoding='utf-8') as _user_file:
    _user_code = _user_file.read()
    exec(_user_code, globals())

""")

        # State saving
        f.write(f"""
# Save session state
try:
    import dill
    from pathlib import Path
    _state_file = Path(r'{state_file}')
    # Exclude infrastructure and bio_api from state
    _excluded = {{'dill', 'Path', 'sys', 'json', 'pickle',
                 'registry', 'bio_api', 'sat', 'open',
                 'list_dataset_patients', 'get_patient_metadata',
                 'check_sat_status', 'segment_medical_structure',
                 'segment_structures_batch', 'list_available_structures',
                 'nib', 'nibabel'}}
    _state = {{k: v for k, v in globals().items()
             if not k.startswith('_') and k not in _excluded}}
    with open(_state_file, 'wb') as _f:
        dill.dump(_state, _f)
except Exception as _e:
    print(f"Warning: Could not save session state: {{_e}}", file=__import__('sys').stderr)

""")

        # Plot tracking
        if capture_plots:
            f.write("\nif _saved_plots:\n    _plots_str = ','.join(_saved_plots)\n    print(f'__PLOTS__:{_plots_str}')\n")

        wrapper_file = Path(f.name)

    try:
        # Prepare environment with memory limit + new process group
        def set_limits():
            """Set resource limits and create new process group for subprocess."""
            os.setpgrp()  # New process group so we can kill children on timeout
            try:
                max_memory_bytes = max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
            except Exception as e:
                logger.warning(f"Could not set memory limit: {e}")

        # Execute wrapper script with Popen for reliable process group cleanup
        import signal
        proc = subprocess.Popen(
            ['python3', str(wrapper_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(workspace_dir),
            preexec_fn=set_limits,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait()
            raise CodeExecutionError(f"Code execution timed out after {timeout} seconds")

        plots = []

        # Extract plot paths from output
        if capture_plots and '__PLOTS__:' in stdout:
            lines = stdout.split('\n')
            for line in lines:
                if line.startswith('__PLOTS__:'):
                    plot_list = line.replace('__PLOTS__:', '').strip()
                    if plot_list:
                        plots = plot_list.split(',')
            # Remove plot marker from stdout
            stdout = '\n'.join(line for line in lines if not line.startswith('__PLOTS__:'))

        success = proc.returncode == 0

        # Move any PNG files from session root to plots/ folder (cleanup misplaced plots)
        if capture_plots:
            for png_file in workspace_dir.glob('*.png'):
                try:
                    import shutil
                    plots_dir = workspace_dir / 'plots'
                    plots_dir.mkdir(exist_ok=True)
                    dest = plots_dir / png_file.name
                    shutil.move(str(png_file), str(dest))
                    logger.debug(f"Moved misplaced plot: {png_file.name} → plots/")
                except Exception as e:
                    logger.warning(f"Could not move PNG file {png_file.name}: {e}")

        # Auto-read outputs.json if it exists (structured results)
        outputs = None
        outputs_file = workspace_dir / 'data' / 'outputs.json'
        if outputs_file.exists():
            try:
                with open(outputs_file, 'r') as f:
                    outputs = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read outputs.json: {e}")

        # Build return dictionary
        return_dict = {
            'success': success,
            'stdout': stdout,
            'stderr': stderr,
            'plots': plots,
            'working_dir': str(workspace_dir),
            'file_path': file_path,
            'execution_info': {
                'return_code': proc.returncode,
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': None  # Could add timing if needed
            }
        }

        # Add outputs if available
        if outputs:
            return_dict['outputs'] = outputs

        # Add error field for failed executions
        if not success:
            return_dict['error'] = stderr

        return return_dict

    except CodeExecutionError:
        raise
    except Exception as e:
        raise CodeExecutionError(f"Execution failed: {str(e)}\n{traceback.format_exc()}")
    finally:
        try:
            wrapper_file.unlink()
        except Exception as e:
            logger.warning(f"Could not delete wrapper file: {e}")





def cleanup_old_sessions(max_age_hours: int = 1) -> Dict[str, Any]:
    """Clean up session workspaces older than specified age.

    Args:
        max_age_hours: Maximum age in hours (default: 1)

    Returns:
        Dictionary with:
        - success: bool
        - cleaned_sessions: list of cleaned session IDs
        - failed_sessions: list of session IDs that failed to clean
        - total_removed_files: int

    Example:
        >>> result = cleanup_old_sessions(max_age_hours=2)
        >>> result['cleaned_sessions']
        ['session_abc123', 'session_xyz789']
    """
    import shutil
    from datetime import datetime, timedelta

    base_dir = Path(OUTPUT_DIR)

    if not base_dir.exists():
        return {
            'success': True,
            'cleaned_sessions': [],
            'failed_sessions': [],
            'total_removed_files': 0
        }

    max_age = timedelta(hours=max_age_hours)
    now = datetime.now()
    cleaned = []
    failed = []
    total_files = 0

    try:
        for session_dir in base_dir.glob('session_*'):
            if not session_dir.is_dir():
                continue

            # Check modification time
            mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
            age = now - mtime

            if age > max_age:
                session_id = session_dir.name
                try:
                    # Count files
                    file_count = sum(1 for _ in session_dir.rglob('*') if _.is_file())
                    total_files += file_count

                    # Remove directory
                    shutil.rmtree(session_dir)
                    cleaned.append(session_id)
                    logger.info(f"Cleaned up old session: {session_id} ({file_count} files)")

                except Exception as e:
                    failed.append(session_id)
                    logger.error(f"Failed to cleanup session {session_id}: {e}")

        return {
            'success': len(failed) == 0,
            'cleaned_sessions': cleaned,
            'failed_sessions': failed,
            'total_removed_files': total_files
        }

    except Exception as e:
        logger.error(f"Error during session cleanup: {e}")
        return {
            'success': False,
            'cleaned_sessions': cleaned,
            'failed_sessions': failed,
            'total_removed_files': total_files
        }
