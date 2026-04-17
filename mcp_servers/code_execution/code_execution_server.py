#!/usr/bin/env python3
"""
Code Execution MCP Server - Safe Python execution for hypothesis testing.

This server provides tools for AI agents to write and execute Python code
for statistical analysis and hypothesis testing on medical imaging data.

Usage:
    python code_execution_server.py

Environment variables:
    CODE_EXEC_OUTPUT_DIR: Output directory (default: /tmp/code_execution_outputs)
    CODE_EXEC_TIMEOUT: Execution timeout in seconds (default: 60)
    CODE_EXEC_MAX_MEMORY_MB: Memory limit in MB (default: 4096)
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

# MCP imports
from mcp.server import Server
from mcp.types import Tool, TextContent
import anyio

# No longer using Pydantic - simple validation functions instead

# Local imports
from execution_engine import (
    execute_code,
    execute_code_file,
    validate_code,
    CodeValidationError,
    CodeExecutionError
)
from data_helpers import (
    prepare_workspace_metadata
)

# Configure logging to file only (not stderr) to avoid corrupting MCP JSON-RPC protocol
log_file = Path(__file__).parent.parent.parent / "tmp" / "code_execution_mcp_server.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Code Execution MCP Server starting - logging to {log_file}")


# ============================================================================
# Basic Input Validation (minimal Pydantic usage)
# ============================================================================

def validate_session_id(session_id: str) -> bool:
    """Validate session ID format."""
    return bool(session_id and len(session_id) < 100 and '..' not in session_id)


def validate_file_path(file_path: str) -> tuple[bool, str]:
    """Validate file path for security."""
    if '..' in file_path:
        return False, "Path traversal not allowed"
    if not file_path.endswith('.py'):
        return False, "File must be a .py file"
    return True, ""


# Create MCP server
server = Server("code-execution")

# Configuration from environment
# Default to tmp directory in project root
_server_root = Path(__file__).parent.parent.parent

# Add project src to path to import shared config
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

DEFAULT_TIMEOUT = int(os.getenv("CODE_EXEC_TIMEOUT", "60"))
MAX_MEMORY_MB = int(os.getenv("CODE_EXEC_MAX_MEMORY_MB", "4096"))

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Concurrency limiter - only one code execution at a time for safety
_execution_limiter = anyio.CapacityLimiter(1)

logger.info(f"Configuration: output_dir={OUTPUT_DIR}, timeout={DEFAULT_TIMEOUT}s, max_memory={MAX_MEMORY_MB}MB")


# ============================================================================
# Valid tool names for validation
VALID_TOOLS = {
    "execute_code_file"
}


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available code execution tools."""
    return [
        Tool(
            name="execute_code_file",
            description="""Execute a Python file from the phase workspace.

Variables persist across all code executions within a phase (stored in workspace/.python_state.dill).

FILE I/O: Standard Python file operations are allowed within the workspace:
- open("data/results.json", "w") - write files
- json.dump(data, f) - save JSON
- df.to_csv("data/results.csv") - save DataFrames
- plt.savefig("plots/figure.png") - save plots

SAT DOMAIN API (auto-available, no import needed):
  Data Access:
  - load_sat_result(results_db_path, patient_id) - load segmentation masks
  - list_sat_patients(results_db_path) - list available patients
  - get_unique_labels(mask) - get labels in a mask

  Basic Metrics:
  - calculate_volume(mask, spacing, label_id=1) - calculate volume in mL
  - calculate_mass(mask, spacing, label_id=1, density_g_ml=1.05) - calculate mass in grams

  Advanced Shape Metrics (domain-specific):
  - calculate_surface_area(mask, spacing, label_id=1) - surface area via marching cubes (mm²)
  - calculate_sphericity_index(mask, spacing, label_id=1) - cardiac remodeling metric (0.4-1.0)
  - calculate_wall_thickness(inner_mask, outer_mask, spacing, label_id=1) - myocardial wall stats (mm)

  Convenience Functions:
  - calculate_ejection_fraction(ed_mask, es_mask, spacing, label_id=1) - direct LVEF calculation (%)

  Utilities:
  - make_serializable(data) - convert numpy/pandas to JSON-safe types

PACKAGES: numpy, pandas, scipy, matplotlib, seaborn, sklearn, statsmodels, nibabel, etc.
BLOCKED: subprocess, socket, urllib, requests (no network/shell access)
BLOCKED: os.system, os.popen, os.fork, eval, exec (no code injection)
FILE ACCESS: Restricted to workspace only""",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to Python file in workspace (e.g., 'code/analysis.py'). Files created by code_execution are in code/ subdirectory."
                    },
                    "timeout": {
                        "type": "integer",
                        "default": DEFAULT_TIMEOUT,
                        "description": "Execution timeout in seconds"
                    },
                    "workspace_base_dir": {
                        "type": "string",
                        "description": "Base directory for workspace. If not provided, uses globally configured OUTPUT_DIR."
                    }
                },
                "required": ["file_path"]
            }
        ),


    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls with validation."""

    try:
        # Validate tool existence first
        if name not in VALID_TOOLS:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": f"Unknown tool: {name}",
                    "suggestion": "Use list_capabilities to see available tools and helpers",
                    "available_tools": sorted(list(VALID_TOOLS)),
                    "tool_not_found": True
                })
            )]

        # Route to handlers
        if name == "execute_code_file":
            return await _execute_code_file(arguments)



        else:
            # Should never reach here due to validation above
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": f"Tool handler not implemented: {name}"
                })
            )]

    except Exception as e:
        logger.error(f"Tool {name} failed: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "tool_name": name
            })
        )]


async def _execute_python_code(arguments: dict) -> List[TextContent]:
    """Execute Python code with safety checks."""

    async with _execution_limiter:  # Only one execution at a time
        code = arguments["code"]
        description = arguments["description"]
        timeout = arguments.get("timeout", DEFAULT_TIMEOUT)
        session_id = arguments.get("session_id")
        results_database = arguments.get("results_database")
        acdc_root = arguments.get("acdc_root")

        logger.info(f"Executing code: {description}")
        logger.info(f"Code length: {len(code)} characters")
        if session_id:
            logger.info(f"Using session: {session_id}")

        # Validate code first
        valid, error = validate_code(code)
        if not valid:
            logger.error(f"Code validation failed: {error}")
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": f"Code validation failed: {error}",
                    "validation_error": True
                })
            )]

        # Determine workspace directory
        if session_id:
            # Use session-based workspace (handled by execute_code)
            workspace_dir = None  # Let execute_code handle it
        else:
            # Create timestamped workspace (existing behavior)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            workspace_dir = Path(OUTPUT_DIR) / f"exec_{timestamp}"
            workspace_dir.mkdir(parents=True, exist_ok=True)

        # Prepare workspace metadata (paths for data loading) if workspace_dir known
        if workspace_dir:
            workspace_metadata = prepare_workspace_metadata(
                results_database=results_database,
                acdc_root=acdc_root,
                output_dir=str(workspace_dir)
            )

            # Save metadata to workspace for code to access
            metadata_path = workspace_dir / "workspace_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(workspace_metadata, f, indent=2)
        else:
            metadata_path = None

        try:
            # Execute code in thread pool (blocking operation)
            result = await anyio.to_thread.run_sync(
                execute_code,
                code,
                timeout,
                MAX_MEMORY_MB,
                str(workspace_dir) if workspace_dir else None,
                True,  # capture_plots
                session_id  # NEW: pass session_id
            )

            logger.info(f"Code execution completed: success={result['success']}")

            # Build response
            response = {
                "success": result["success"],
                "description": description,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "plots": result["plots"],
                "working_dir": result["working_dir"],
                "execution_info": {
                    "timeout": timeout,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "return_code": result["return_code"]
                }
            }

            # Add session_id if present
            if session_id:
                response["session_id"] = session_id

            # Add metadata path if exists
            if metadata_path:
                response["workspace_metadata"] = str(metadata_path)

            if not result["success"]:
                response["error"] = result["error"]

            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]

        except CodeExecutionError as e:
            logger.error(f"Code execution error: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "execution_error": True
                })
            )]





async def _execute_code_file(arguments: dict) -> List[TextContent]:
    """Execute Python file from workspace."""

    # Extract arguments
    file_path = arguments.get("file_path")
    timeout = arguments.get("timeout", DEFAULT_TIMEOUT)
    max_memory_mb = arguments.get("max_memory_mb", MAX_MEMORY_MB)
    capture_plots = arguments.get("capture_plots", True)
    workspace_base_dir = arguments.get("workspace_base_dir")
    data_paths = arguments.get("data_paths")

    # Basic validation
    if not file_path:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": "Missing required field: file_path"
            })
        )]

    # Validate file path
    valid, error_msg = validate_file_path(file_path)
    if not valid:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": error_msg
            })
        )]

    async with _execution_limiter:  # Only one execution at a time

        logger.info(f"Executing file: {file_path} (workspace_base: {workspace_base_dir or 'default'})")

        # Execute file
        try:
            result = await anyio.to_thread.run_sync(
                execute_code_file,
                file_path,
                timeout,
                max_memory_mb,
                capture_plots,
                data_paths,
                workspace_base_dir
            )

            logger.info(f"File execution completed: success={result['success']}")

            # Build response
            response = {
                "success": result["success"],
                "file_path": file_path,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "plots": result["plots"],
                "working_dir": result["working_dir"],
                "execution_info": result["execution_info"],
            }

            # Add outputs if available (from outputs.json)
            if "outputs" in result:
                response["outputs"] = result["outputs"]

            # Add error if failed
            if not result["success"]:
                response["error"] = result.get("error", "Unknown error")

            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]

        except CodeValidationError as e:
            logger.error(f"Code validation error: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "validation_error": True
                })
            )]

        except CodeExecutionError as e:
            logger.error(f"Code execution error: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "execution_error": True
                })
            )]


async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server

    logger.info("Starting MCP server on stdio")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import sys
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
