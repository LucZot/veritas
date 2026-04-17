"""Helper utilities for preparing data for code execution.

This module provides simple helpers that prepare data paths and metadata
for the coding agent to use. The agent will write its own data loading code
using the vision module's ResultsDatabase and ACDCLoader directly.

Keep it simple: We just provide paths and basic info, agent does the rest.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any


logger = logging.getLogger(__name__)


def prepare_workspace_metadata(
    data_paths: Optional[Dict[str, str]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Prepare workspace metadata for coding agent.

    This creates a simple JSON file with paths that the agent can use
    in its code. Agent writes its own loading code using appropriate loaders.

    Args:
        data_paths: Dictionary of data source names to paths (optional)
                   Example: {"results_database": "/path/to/results",
                            "acdc_root": "/data/ACDC",
                            "my_dataset": "/data/custom"}
        output_dir: Where to save outputs (default: /tmp/code_execution_outputs)

    Returns:
        Dictionary with workspace info

    Example:
        >>> metadata = prepare_workspace_metadata(
        ...     data_paths={
        ...         "results_database": "/path/to/results",
        ...         "custom_data": "/data/my_dataset"
        ...     }
        ... )
        >>> # Agent can access via metadata['data_sources']['results_database']
    """
    if output_dir is None:
        output_dir = "/tmp/code_execution_outputs"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metadata = {
        'workspace': {
            'output_dir': str(Path(output_dir).absolute()),
            'timestamp': None,  # Will be set by execution engine
        },
        'data_sources': {}
    }

    # Add all data paths if provided
    if data_paths:
        for source_name, source_path in data_paths.items():
            if Path(source_path).exists():
                metadata['data_sources'][source_name] = str(Path(source_path).absolute())
                logger.info(f"Data source '{source_name}': {source_path}")
            else:
                logger.warning(f"Data source '{source_name}' not found: {source_path}")

    # Save metadata to workspace
    metadata_path = Path(output_dir) / "workspace_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Workspace metadata saved to {metadata_path}")

    return metadata


def get_example_loading_code() -> str:
    """Return example code snippet for loading SAT data.

    This helps the agent understand how to load data using vision modules.

    Returns:
        String with example Python code
    """
    return """
# Example: Loading SAT segmentation data for analysis

from veritas.vision.results_db import ResultsDatabase
from veritas.vision.datasets.acdc_loader import ACDCLoader
import pandas as pd
import json

# Load workspace metadata
with open('/workspace/workspace_metadata.json', 'r') as f:
    metadata = json.load(f)

# Initialize database
db = ResultsDatabase(metadata['data_sources']['results_database'])

# Query all results
results = db.query_results()

# Convert to DataFrame for analysis
data = []
for result in results:
    record = {
        'patient_id': result['patient_id'],
        'model_name': result['model_name'],
        'mask_path': result['mask_path'],
        # Add more fields as needed
    }

    # Load metrics if available
    if 'metrics_path' in result and result['metrics_path']:
        with open(result['metrics_path'], 'r') as f:
            metrics = json.load(f)
        record['dice_score'] = metrics.get('dice_score')
        record['volume_ml'] = metrics.get('volume_ml')

    data.append(record)

df = pd.DataFrame(data)

# Add group labels if needed
acdc_root = metadata['data_sources'].get('acdc_root')
if acdc_root:
    loader = ACDCLoader(acdc_root)
    group_map = {}
    for pid in df['patient_id'].unique():
        try:
            meta = loader.get_image_metadata(pid)
            group_map[pid] = meta['group']
        except:
            group_map[pid] = 'UNKNOWN'

    df['group'] = df['patient_id'].map(group_map)

# Now ready for analysis
print(df.head())
print(df.describe())

# ==============================================================================
# V2 SIMPLIFIED API - Session Persistence & File I/O
# ==============================================================================
# **Variables AUTOMATICALLY persist across code blocks in same session!**
#
# For data flow between steps:
#   ✓ CORRECT: Just use regular Python variables (they persist!)
#   ✗ WRONG: Don't save/load between steps in same session
#
# FILE I/O (use standard Python - read anywhere, write only in workspace):
#   ✓ open("data/results.json", "w") with json.dump() - save JSON
#   ✓ df.to_csv("data/results.csv") - save DataFrames
#   ✓ plt.savefig("plots/figure.png") - save plots
#   ✓ open("/external/data.json", "r") - read from anywhere
#
# SAT DOMAIN API (auto-available, no import needed):
#   Data Access:
#   ✓ load_sat_result(db_path, patient_id) - load segmentation masks
#   ✓ list_sat_patients(db_path) - list available patients
#   ✓ get_unique_labels(mask) - get unique label IDs
#
#   Basic Metrics:
#   ✓ calculate_volume(mask, spacing, label_id=1) - calculate volume in mL
#   ✓ calculate_mass(mask, spacing, label_id=1, density_g_ml=1.05) - calculate mass in grams
#
#   Advanced Shape Metrics (domain-specific):
#   ✓ calculate_surface_area(mask, spacing, label_id=1) - surface area via marching cubes (mm²)
#   ✓ calculate_sphericity_index(mask, spacing, label_id=1) - cardiac remodeling metric (0.4-1.0)
#   ✓ calculate_wall_thickness(inner_mask, outer_mask, spacing, label_id=1) - myocardial wall stats
#
#   Convenience Functions:
#   ✓ calculate_ejection_fraction(ed_mask, es_mask, spacing, label_id=1) - direct LVEF calculation (%)
#
#   Utilities:
#   ✓ make_serializable(data) - convert numpy/pandas to JSON-safe types
#
# SECURITY (blacklist model):
#   ✗ subprocess, socket, urllib, requests - FORBIDDEN (network/shell)
#   ✗ eval(), exec(), os.system() - FORBIDDEN (code injection)
#   ✓ os.path, os.makedirs - ALLOWED (file operations)
#   ✓ open() - ALLOWED (read anywhere, write only in workspace)
# ==============================================================================

# ==============================================================================
# Example: Statistical Analysis with File I/O
# ==============================================================================
# session: my_analysis

from scipy import stats
import matplotlib.pyplot as plt
import json
import os

# Create directories (allowed in v2)
os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Step 1: Load data using SAT API
results = []
for patient_id in list_sat_patients('/path/to/results_db'):
    result = load_sat_result('/path/to/results_db', patient_id)
    volume = calculate_volume(result['mask'], label_id=1, spacing=result['spacing'])
    results.append({'patient_id': patient_id, 'volume': volume})

# Step 2: Analyze (variables persist!)
import pandas as pd
df = pd.DataFrame(results)
t_stat, p_value = stats.ttest_ind(df[df['group']=='A']['volume'],
                                   df[df['group']=='B']['volume'])

# Step 3: Save results using standard Python I/O
with open('data/results.json', 'w') as f:
    json.dump({
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }, f, indent=2)

# Step 4: Save plot
plt.figure(figsize=(8, 6))
plt.boxplot([df[df['group']=='A']['volume'], df[df['group']=='B']['volume']])
plt.savefig('plots/comparison.png', dpi=150, bbox_inches='tight')
plt.close()
"""
