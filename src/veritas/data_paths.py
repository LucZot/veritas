"""Centralized data path configuration.

All dataset paths derive from the BIO_DATA_ROOT environment variable.

  export BIO_DATA_ROOT=/path/to/your/data
"""

import os
from pathlib import Path

_bio_data_root = os.environ.get("BIO_DATA_ROOT")
if not _bio_data_root:
    raise EnvironmentError(
        "BIO_DATA_ROOT environment variable is not set. "
        "Please set it to the root directory of your datasets, e.g.:\n"
        "  export BIO_DATA_ROOT=/path/to/your/data"
    )
DATA_ROOT = Path(_bio_data_root)

ACDC_DATABASE = DATA_ROOT / "ACDC" / "database"
PDGM_ROOT = DATA_ROOT / "UCSF-PDGM"
SAT_CACHE_ACDC = DATA_ROOT / "sat_cache" / "acdc"
SAT_CACHE_PDGM = DATA_ROOT / "sat_cache" / "ucsf_pdgm"

# Placeholder used in JSON configs, resolved at runtime
DATA_ROOT_PLACEHOLDER = "__DATA_ROOT__"


def resolve_data_root(value: str) -> str:
    """Replace __DATA_ROOT__ placeholder in a string."""
    if DATA_ROOT_PLACEHOLDER in value:
        return value.replace(DATA_ROOT_PLACEHOLDER, str(DATA_ROOT))
    return value
