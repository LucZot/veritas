#!/bin/bash
# SAT Setup Script
# Idempotent: clones SAT only if missing, downloads each checkpoint only if missing.

set -e

SAT_DIR="${SAT_REPO_PATH:-$HOME/SAT}"
CHECKPOINT_DIR="${SAT_CHECKPOINT_DIR:-$SAT_DIR/checkpoints}"
HF_REPO="zzh99/SAT"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================================"
echo "SAT (Segment Anything for Medical Imaging) Setup"
echo "================================================================================"
echo ""

# --- Conda env ----------------------------------------------------------------

echo -e "${BLUE}[1/5] Conda environment${NC}"

if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}Warning: conda not found. Skipping env creation.${NC}"
    echo "  You will need a Python 3.11 env with huggingface-hub and mcp installed."
    SAT_PYTHON_DETECTED=""
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | grep -qE "^sat\s"; then
        echo -e "${GREEN}✓${NC} conda env 'sat' exists"
    else
        echo "Creating conda env 'sat' (Python 3.11)..."
        conda create -n sat python=3.11 -y
    fi
    conda activate sat
    SAT_PYTHON_DETECTED="$(which python)"

    for pkg in huggingface_hub mcp; do
        if ! python -c "import $pkg" &> /dev/null; then
            echo "Installing $pkg..."
            pip install -q "${pkg//_/-}>=0.19.0" || pip install -q "$pkg"
        fi
    done
    echo -e "${GREEN}✓${NC} env ready: $SAT_PYTHON_DETECTED"
fi
echo ""

# --- Repo clone ---------------------------------------------------------------

echo -e "${BLUE}[2/5] SAT repository${NC}"
if [ -d "$SAT_DIR/.git" ]; then
    echo -e "${GREEN}✓${NC} repo present at $SAT_DIR (skipping clone)"
else
    echo "Cloning SAT into $SAT_DIR..."
    git clone https://github.com/zhaoziheng/SAT.git "$SAT_DIR"
fi
echo ""

# --- SAT dependencies ---------------------------------------------------------

echo -e "${BLUE}[3/5] SAT dependencies${NC}"
if [ -n "$SAT_PYTHON_DETECTED" ]; then
    # Core requirements
    if [ -f "$SAT_DIR/requirements.txt" ]; then
        echo "Installing SAT requirements..."
        pip install -q -r "$SAT_DIR/requirements.txt"
        echo -e "${GREEN}✓${NC} requirements.txt installed"
    else
        echo -e "${YELLOW}Warning: $SAT_DIR/requirements.txt not found — skipping${NC}"
    fi

    # Custom U-Net architecture (required by SAT)
    DYN_NET="$SAT_DIR/model/dynamic-network-architectures-main"
    if [ -d "$DYN_NET" ]; then
        echo "Installing dynamic-network-architectures (editable)..."
        pip install -q -e "$DYN_NET"
        echo -e "${GREEN}✓${NC} dynamic-network-architectures installed"
    else
        echo -e "${YELLOW}Warning: $DYN_NET not found — skipping (may need manual install)${NC}"
    fi
else
    echo -e "${YELLOW}Skipping (conda not found)${NC}"
fi
echo ""

# --- Checkpoints --------------------------------------------------------------

echo -e "${BLUE}[4/5] Model checkpoints${NC}"
mkdir -p "$CHECKPOINT_DIR"

CHECKPOINT_DIR="$CHECKPOINT_DIR" HF_REPO="$HF_REPO" python3 << 'PYTHON_SCRIPT'
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install 'huggingface-hub>=0.19.0'")
    sys.exit(1)

checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"])
hf_repo = os.environ["HF_REPO"]

files = [
    ("Nano/nano.pth",              "SAT-Nano model       (~1.5 GB)"),
    ("Nano/nano_text_encoder.pth", "SAT-Nano text enc.   (~440 MB)"),
    ("Pro/SAT_Pro.pth",            "SAT-Pro model        (~5.9 GB)"),
    ("Pro/text_encoder.pth",       "SAT-Pro text enc.    (~440 MB)"),
]

for rel, label in files:
    target = checkpoint_dir / rel
    if target.exists() and target.stat().st_size > 0:
        print(f"  ✓ {label} — already present")
        continue
    print(f"  ↓ {label} — downloading...")
    hf_hub_download(repo_id=hf_repo, filename=rel, local_dir=checkpoint_dir)

print("  ✓ all checkpoints in place")
PYTHON_SCRIPT
echo ""

# --- Patch mcp_servers.json ---------------------------------------------------

echo -e "${BLUE}[5/5] Wiring up mcp_servers.json${NC}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MCP_JSON="$REPO_ROOT/mcp_servers.json"
MCP_EXAMPLE="$REPO_ROOT/mcp_servers.example.json"

if [ ! -f "$MCP_JSON" ]; then
    if [ -f "$MCP_EXAMPLE" ]; then
        cp "$MCP_EXAMPLE" "$MCP_JSON"
        echo "  Created $MCP_JSON from mcp_servers.example.json"
    else
        echo -e "  ${YELLOW}No mcp_servers.json or mcp_servers.example.json found — skipping patch${NC}"
    fi
fi

if [ -f "$MCP_JSON" ] && [ -n "$SAT_PYTHON_DETECTED" ]; then
    MCP_JSON_PATH="$MCP_JSON" \
    SAT_PYTHON_DETECTED="$SAT_PYTHON_DETECTED" \
    SAT_DIR="$SAT_DIR" \
    CHECKPOINT_DIR="$CHECKPOINT_DIR" \
    python3 << 'PYTHON_PATCH'
import json
import os
from pathlib import Path

mcp_path = Path(os.environ["MCP_JSON_PATH"])
sat_python = os.environ["SAT_PYTHON_DETECTED"]
sat_dir = os.environ["SAT_DIR"]
ckpt_dir = os.environ["CHECKPOINT_DIR"]

with mcp_path.open() as f:
    cfg = json.load(f)

servers = cfg.get("servers", [])
sat_block = next((s for s in servers if s.get("name") == "sat"), None)

new_block = {
    "name": "sat",
    "_description": "SAT medical-image segmentation (Phase 2A).",
    "command": sat_python,
    "args": ["mcp_servers/sat_segmentation/sat_server.py"],
    "env": {
        "SAT_REPO_PATH": sat_dir,
        "SAT_CHECKPOINT_DIR": ckpt_dir,
        "DATASET_PATH": "${BIO_DATA_ROOT}",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"mcp_servers/sat_segmentation:{sat_dir}",
    },
}

if sat_block is None:
    servers.append(new_block)
    action = "added"
else:
    sat_block.clear()
    sat_block.update(new_block)
    action = "updated"

cfg["servers"] = servers
with mcp_path.open("w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")

print(f"  ✓ {action} 'sat' server in {mcp_path}")
print(f"      command            = {sat_python}")
print(f"      SAT_REPO_PATH      = {sat_dir}")
print(f"      SAT_CHECKPOINT_DIR = {ckpt_dir}")
PYTHON_PATCH
elif [ -z "$SAT_PYTHON_DETECTED" ]; then
    echo -e "  ${YELLOW}SAT_PYTHON_DETECTED is empty (conda not found?) — skipping mcp_servers.json patch${NC}"
fi

echo ""
echo -e "${GREEN}✓${NC} Setup complete. Switch back to the veritas env and run experiments:"
echo "  conda activate veritas"
echo ""
