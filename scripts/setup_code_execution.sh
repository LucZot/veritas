#!/bin/bash
# Code Execution MCP Server Setup Script
# Installs dependencies for the code execution sandbox

set -e  # Exit on error

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "Code Execution MCP Server Setup"
echo "================================================================================"
echo ""

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CODE_EXEC_DIR="$REPO_ROOT/mcp_servers/code_execution"

echo -e "${BLUE}Installation Directory${NC}"
echo "================================================================================"
echo "Repository: $REPO_ROOT"
echo "Code Execution: $CODE_EXEC_DIR"
echo ""

# Check if requirements.txt exists
if [ ! -f "$CODE_EXEC_DIR/requirements.txt" ]; then
    echo -e "${YELLOW}Error: requirements.txt not found in $CODE_EXEC_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}Installing Dependencies${NC}"
echo "================================================================================"
echo "This will install:"
echo "  - MCP server framework (mcp, anyio)"
echo "  - Scientific computing (numpy, pandas, scipy, sklearn, statsmodels)"
echo "  - Visualization (matplotlib, seaborn)"
echo "  - Medical imaging (nibabel, SimpleITK)"
echo ""

# Ask for confirmation
read -p "Continue with installation? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

# Install dependencies
echo ""
echo "Installing Python packages..."
pip install -r "$CODE_EXEC_DIR/requirements.txt"

echo ""
echo -e "${GREEN}✓${NC} Dependencies installed successfully"
echo ""

# Verify installation
echo -e "${BLUE}Verifying Installation${NC}"
echo "================================================================================"

# Check key packages
PACKAGES=("mcp" "numpy" "pandas" "scipy" "sklearn" "matplotlib")
ALL_INSTALLED=true

for pkg in "${PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $pkg"
    else
        echo -e "${YELLOW}✗${NC} $pkg (installation may have failed)"
        ALL_INSTALLED=false
    fi
done

echo ""

if [ "$ALL_INSTALLED" = true ]; then
    echo -e "${GREEN}✓ All packages verified successfully!${NC}"
else
    echo -e "${YELLOW}⚠ Some packages may not have installed correctly${NC}"
    echo "Try running: pip install -r $CODE_EXEC_DIR/requirements.txt"
    exit 1
fi
