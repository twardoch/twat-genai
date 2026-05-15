#!/usr/bin/env bash
# install.sh — Install twat-genai locally
# Generative AI utilities plugin for twat
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing twat-genai..."
uv pip install -e . 2>/dev/null || pip install -e .
echo "Done."
