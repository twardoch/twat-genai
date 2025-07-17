#!/bin/bash
# Build script for twat-genai

set -e

echo "🏗️  Building twat-genai..."

# Ensure virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/
rm -rf build/
rm -rf src/twat_genai.egg-info/

# Install build dependencies
echo "📦 Installing build dependencies..."
pip install -q build

# Build package
echo "🔨 Building package..."
python -m build

echo "✅ Build completed successfully!"
echo "📂 Built packages are in dist/"
ls -la dist/