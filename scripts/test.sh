#!/bin/bash
# Test script for twat-genai

set -e

echo "🧪 Testing twat-genai..."

# Ensure virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run linting
echo "🔍 Running linting..."
ruff check src/twat_genai tests --fix || true
ruff format src/twat_genai tests

# Run type checking
echo "🔍 Running type checking..."
mypy src/twat_genai tests || true

# Run tests
echo "🧪 Running tests..."
python -m pytest tests/ -v --tb=short

# Run coverage
echo "📊 Running coverage..."
python -m pytest --cov=src/twat_genai --cov-report=term-missing --cov-report=html tests/

echo "✅ All tests completed!"
