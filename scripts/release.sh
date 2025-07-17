#!/bin/bash
# Release script for twat-genai

set -e

echo "🚀 Preparing release for twat-genai..."

# Check if version tag is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 v1.0.0"
    exit 1
fi

VERSION=$1

# Ensure virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Validate version format
if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "❌ Version must be in format vX.Y.Z (e.g., v1.0.0)"
    exit 1
fi

# Check if working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "❌ Working directory is not clean. Please commit or stash changes."
    exit 1
fi

# Update to latest main branch
echo "📥 Updating to latest main branch..."
git checkout main
git pull origin main

# Run tests
echo "🧪 Running tests..."
./scripts/test.sh

# Build package
echo "🏗️  Building package..."
./scripts/build.sh

# Create and push tag
echo "🏷️  Creating tag $VERSION..."
git tag -a "$VERSION" -m "Release $VERSION"
git push origin "$VERSION"

echo "✅ Release $VERSION created successfully!"
echo "🎉 GitHub Actions will automatically build and publish the release."