# Installation Guide

This guide covers various ways to install and use `twat-genai`.

## Quick Install

### From PyPI (Recommended)

```bash
pip install twat-genai
```

### From GitHub Releases (Binary)

Download the latest binary from the [releases page](https://github.com/twardoch/twat-genai/releases):

- **Linux**: `twat-genai-Linux`
- **macOS**: `twat-genai-macOS` 
- **Windows**: `twat-genai-Windows.exe`

Make the binary executable (Linux/macOS):
```bash
chmod +x twat-genai-Linux
./twat-genai-Linux --help
```

### From Source

```bash
git clone https://github.com/twardoch/twat-genai.git
cd twat-genai
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

## Development Installation

For development, use the provided scripts:

```bash
git clone https://github.com/twardoch/twat-genai.git
cd twat-genai

# Install UV package manager
pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[all]"

# Install development dependencies
uv pip install pytest pytest-cov pytest-asyncio pytest-mock ruff mypy
```

## Requirements

- Python 3.10 or higher
- FAL API key from [fal.ai](https://fal.ai/)

## Configuration

Set your FAL API key:

```bash
export FAL_KEY="your-fal-api-key"
```

Or create a `.env` file:
```
FAL_KEY=your-fal-api-key
```

## Verification

Test your installation:

```bash
twat-genai --help
```

For development installation:
```bash
python -m twat_genai --help
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed with `pip install -e ".[all]"`
2. **FAL API errors**: Verify your API key is set correctly
3. **Version conflicts**: Use a virtual environment to avoid conflicts

### Getting Help

- Check the [main README](README.md) for usage examples
- Report issues on [GitHub Issues](https://github.com/twardoch/twat-genai/issues)
- Review the [documentation](https://github.com/twardoch/twat-genai#readme)