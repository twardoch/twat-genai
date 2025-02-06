# twat-genai

(work in progress)

Image generation package that leverages fal.ai's models for AI image generation. Provides a flexible command-line interface and Python API for generating images using various AI models and techniques.

## Features

- Multiple AI image generation modes:
  - Text-to-image generation
  - Image-to-image transformation
  - Canny edge-guided generation
  - Depth-guided generation
- Support for LoRA (Low-Rank Adaptation) models with a built-in library of style presets
- Flexible prompt expansion with alternatives using brace syntax
- Concurrent image generation for multiple prompts
- Comprehensive metadata storage for generated images
- Modern Python packaging with PEP 621 compliance
- Type hints and runtime type checking
- Comprehensive test suite and documentation
- CI/CD ready configuration

## Installation

```bash
pip install twat-genai
```


## Usage

### Command Line Interface

```bash
## Basic text-to-image generation
python -m twat_genai "a beautiful sunset" --output_dir images
## Using a specific style from the LoRA library
python -m twat_genai "a beautiful sunset" --lora "shou_xin"
## Image-to-image transformation
python -m twat_genai "enhance this photo" --model image --input_image input.jpg
## Multiple prompts with alternatives
python -m twat_genai "a {red; blue; green} house with {white; black} windows"
```

### Python API

```python
import twat_genai
from twat_genai.main import async_main, ModelTypes

## Generate images asynchronously
results = await async_main(
prompts="a beautiful sunset",
output_dir="generated_images",
model=ModelTypes.TEXT,
lora="shou_xin",
image_size="SQ"
)   
```


## Key Features in Detail

### Prompt Expansion
The tool supports flexible prompt expansion using brace syntax:
- `"a {red; blue} house"` generates two images: "a red house" and "a blue house"
- Nested alternatives are supported
- Semicolons separate alternatives

### LoRA Styles
Built-in library of LoRA styles for different artistic effects:
- Gesture drawing
- Sketch and smudge effects
- 2-color illustrations
- Pencil sketches
- Tarot card style
- And more...

### Image Generation Modes
- **Text-to-Image**: Generate images from text descriptions
- **Image-to-Image**: Transform existing images
- **Canny Edge**: Use edge detection to guide generation
- **Depth-Guided**: Use depth information for generation

### Output Management
- Automatic file naming with customizable prefixes/suffixes
- Metadata storage in JSON format
- Various image size options (square, landscape, portrait)
- Support for custom dimensions

## Development

This project uses [Hatch](https://hatch.pypa.io/) for development workflow management.

### Setup Development Environment

```bash
## Install hatch if you haven't already
pip install hatch
## Create and activate development environment
hatch shell
## Run tests
hatch run test
## Run tests with coverage
hatch run test-cov
## Run linting
hatch run lint
## Format code
hatch run format
```

## License

MIT License 