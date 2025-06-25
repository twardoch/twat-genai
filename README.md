# twat-genai

## Overview

`twat-genai` is a Python package designed to provide a unified interface for various generative AI image tasks, currently focusing on models available through the FAL (fal.ai) platform. It allows users to perform Text-to-Image, Image-to-Image, ControlNet-like operations (Canny, Depth), Image Upscaling, and Image Outpainting via both a command-line interface (CLI) and a Python API.

The package aims for modularity and extensibility, separating concerns into core components, engine implementations, and the user interface.

## Installation

1.  **Prerequisites**:
    *   Python 3.10 or higher.
    *   `uv` (recommended for faster installation): `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`.
    *   `git` (for versioning).
2.  **Clone the repository**:
    ```bash
    git clone https://github.com/twardoch/twat-genai.git
    cd twat-genai
    ```
3.  **Set up environment and install**:
    ```bash
    # Create virtual environment (optional but recommended)
    uv venv
    source .venv/bin/activate  # On Linux/macOS
    # .venv\Scripts\activate  # On Windows

    # Install the package in editable mode with all dependencies
    uv pip install -e ".[all]"
    ```
4.  **Set FAL API Key**:
    Obtain an API key from [fal.ai](https://fal.ai/) and set it as an environment variable:
    ```bash
    export FAL_KEY="your-fal-api-key"
    # Or add FAL_KEY="your-fal-api-key" to a .env file in the project root.
    ```

## Architecture

The package follows a layered architecture:

1.  **Core (`src/twat_genai/core`)**: Contains fundamental building blocks and utilities independent of specific AI engines.
    *   `config.py`: Defines core data structures like `ImageInput` (representing input images via URL, path, or PIL object), `ImageResult` (standardized output containing metadata, paths, and results), `ImageSizeWH`, type aliases (`Prompts`, `OutputDir`, etc.), and `ImageInput` validation logic.
    *   `lora.py`: Defines Pydantic models for representing LoRA configurations (`LoraRecord`, `LoraRecordList`, `LoraLib`, `LoraSpecEntry`, `CombinedLoraSpecEntry`). Handles loading LoRA definitions from JSON.
    *   `image.py`: Defines image format (`ImageFormats`) and size (`ImageSizes`) enums, and provides a utility function `save_image` for saving PIL images to disk. Also includes `validate_image_size` for parsing size strings.
    *   `image_utils.py`: Provides utilities for handling images, such as asynchronous downloading from URLs (`download_image_to_temp`, `download_image` using `httpx`) and resizing based on model constraints (`resize_image_if_needed`, utilizing size limits defined in `engines/fal/config.py`).
    *   `prompt.py`: Includes logic for parsing and expanding Midjourney-style prompt syntax, handling image prompts (URLs), multi-prompts (`::`), weighted parts, permutation prompts (`{alt1, alt2}`), and parameters (`--param value`). Provides `normalize_prompts` for processing input prompts.
    *   `__init__.py`: Makes the `core` directory a Python package.

2.  **Engines (`src/twat_genai/engines`)**: Implements the logic for interacting with specific AI platforms or model families.
    *   `base.py`: Defines the abstract base class `ImageGenerationEngine` which specifies the common interface (`initialize`, `generate`, `shutdown`, async context manager support) and the base `EngineConfig` Pydantic model (containing `guidance_scale`, `num_inference_steps`, `image_size`, `enable_safety_checker`).
    *   **FAL Engine (`src/twat_genai/engines/fal`)**: Implementation for the FAL platform.
        *   `__init__.py`: Contains the `FALEngine` class, which inherits from `ImageGenerationEngine`. It orchestrates the process by:
            *   Initializing the `FalApiClient` on `initialize()` or first `generate()` call, checking for the `FAL_KEY` env var.
            *   Handling input image preparation (downloading URLs via `core.image_utils`, resizing for upscalers via `core.image_utils`, uploading local/temp files via `client.upload_image`) in `_prepare_image_input`. Handles temp file cleanup.
            *   Receiving generation requests via the `generate` method.
            *   Determining the operation type (TTI, I2I, Upscale, Outpaint, Canny, Depth) based on the `model` parameter (`ModelTypes` enum).
            *   FalApiClient (e.g., `process_upscale`, `process_outpaint`, `process_tti`, `process_i2i`).
            *   Handling `shutdown`.
        *   `client.py`: Contains the `FalApiClient` class responsible for direct interaction with the FAL API using the `fal_client` library.
            *   Provides `upload_image` using `fal_client.upload_file_async`.
            *   Includes methods for specific operations: `process_tti`, `process_i2i`, `process_canny`, `process_depth`, `process_upscale`, `process_outpaint`, `process_genfill`.
            *   These methods delegate job submission to the top-level helper `_submit_fal_job` (which uses `fal_client.submit_async`).
            *   Result retrieval and processing happen in `_get_fal_result`, which polls using `fal_client.status_async` and fetches with `fal_client.result_async`.
            *   Parses results using a dispatch mechanism (`_get_result_extractor`) that selects an appropriate parsing function (currently `_extract_generic_image_info`) to standardize output from different FAL endpoints into `image_info` dicts.
            *   Handles downloading the final image using the top-level helper `_download_image_helper` (uses `httpx`).
            *   Constructs and returns the final `ImageResult` object, saving metadata to a JSON file if `output_dir` is provided.
        *   `config.py`: Defines FAL-specific Pydantic configuration models used as *schemas* for arguments:
            *   `ModelTypes` enum mapping model names to FAL API endpoint strings.
            *   `ImageToImageConfig`, `UpscaleConfig` (with many tool-specific fields), and `OutpaintConfig`. These models expect an image input conforming to the `FALImageInput` structure.
        *   `lora.py`: Contains FAL-specific LoRA handling logic:
            *   Loading the LoRA library from JSON (`get_lora_lib`, using `__main___loras.json` or `TWAT_GENAI_LORA_LIB` path).
            *   Parsing LoRA specification strings (library keys, `url:scale` syntax) via `parse_lora_phrase`.
            *   Normalizing various input spec formats (`str`, `list`, `dict`) into a list of `LoraSpecEntry` or `CombinedLoraSpecEntry` objects using `normalize_lora_spec`.
            *   Building the final LoRA argument list (list of dicts with `path` and `scale`) and augmenting the text prompt for the FAL API call using `build_lora_arguments`.
        *   `models.py`: Defines the concrete `FALImageInput` class, inheriting from `core.config.ImageInput`. Its `to_url` async method implements the logic to convert Path or PIL image inputs into uploadable URLs using `fal_client.upload_file_async`.

3.  **CLI (`src/twat_genai/cli.py`)**: Provides the command-line interface using the `python-fire` library.
    *   Defines the main `TwatGenAiCLI` class whose methods are exposed as CLI subcommands.
    *   Maps CLI arguments to the appropriate `ModelTypes`, constructs the base `EngineConfig`, and creates specific configuration objects (`ImageToImageConfig`, `UpscaleConfig`, `OutpaintConfig`) as needed.
    *   Handles parsing of `input_image` (path vs URL) into `core.config.ImageInput`.
    *   Each command method (e.g., `text`, `image`, `upscale`) calls a shared `_run_generation` async helper, which in turn instantiates and runs the `FALEngine`. `asyncio.run()` is used within each command method.
    *   Handles output directory resolution, defaulting to `./generated_images` or a subfolder of the input image's directory.
    *   Includes helper functions for parsing arguments like `image_size`.

4.  **Entry Point (`src/twat_genai/__main__.py`)**: A minimal script that allows the package to be run as a module (`python -m twat_genai`). It simply imports `TwatGenAiCLI` and uses `fire.Fire(TwatGenAiCLI)`.

5.  **Tests (`tests/`)**: Contains unit tests using `pytest` and `unittest.mock`.
    *   `test_fal_client.py`: Tests the `FalApiClient` methods (like `process_tti`, `_extract_generic_image_info`) and internal helpers, using mocked dependencies (`fal_client` calls, LoRA building, file system access).
    *   `test_image_utils.py`: Tests image downloading (`download_image_to_temp`, `download_image`) and resizing (`resize_image_if_needed`) utilities.
    *   `test_twat_genai.py`: Basic package tests (e.g., checking `__version__`).
    *   `conftest.py`: Provides shared `pytest` fixtures, such as `mock_fal_api_client` for injecting a mocked client instance into tests.

## Functionality

1.  **Initialization**: When the `FALEngine` is instantiated (either via CLI or Python API) and its `initialize()` method is called (or implicitly via context manager `__aenter__` or first `generate()` call), it ensures the `FAL_KEY` environment variable is set and initializes the `FalApiClient`.
2.  **Input Handling**:
    *   Accepts prompts as strings or lists of strings. Supports brace expansion for permutations (e.g., `a {red,blue} car`) and semicolon separation for multiple distinct prompts via `core.prompt.normalize_prompts`.
    *   Accepts image inputs as URLs, local file paths, or PIL Images via the `core.config.ImageInput` model. The `engines.fal.models.FALImageInput` subclass handles the conversion of paths/PIL images to URLs by uploading them using `fal_client.upload_file_async` when its `to_url()` method is invoked (typically within `FALEngine._prepare_image_input`).
    *   Handles LoRA specifications via strings (keywords defined in `__main___loras.json`, `url:scale` pairs, `;` separated lists) or structured lists/dicts. The `engines.fal.lora.build_lora_arguments` function parses the spec and prepares the arguments for the FAL API.
3.  **Configuration**: Uses Pydantic models (`EngineConfig`, `ImageToImageConfig`, `UpscaleConfig`, `OutpaintConfig`) for type validation and structuring configuration. These models are populated based on CLI arguments or Python API calls and passed to the `FALEngine.generate` method.
4.  **Image Preparation (`FALEngine._prepare_image_input`)**: Before calling the FAL API for modes requiring an input image (I2I, Canny, Depth, Upscale, Outpaint), this method ensures a usable image URL is available:
    *   If the input is a URL, it's downloaded to a temporary file using `core.image_utils.download_image_to_temp`.
    *   If the input is a PIL Image, it's saved to a temporary file.
    *   If the operation is an upscale, it checks the dimensions against size limits (defined in `engines/fal/config.py`). If the image exceeds the limits for the specific `model_type`, it's resized using `core.image_utils.resize_image_if_needed`, saving the result to another temporary file.
    *   The final image file (original path, downloaded temp, or resized temp) is uploaded using `client.upload_image` to get a FAL-usable URL.
    *   Temporary files created during this process are cleaned up afterwards.
5.  **API Interaction (`FalApiClient`)**:
    *   Builds the final API argument dictionary based on the operation type, base config, specific config, and processed LoRA/prompt information.
    *   Submits the job to the appropriate FAL endpoint (defined in `ModelTypes` enum) using `fal_client.submit_async` via the `_submit_fal_job` helper.
    *   Polls for job status using `fal_client.status_async` in `_get_fal_result`.
    *   Fetches the final result dictionary using `fal_client.result_async` in `_get_fal_result`.
    *   Parses the result structure using `_extract_generic_image_info` (selected via `_get_result_extractor`) to extract key information like image URL(s), dimensions, content type, and seed (if provided by the API).
6.  **Result Handling (`FalApiClient._get_fal_result`)**:
    *   Downloads the generated image URL(s) to the specified `output_dir` using `_download_image_helper`.
    *   Constructs a standardized `ImageResult` object (`core.config.ImageResult`) containing:
        *   Request ID, timestamp.
        *   The raw API result dictionary.
        *   Parsed `image_info` dictionary (containing the final local path, URL, dimensions, content type, seed, metadata path, etc.).
        *   Original prompt and the full `job_params` dictionary used for the request (for reproducibility/logging).
    *   Saves the `ImageResult` object (excluding the PIL image itself) as a JSON metadata file alongside the downloaded image in the `output_dir`.
    *   Returns the `ImageResult` object (without the PIL image loaded by default).
7.  **CLI Operation (`cli.py`)**: The `TwatGenAiCLI` class methods translate flags and arguments into the necessary configuration objects (`EngineConfig`, `ImageInput`, `ImageToImageConfig`, `UpscaleConfig`, `OutpaintConfig`). Each method then calls `_run_generation`, which instantiates the `FALEngine` and invokes its `generate` method. `asyncio.run()` is used within each CLI method to manage the async execution. Basic information about the generated result(s) is printed. Error handling for invalid arguments or configuration issues is included.

## Usage

### CLI

The main entry point is the `twat-genai` command (or `python -m twat_genai`).

```bash
# Basic Text-to-Image (TTI)
twat-genai --prompts "a futuristic cityscape at sunset" --output_dir generated --filename_prefix city

# TTI with multiple prompts and brace expansion
twat-genai --prompts "a photo of a {red,blue} car; a drawing of a {cat,dog}" --image_size HD

# Image-to-Image (I2I)
twat-genai --model image --input_image path/to/input.jpg --prompts "make it look like an oil painting" --strength 0.65

# ControlNet Canny
twat-genai --model canny --input_image path/to/drawing.png --prompts "a detailed spaceship based on the sketch"

# Upscale using ESRGAN
twat-genai --model upscale --input_image path/to/lowres.png --upscale_tool esrgan --scale 4 --output_dir upscaled

# Upscale using Ideogram (requires prompt)
twat-genai --model upscale --input_image path/to/photo.jpg --upscale_tool ideogram --prompts "enhance the details of the landscape photo"

# Outpaint an image
twat-genai --model outpaint --input_image path/to/center_image.png --prompts "expand the scene with a forest on the left and a river on the right" --target_width 2048 --target_height 1024

# Using LoRA from library (defined in __main___loras.json)
twat-genai --prompts "a portrait" --lora "gstdrw style"

# Using specific LoRA URL with scale
twat-genai --prompts "a robot" --lora "https://huggingface.co/path/to/lora:0.8"

# Verbose logging
twat-genai --prompts "debug prompt" --verbose
```

**Common Arguments:**

*   `--prompts`: One or more prompts (string or list). Use `;` to separate multiple prompts in a single string. Use `{a,b}` for permutations.
*   `--output_dir`: Directory to save results (defaults to `generated_images`).
*   `--model`: Operation type (`text`, `image`, `canny`, `depth`, `upscale`, `outpaint`). Default: `text`.
*   `--input_image`: Path or URL to an input image (required for non-text models).
*   `--filename_suffix`, `--filename_prefix`: Customize output filenames.
*   `--image_size`: Output image size preset (`SQ`, `SQL`, `SD`, `HD`, `SDV`, `HDV`) or custom `width,height`. Default: `SQ`.
*   `--guidance_scale`, `--num_inference_steps`: Control generation process.
*   `--negative_prompt`: Specify negative prompts.
*   `--lora`: LoRA specification string (see examples).
*   `--strength`: Control influence of input image in I2I (0.0 to 1.0). Default: 0.75.
*   `--upscale_tool`: Name of the upscaler model (e.g., `esrgan`, `ideogram`, `ccsr`). Required for `--model upscale`.
*   `--scale`: Target upscale factor (e.g., 2, 4). Used by some upscalers.
*   `--target_width`, `--target_height`: Target dimensions for outpainting. Required for `--model outpaint`.
*   `--verbose`: Enable debug logging.

### Python API

```python
import asyncio
from pathlib import Path
from twat_genai import (
    FALEngine, ModelTypes, EngineConfig, ImageInput,
    ImageToImageConfig, FALImageInput, UpscaleConfig, OutpaintConfig
)

async def run_generation():
    output_dir = Path("api_generated")
    # Use context manager for initialization and shutdown
    async with FALEngine(output_dir=output_dir) as engine:

        # --- Text-to-Image ---
        print("--- Running Text-to-Image ---")
        tti_config = EngineConfig(image_size="HD", guidance_scale=4.0)
        result_tti = await engine.generate(
            prompt="A photorealistic image of a bioluminescent forest at night",
            config=tti_config,
            model=ModelTypes.TEXT,
            filename_prefix="bioluminescent_forest"
        )
        print(f"TTI Result Path: {result_tti.image_info.get('path')}")
        print(f"TTI Metadata Path: {result_tti.image_info.get('metadata_path')}")

        # --- Image-to-Image ---
        print("\n--- Running Image-to-Image ---")
        # Use a placeholder URL for the example
        input_img_i2i_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
        print(f"Using input image URL: {input_img_i2i_url}")
        # Or use a local path if needed:
        # input_img_i2i = ImageInput(path=Path("input.jpg"))
        # if not input_img_i2i.path or not input_img_i2i.path.exists():
        #      print("Skipping I2I: input.jpg not found.")
        # else:
        i2i_base_config = EngineConfig(num_inference_steps=30)
        # Create FALImageInput directly from URL
        i2i_model_config = ImageToImageConfig(
            model_type=ModelTypes.IMAGE, # Technically redundant if passed to generate
            input_image=FALImageInput(url=input_img_i2i_url),
            strength=0.7,
            negative_prompt="blurry, low quality"
        )
        result_i2i = await engine.generate(
            prompt="Convert this image to a comic book style",
            config=i2i_base_config,
            model=ModelTypes.IMAGE,
            image_config=i2i_model_config, # Pass the specific config
            lora_spec="shou_xin:0.5", # Example LoRA spec
            filename_prefix="comic_style"
        )
        print(f"I2I Result Path: {result_i2i.image_info.get('path')}")

        # --- Upscale ---
        print("\n--- Running Upscale ---")
        # Use a placeholder URL for the example
        input_img_upscale_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/mountains-input.png"
        print(f"Using input image URL: {input_img_upscale_url}")
        # Or use a local path:
        # input_img_upscale = ImageInput(path=Path("low_res.png"))
        # if not input_img_upscale.path or not input_img_upscale.path.exists():
        #     print("Skipping Upscale: low_res.png not found.")
        # else:
        upscale_base_config = EngineConfig() # Upscalers might ignore some base params
        upscale_model_config = UpscaleConfig(
                input_image=FALImageInput(url=input_img_upscale_url),
                # Prompt might be needed for some upscalers like Ideogram
                prompt="Enhance details, sharp focus, high resolution",
                scale=4 # General scale, specific tools might override
        )
        result_upscale = await engine.generate(
                prompt="Enhance details, sharp focus, high resolution", # Often passed directly too
                config=upscale_base_config,
                model=ModelTypes.UPSCALER_ESRGAN, # Choose the specific upscaler
                upscale_config=upscale_model_config, # Pass the specific config
                filename_prefix="upscaled_esrgan"
        )
        print(f"Upscale Result Path: {result_upscale.image_info.get('path')}")

        # --- Outpaint ---
        print("\n--- Running Outpaint ---")
        # Use a placeholder URL for the example
        input_img_outpaint_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/mountains-input.png"
        print(f"Using input image URL: {input_img_outpaint_url}")
        # Or use a local path:
        # input_img_outpaint = ImageInput(path=Path("center.png"))
        # if not input_img_outpaint.path or not input_img_outpaint.path.exists():
        #      print("Skipping Outpaint: center.png not found.")
        # else:
        outpaint_base_config = EngineConfig()
        outpaint_model_config = OutpaintConfig(
                input_image=FALImageInput(url=input_img_outpaint_url),
                prompt="Expand the mountain scene with a clear blue sky and forest below",
                target_width=1536,
                target_height=1024
        )
        result_outpaint = await engine.generate(
                prompt="Expand the mountain scene with a clear blue sky and forest below", # Pass prompt here too
                config=outpaint_base_config,
                model=ModelTypes.OUTPAINT_BRIA,
                outpaint_config=outpaint_model_config,
                filename_prefix="outpainted_scene"
        )
        print(f"Outpaint Result Path: {result_outpaint.image_info.get('path')}")


if __name__ == "__main__":
    # Remove the dummy file creation logic to make the example cleaner
    # Users should provide their own images or URLs if modifying the example.
    # print("Note: This example uses placeholder URLs. Provide image paths or URLs for real use.")
    # try:
    #     from PIL import Image
    #     dummy_size = (512, 512)
    #     if not Path("input.jpg").exists(): Image.new('RGB', dummy_size, color = 'red').save("input.jpg")
    #     if not Path("low_res.png").exists(): Image.new('RGB', (256, 256), color = 'blue').save("low_res.png")
    #     if not Path("center.png").exists(): Image.new('RGB', (512, 512), color = 'green').save("center.png")
    # except ImportError:
    #     print("Pillow not installed, cannot create dummy images if paths are used.")

    # Check for FAL_KEY
    import os
    if not os.getenv("FAL_KEY"):
        print("Error: FAL_KEY environment variable not set.")
        print("Please set your FAL API key: export FAL_KEY='your-key'")
    else:
        asyncio.run(run_generation())

## Maintenance & Development

*   **Code Quality**: The project uses `ruff` for linting and formatting, `mypy` for static type checking, and `pytest` for unit testing.
    *   Run checks locally: `uv run lint` and `uv run test`.
    *   Pre-commit hooks are configured in `.pre-commit-config.yaml` to enforce quality standards before committing. Install hooks with `pre-commit install`.
*   **Dependencies**: Managed using `uv` and specified in `pyproject.toml`. Install with `uv pip install -e ".[all]"`.
*   **Versioning**: Handled by `hatch-vcs`, deriving the version from `git` tags.
*   **Documentation**:
    *   `README.md`: This file.
    *   `LOG.md`: Changelog.
    *   `TODO.md`: Task tracking.
    *   Docstrings in code.
*   **CI/CD**: GitHub Actions workflows in `.github/workflows` handle:
    *   `push.yml`: Runs quality checks, tests, and builds distributions on pushes/PRs to `main`.
    *   `release.yml`: Builds and publishes the package to PyPI and creates a GitHub Release when a tag matching `v*` is pushed.


