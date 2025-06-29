# `twat-genai`: Generative AI Toolkit

**`twat-genai`** is a powerful Python package that provides a unified command-line interface (CLI) and Python API for a variety of generative AI image tasks. It simplifies interaction with different AI models and services, currently focusing on the [FAL (fal.ai)](https://fal.ai/) platform. This tool is part of the [TWAT (Twardoch Utility Tools)](https://pypi.org/project/twat/) collection.

Whether you're a developer, artist, or researcher, `twat-genai` empowers you to programmatically generate and manipulate images, automate creative workflows, and experiment with cutting-edge AI models.

## Key Features

*   **Unified Interface:** Access Text-to-Image, Image-to-Image, ControlNet-like operations (Canny edges, Depth maps), Image Upscaling, and Image Outpainting through a consistent CLI and API.
*   **FAL.ai Integration:** Leverages the [fal.ai](https://fal.ai/) platform for model hosting and execution.
*   **Flexible Input:** Use text prompts with Midjourney-style syntax (permutations, multi-prompts), and provide input images via URLs, local file paths, or PIL Image objects.
*   **LoRA Support:** Easily apply LoRA (Low-Rank Adaptation) models from a predefined library or by specifying URLs.
*   **Comprehensive Configuration:** Fine-tune generation parameters using Pydantic-based configuration objects.
*   **Standardized Output:** Receive results in a consistent `ImageResult` format, including metadata and paths to generated files.
*   **Extensible Design:** Built with modularity in mind, allowing for future expansion to other AI engines and models.

## Who Is It For?

*   **Developers:** Integrate generative AI capabilities into your Python applications.
*   **Artists & Designers:** Experiment with AI-powered image creation and manipulation, and automate parts of your creative workflow.
*   **Researchers:** Conduct experiments and batch process image generation tasks.
*   **CLI Users:** Quickly generate images or perform image operations directly from your terminal.

## Why Use `twat-genai`?

*   **Simplicity:** Abstracts away the complexities of individual AI model APIs.
*   **Consistency:** Provides a standardized way to interact with different types of generative models.
*   **Automation:** Enables scripting and automation of image generation tasks.
*   **Reproducibility:** Saves metadata with generation parameters for better tracking.
*   **Power & Flexibility:** Offers fine-grained control over the generation process.

## Installation

### Prerequisites

*   Python 3.10 or higher.
*   `uv` (recommended for faster installation and environment management):
    *   Install `uv`: `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`.
*   `git` (for cloning the repository).

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/twardoch/twat-genai.git
    cd twat-genai
    ```

2.  **Set Up Virtual Environment and Install:**
    ```bash
    # Create a virtual environment (recommended)
    uv venv
    source .venv/bin/activate  # On Linux/macOS
    # .venv\Scripts\activate    # On Windows

    # Install the package in editable mode with all dependencies
    uv pip install -e ".[all]"
    ```

3.  **Set FAL API Key:**
    You need an API key from [fal.ai](https://fal.ai/).
    *   Set it as an environment variable:
        ```bash
        export FAL_KEY="your-fal-api-key"
        ```
    *   Or, create a `.env` file in the project root (`twat-genai/`) with the following content:
        ```
        FAL_KEY="your-fal-api-key"
        ```

## Basic Usage (CLI)

The main command is `twat-genai`. You can explore commands and options with `twat-genai --help`.

### Common Arguments

*   `--prompts "your prompt"`: The text prompt for generation.
    *   Multiple prompts (semicolon separated): `"a cat; a dog"`
    *   Permutations: `"a {red,blue} car"` (generates "a red car" and "a blue car")
*   `--output_dir <path>`: Directory to save generated images (default: `generated_images`).
*   `--model <model_type>`: Specifies the operation (e.g., `text`, `image`, `canny`, `upscale`, `outpaint`).
*   `--input_image <path_or_url>`: Path or URL to an input image (for `image`, `canny`, `depth`, `upscale`, `outpaint`).
*   `--image_size <preset_or_WxH>`: Output image size (e.g., `SQ`, `HD`, `1024,768`). Default: `SQ` (1024x1024).
*   `--lora "<lora_name_or_url:scale>"`: Apply a LoRA.
    *   From library: `--lora "gstdrw style"`
    *   URL with scale: `--lora "https://huggingface.co/path/to/lora:0.7"`
    *   Multiple LoRAs: `--lora "name1:0.5; url2:0.8"`
*   `--verbose`: Enable detailed logging.
*   `--filename_prefix <prefix>`: Prepend text to output filenames.
*   `--negative_prompt "text"`: Specify what to avoid in the image.

### Examples

**1. Text-to-Image (TTI):**
```bash
# Basic TTI
twat-genai text --prompts "A futuristic cityscape at sunset, neon lights, cinematic"

# TTI with specific size and multiple prompts
twat-genai text --prompts "photo of a majestic lion; illustration of a mythical phoenix" --image_size HD

# TTI with a LoRA
twat-genai text --prompts "portrait of a warrior" --lora "shou_xin:0.8" --output_dir my_portraits
```

**2. Image-to-Image (I2I):**
```bash
twat-genai image --input_image path/to/my_photo.jpg --prompts "transform into a vibrant oil painting" --strength 0.65
```

**3. ControlNet-like Operations (Canny Edge):**
```bash
twat-genai canny --input_image path/to/sketch.png --prompts "detailed spaceship based on the sketch, metallic texture"
```
*(Depth map generation is similar: `twat-genai depth ...`)*

**4. Image Upscaling:**
```bash
# Upscale using ESRGAN (general purpose)
twat-genai upscale --input_image path/to/low_res_image.png --tool esrgan --output_dir upscaled_images

# Upscale using Ideogram (requires prompt, good for creative upscaling)
twat-genai upscale --input_image path/to/artwork.jpg --tool ideogram --prompts "enhance details, painterly style" --scale 2

# Upscale using Clarity (photo enhancement)
twat-genai upscale --input_image path/to/photo.jpg --tool clarity --prompts "ultra realistic photo, sharp details"
```
*Supported tools for `--tool`: `aura_sr`, `ccsr`, `clarity`, `drct`, `esrgan`, `ideogram`, `recraft_clarity`, `recraft_creative`.*

**5. Image Outpainting:**
```bash
# Outpaint using Bria (default)
twat-genai outpaint --input_image path/to/center_image.png \
    --prompts "expand the scene with a lush forest on the left and a serene lake on the right" \
    --target_width 2048 --target_height 1024

# Outpaint using Flux (alternative, may require different prompting)
twat-genai outpaint --input_image path/to/center_image.png --tool flux \
    --prompts "fantasy landscape expanding outwards" \
    --target_width 1920 --target_height 1080
```

## Basic Usage (Python API)

The Python API offers more flexibility for integration into your projects.

```python
import asyncio
from pathlib import Path
from twat_genai import (
    FALEngine,
    ModelTypes,
    EngineConfig,
    ImageInput, # Base ImageInput
    FALImageInput, # For FAL-specific interactions if needed, usually handled by FALEngine
    ImageToImageConfig,
    UpscaleConfig,
    OutpaintConfig,
)

async def main():
    output_dir = Path("api_generated_images")
    output_dir.mkdir(exist_ok=True)

    # Ensure FAL_KEY is set in your environment or .env file

    async with FALEngine(output_dir=output_dir) as engine:
        # --- 1. Text-to-Image ---
        print("Running Text-to-Image...")
        base_config_tti = EngineConfig(image_size="HD", num_inference_steps=30)
        result_tti = await engine.generate(
            prompt="A stunning fantasy castle on a floating island, hyperrealistic",
            config=base_config_tti,
            model=ModelTypes.TEXT, # Specify Text-to-Image model
            filename_prefix="fantasy_castle",
            lora_spec="shou_xin:0.5" # Example LoRA
        )
        if result_tti and result_tti.image_info.get("path"):
            print(f"TTI image saved to: {result_tti.image_info['path']}")
            print(f"TTI metadata: {result_tti.image_info.get('metadata_path')}")
        else:
            print(f"TTI generation failed or image path not found. Result: {result_tti}")


        # --- 2. Image-to-Image ---
        print("\nRunning Image-to-Image...")
        # Replace with your image URL or local path
        input_image_i2i_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
        # For local path: input_img_i2i = ImageInput(path=Path("path/to/your/image.jpg"))

        base_config_i2i = EngineConfig(num_inference_steps=25)
        # FALImageInput will be created internally by FALEngine from ImageInput
        i2i_specific_config = ImageToImageConfig(
            input_image=ImageInput(url=input_image_i2i_url), # Provide base ImageInput
            strength=0.7,
            negative_prompt="blurry, low quality",
            model_type=ModelTypes.IMAGE # Redundant if model passed to generate()
        )
        result_i2i = await engine.generate(
            prompt="Convert this to a cyberpunk city scene",
            config=base_config_i2i,
            model=ModelTypes.IMAGE, # Specify Image-to-Image model
            image_config=i2i_specific_config, # Pass the I2I specific config
            filename_prefix="cyberpunk_city"
        )
        if result_i2i and result_i2i.image_info.get("path"):
            print(f"I2I image saved to: {result_i2i.image_info['path']}")
        else:
            print(f"I2I generation failed or image path not found. Result: {result_i2i}")

        # --- 3. Upscale ---
        print("\nRunning Upscale...")
        # Replace with your image URL or local path
        input_image_upscale_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/mountains-input.png" # Example low-res

        # Base config might be ignored or partially used by upscalers
        base_config_upscale = EngineConfig()
        upscale_specific_config = UpscaleConfig(
            input_image=ImageInput(url=input_image_upscale_url),
            prompt="Enhance details, sharp focus, high resolution", # For Ideogram, Clarity etc.
            # scale=4 # General scale, some tools have specific scale params like ccsr_scale
            # Example for esrgan:
            # esrgan_model="RealESRGAN_x4plus",
            # Example for clarity:
            clarity_creativity=0.5
        )
        result_upscale = await engine.generate(
            prompt="Enhance details, sharp focus, high resolution", # Prompt for context
            config=base_config_upscale,
            # Choose a specific upscaler model from ModelTypes
            model=ModelTypes.UPSCALER_CLARITY,
            upscale_config=upscale_specific_config,
            filename_prefix="upscaled_image_clarity"
        )
        if result_upscale and result_upscale.image_info.get("path"):
            print(f"Upscaled image saved to: {result_upscale.image_info['path']}")
        else:
            print(f"Upscale failed or image path not found. Result: {result_upscale}")

        # --- 4. Outpaint ---
        print("\nRunning Outpaint...")
        # Replace with your image URL or local path
        input_image_outpaint_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/mountains-input.png"

        base_config_outpaint = EngineConfig()
        outpaint_specific_config = OutpaintConfig(
            input_image=ImageInput(url=input_image_outpaint_url),
            prompt="Expand the mountain scene with a vast, starry night sky above and misty valleys below.",
            target_width=1536,
            target_height=1024,
            outpaint_tool="bria", # or "flux"
            # For Bria, you can enable GenFill post-processing via border_thickness_factor > 0
            border_thickness_factor=0.05 # Example: 5% of min dimension for border
        )
        result_outpaint = await engine.generate(
            prompt=outpaint_specific_config.prompt, # Pass prompt from config or directly
            config=base_config_outpaint,
            model=ModelTypes.OUTPAINT_BRIA, # Or ModelTypes.OUTPAINT_FLUX
            outpaint_config=outpaint_specific_config,
            filename_prefix="outpainted_scene_bria"
        )
        if result_outpaint and result_outpaint.image_info.get("path"):
            print(f"Outpainted image saved to: {result_outpaint.image_info['path']}")
        else:
            print(f"Outpaint failed or image path not found. Result: {result_outpaint}")


if __name__ == "__main__":
    import os
    if not os.getenv("FAL_KEY"):
        print("Error: FAL_KEY environment variable not set.")
        print("Please set your FAL API key, e.g., export FAL_KEY='your-key'")
    else:
        asyncio.run(main())

```

---

## Technical Details

This section provides a deeper dive into the architecture, workflow, and contribution guidelines for `twat-genai`.

### Architecture Overview

`twat-genai` is designed with a layered architecture to promote modularity and extensibility:

1.  **Core (`src/twat_genai/core`)**:
    *   **Purpose:** Provides fundamental building blocks, data structures, and utilities that are independent of any specific AI generation engine.
    *   **Key Modules:**
        *   `config.py`: Defines core Pydantic models like `ImageInput` (for representing input images via URL, path, or PIL object), `ImageResult` (standardized output structure), `ImageSizeWH`, and various type aliases.
        *   `image.py`: Contains `ImageSizes` (enum for presets like SQ, HD), `ImageFormats` (enum for JPG, PNG), and image saving utilities.
        *   `image_utils.py`: Offers asynchronous utilities for image downloading (`download_image_to_temp`, `download_image`), resizing based on model constraints (`resize_image_if_needed`), and mask creation for outpainting/inpainting (`create_outpaint_mask`, `create_flux_inpainting_assets`, `create_genfill_border_mask`).
        *   `lora.py`: Defines Pydantic models for LoRA configurations (`LoraRecord`, `LoraLib`, `LoraSpecEntry`).
        *   `prompt.py`: Implements parsing and normalization for Midjourney-style prompts, including permutation expansion (`{alt1,alt2}`), multi-prompts (`::`), and parameter handling.

2.  **Engines (`src/twat_genai/engines`)**:
    *   **Purpose:** Implements the logic for interacting with specific AI platforms or model families.
    *   **`base.py`**:
        *   Defines the `ImageGenerationEngine` abstract base class (ABC), which specifies the common interface (`initialize`, `generate`, `shutdown`) for all engines.
        *   Defines the base `EngineConfig` Pydantic model (common parameters like `guidance_scale`, `num_inference_steps`, `image_size`).
    *   **FAL Engine (`src/twat_genai/engines/fal`)**:
        *   **Purpose:** Concrete implementation for the [fal.ai](https://fal.ai/) platform.
        *   `__init__.py` (`FALEngine`):
            *   The main orchestrator for FAL operations. Inherits from `ImageGenerationEngine`.
            *   Initializes `FalApiClient` and checks for `FAL_KEY`.
            *   Handles input image preparation: downloads URLs (via `core.image_utils`), resizes images for upscalers if needed (via `core.image_utils`), and uses `FALImageInput.to_url()` (which calls `FalApiClient.upload_image()`) to get a FAL-usable URL for local/PIL images. Manages temporary file cleanup.
            *   The `generate()` method determines the operation type (TTI, I2I, Upscale, Outpaint, etc.) based on the `model` (a `ModelTypes` enum value) and routes to the appropriate `FalApiClient` method.
            *   Manages GenFill post-processing for Bria outpainting and asset creation for Flux outpainting.
        *   `client.py` (`FalApiClient`):
            *   Responsible for all direct interactions with the FAL API using the `fal-client` library.
            *   Provides `upload_image()` using `fal_client.upload_file_async()`.
            *   Contains specific methods for each operation: `process_tti()`, `process_i2i()`, `process_canny()`, `process_depth()`, `process_upscale()`, `process_outpaint()`, `process_genfill()`.
            *   These methods submit jobs via `_submit_fal_job()` (which uses `fal_client.submit_async()`).
            *   Result retrieval and processing occur in `_get_fal_result()`, which polls using `fal_client.status_async()` and fetches with `fal_client.result_async()`.
            *   Parses results using `_extract_generic_image_info()` to standardize output into an `ImageResult` object.
            *   Downloads the final image using `_download_image_helper()` (which uses `httpx`).
        *   `config.py`: Defines FAL-specific Pydantic models used as schemas for API arguments.
            *   `ModelTypes` (enum): Maps user-friendly model names/operations to specific FAL API endpoint strings.
            *   `ImageToImageConfig`, `UpscaleConfig`, `OutpaintConfig`: Pydantic models for operation-specific parameters, often including an `ImageInput`.
            *   `UPSCALE_TOOL_MAX_INPUT_SIZES`: A dictionary defining maximum input dimensions for various upscaler models.
        *   `lora.py`: Contains FAL-specific LoRA handling.
            *   `get_lora_lib()`: Loads LoRA definitions from JSON (from `TWAT_GENAI_LORA_LIB` env var, `twat-os` managed path, or the bundled `__main___loras.json`).
            *   `parse_lora_phrase()`: Parses individual LoRA strings (library keys or `url:scale` syntax).
            *   `normalize_lora_spec()`: Converts various input LoRA spec formats into a list of `LoraSpecEntry` or `CombinedLoraSpecEntry`.
            *   `build_lora_arguments()`: Asynchronously prepares the final LoRA argument list (e.g., `[{ "path": "url", "scale": 0.7 }]`) and augments the text prompt for the FAL API.
        *   `models.py` (`FALImageInput`):
            *   Subclasses `core.config.ImageInput`.
            *   Its `to_url()` async method converts local file paths or PIL Image objects into FAL-usable URLs by uploading them via `fal_client.upload_file_async()`.

3.  **CLI (`src/twat_genai/cli.py`)**:
    *   **Purpose:** Provides the command-line interface.
    *   Uses the `python-fire` library to expose methods of the `TwatGenAiCLI` class as subcommands.
    *   Parses CLI arguments, maps them to `ModelTypes`, and constructs the necessary configuration objects (`EngineConfig`, `ImageToImageConfig`, etc.).
    *   Handles `input_image` parsing into `core.config.ImageInput`.
    *   Each command method (e.g., `text`, `image`, `upscale`) calls a shared `_run_generation()` async helper, which instantiates and runs the `FALEngine`. `asyncio.run()` is used within the top-level CLI methods that need to call async code.

4.  **Entry Point (`src/twat_genai/__main__.py`)**:
    *   A minimal script that enables the package to be run as a module (`python -m twat_genai`). It imports `TwatGenAiCLI` and uses `fire.Fire(TwatGenAiCLI)`.

5.  **Default LoRA Library (`src/twat_genai/__main___loras.json`)**:
    *   A JSON file containing predefined LoRA shortcuts (keywords mapping to LoRA URLs and default scales).

### Detailed Workflow Example (e.g., Image-to-Image)

Here's a simplified flow of an Image-to-Image request:

1.  **User Invocation:**
    *   **CLI:** `twat-genai image --input_image my_image.jpg --prompts "make it vintage" --strength 0.6`
    *   **API:** `await engine.generate(prompt="make it vintage", model=ModelTypes.IMAGE, image_config=i2i_cfg_obj, ...)`
2.  **Argument Parsing (CLI):** `TwatGenAiCLI` parses arguments. `input_image` becomes an `ImageInput` object. `strength`, `prompts`, etc., are collected.
3.  **Configuration Setup:**
    *   An `EngineConfig` is created with general settings (e.g., `image_size` if specified).
    *   An `ImageToImageConfig` is created, holding the `ImageInput` object and I2I-specific parameters like `strength` and `negative_prompt`.
4.  **Engine Execution:**
    *   `FALEngine` instance is created (if not already, e.g., via `async with`). `initialize()` is called, setting up the `FalApiClient`.
    *   `FALEngine.generate()` is called with the prompt, base `EngineConfig`, `model=ModelTypes.IMAGE`, and the `ImageToImageConfig`.
5.  **Input Preparation (`FALEngine._prepare_image_input`):**
    *   The `ImageInput` from `ImageToImageConfig` is processed.
    *   If it's a local path (e.g., `my_image.jpg`) or a PIL Image, `FALImageInput.to_url()` is effectively called.
    *   `FALImageInput.to_url()` uses `fal_client.upload_file_async()` to upload the image to FAL's temporary storage, returning a URL.
    *   If the input is already a URL, it might be used directly or downloaded/re-uploaded if processing (like resizing for upscalers, though not typical for I2I) is needed. Original image dimensions are determined.
6.  **Prompt & LoRA Processing:**
    *   The input prompt is normalized (e.g., expanding permutations) by `core.prompt.normalize_prompts()`.
    *   If a `lora_spec` is provided, `engines.fal.lora.build_lora_arguments()` parses it, resolves library keys, and prepares a list of LoRA dictionaries for the API, potentially modifying the prompt string to include LoRA trigger words.
7.  **API Client Interaction (`FalApiClient.process_i2i`):**
    *   The `FalApiClient.process_i2i()` method is invoked with the prompt, the FAL-usable image URL, LoRA arguments, and other parameters.
    *   It assembles the final dictionary of arguments for the FAL API endpoint (`fal-ai/flux-lora/image-to-image`).
    *   `_submit_fal_job()` is called, which uses `fal_client.submit_async()` to send the request to FAL. A request ID is returned.
8.  **Result Handling (`FalApiClient._get_fal_result`):**
    *   The client polls FAL for the job status using `fal_client.status_async(request_id)` until completion.
    *   The final result JSON is fetched using `fal_client.result_async(request_id)`.
    *   `_extract_generic_image_info()` parses this JSON to find the URL(s) of the generated image(s) and other metadata like seed, dimensions.
    *   If `output_dir` is specified, `_download_image_helper()` downloads the generated image(s) using `httpx` and saves them.
    *   An `ImageResult` Pydantic model is populated with the request ID, timestamp, raw API result, parsed image information (including local path if saved), original prompt, and job parameters. Metadata is saved to a JSON file alongside the image.
9.  **Return Value:** The `ImageResult` object is returned to the caller (`FALEngine.generate()`, then to the CLI method or API user).
10. **CLI Output:** The CLI prints a summary of the result, including the path to the saved image.

### Key Data Structures (Pydantic Models)

*   **`core.config.ImageInput`**: Represents an input image (URL, local path, or PIL Image).
    *   `engines.fal.models.FALImageInput`: Subclass that handles uploading local/PIL images to FAL.
*   **`core.config.ImageResult`**: Standardized structure for returning generation results, including metadata, image path, and raw API response.
*   **`engines.base.EngineConfig`**: Base configuration for all engines (e.g., `guidance_scale`, `num_inference_steps`, `image_size`).
*   **`engines.fal.config.ModelTypes`**: Enum mapping operation types to FAL API endpoint strings (e.g., `TEXT` -> `"fal-ai/flux-lora"`).
*   **`engines.fal.config.ImageToImageConfig`**: Parameters for I2I, Canny, Depth (e.g., `input_image`, `strength`).
*   **`engines.fal.config.UpscaleConfig`**: Parameters for various upscaling tools (e.g., `input_image`, `scale`, tool-specific options like `esrgan_model`).
*   **`engines.fal.config.OutpaintConfig`**: Parameters for outpainting (e.g., `input_image`, `prompt`, `target_width`, `target_height`, `outpaint_tool`).
*   **`core.lora.LoraRecord`, `LoraLib`, `LoraSpecEntry`, `CombinedLoraSpecEntry`**: Define how LoRAs are represented, stored in a library, and specified for use.

### Extensibility

*   **Adding a New AI Engine:**
    1.  Create a new module under `src/twat_genai/engines/`.
    2.  Implement a class that inherits from `ImageGenerationEngine` (in `src/twat_genai/engines/base.py`).
    3.  Implement the abstract methods: `initialize()`, `generate()`, and `shutdown()`.
    4.  Define any engine-specific configuration models (similar to `UpscaleConfig` for FAL).
    5.  Update the CLI and potentially the main API entry points to allow selection and use of the new engine.
*   **Adding New Models/Operations to the FAL Engine:**
    1.  Add a new entry to the `ModelTypes` enum in `src/twat_genai/engines/fal/config.py` with the FAL endpoint string.
    2.  If the new operation requires unique parameters, create a new Pydantic config model (e.g., `NewOperationConfig`) in `engines/fal/config.py`.
    3.  Add a corresponding processing method in `FalApiClient` (e.g., `process_new_operation()`). This method will handle argument assembly and job submission.
    4.  Update `FALEngine.generate()` to handle the new `ModelTypes` value, call the new `FalApiClient` method, and pass the appropriate configuration.
    5.  Add a new subcommand to `TwatGenAiCLI` in `src/twat_genai/cli.py` for the new operation.

### Coding and Contribution Rules

We welcome contributions! Please follow these guidelines:

*   **Code Quality Tools:**
    *   **Linting & Formatting:** We use [Ruff](https://beta.ruff.rs/docs/). Run `uv run lint` (which executes `ruff check .` and `ruff format .`).
    *   **Type Checking:** We use [MyPy](http://mypy-lang.org/). Run `uv run type-check` (which executes `mypy src/twat_genai tests`).
*   **Pre-commit Hooks:**
    *   The project includes a `.pre-commit-config.yaml`.
    *   Install hooks with `pre-commit install`. This will automatically run checks (like Ruff and MyPy) before each commit.
*   **Dependency Management:**
    *   Dependencies are managed using `uv` and specified in `pyproject.toml`.
    *   Install development dependencies with `uv pip install -e ".[all]"`.
*   **Versioning:**
    *   The project version is dynamically determined from `git` tags using `hatch-vcs`.
    *   Releases are made by creating a new `git` tag (e.g., `v0.2.0`).
*   **Branching Strategy:**
    *   Develop features in separate branches (e.g., `feature/my-new-feature`, `fix/bug-fix`).
    *   Submit Pull Requests (PRs) to the `main` branch for review.
*   **Commit Messages:**
    *   Please follow [Conventional Commits](https://www.conventionalcommits.org/) guidelines.
    *   Examples: `feat: Add support for XYZ model`, `fix: Correct parameter handling in I2I`, `docs: Update README with API examples`.
*   **Testing:**
    *   Tests are written using `pytest` and are located in the `tests/` directory.
    *   Run tests with `uv run test`.
    *   All new features and bug fixes should be accompanied by corresponding tests.
    *   Ensure good test coverage. Check coverage with `uv run test-cov`.
*   **Documentation:**
    *   Keep this `README.md` file updated with any changes to functionality, API, or CLI.
    *   Write clear and concise docstrings for all public modules, classes, and functions.
    *   Use type hints extensively.
*   **Python Version:** The project targets Python 3.10 and above.

By following these guidelines, you help maintain the quality and consistency of the `twat-genai` codebase.
