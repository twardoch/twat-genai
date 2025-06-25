1.  **Project Setup & Initial Cleanup:**
    *   Verify `npc-engine` installation or proceed with manual analysis if installation continues to fail. (Already attempted, will proceed manually)
    *   Create `PLAN.md`, `TODO.md`, and `CHANGELOG.md`.
2.  **Address Code Duplication and Unused Code:**
    *   **`core/models.py`**: Investigate `src/twat_genai/core/models.py`. If it's entirely redundant with `core/config.py` and `core/image.py` (as suggested in `README.md`), remove it. Update any imports if necessary.
    *   **`engines/fal/config.py`**: Remove the duplicate `FalApiClient` definition from `src/twat_genai/engines/fal/config.py`. Ensure all usages point to the correct client in `engines/fal/client.py`.
    *   **`ImageSizeWH`**: Consolidate the `ImageSizeWH` definition. Prefer keeping it in `core/config.py` as it's a configuration-related model. Update `core/image.py` to import it from `core/config.py`.
    *   **Unused `run_upscale` and `run_outpaint`**: Examine `run_upscale` in `src/twat_genai/engines/fal/upscale.py` and `run_outpaint` in `src/twat_genai/engines/fal/outpaint.py`. If they are truly unused and their logic is incorporated elsewhere (likely in `FALEngine` or `FalApiClient`), remove them.
3.  **Streamline Configuration and Core Logic:**
    *   **`ImageInput.to_url`**: Confirm that `FALImageInput.to_url` in `src/twat_genai/engines/fal/models.py` is the sole and correct implementation. The base method in `core/config.py` is designed to be overridden, which is good. No change needed here unless further review shows issues.
    *   **Output Directory Logic (`cli.py`)**:
        *   Review the `get_output_dir` function in `src/twat_genai/cli.py`.
        *   For an MVP, consider simplifying the logic by removing the `twat.PathManager` dependency or making its usage more straightforward. Prioritize a simple default (e.g., `./generated_images`) and user-provided paths. The input image-based subfolder is a good feature to keep.
    *   **CLI Async Structure (`cli.py`)**: Evaluate if refactoring the `TwatGenAiCLI` methods to call a central async orchestrator (instead of `asyncio.run()` in each method) simplifies the code or improves clarity. This is a minor point and might be deferred.
4.  **Review and Refine CLI Arguments and Commands:**
    *   **Upscale Tool Parameters (`cli.py`)**: The `upscale` command in `cli.py` has many specific arguments for different tools (e.g., `ideogram_detail`, `esrgan_model`). While this provides fine-grained control, assess if a more generic approach (e.g., passing a dict of tool-specific params) could simplify the CLI signature for an MVP, or if the current explicitness is preferred. For now, assume current explicitness is fine but keep in mind for future versions.
    *   **Default Prompts for Upscalers (`cli.py`)**: The CLI's `_run_generation` method adds an empty string prompt `[""]` if no prompts are provided for upscalers. Confirm this is the desired behavior for all upscale models supported by FAL.
5.  **Documentation and Helper Scripts:**
    *   **`README.md`**: Update `README.md` to reflect any changes made, especially regarding removed files/code and simplified logic.
    *   **`cleanup.py`**: Decide on the role of `cleanup.py`. If CI/CD handles most of its functions (linting, testing, building), consider removing it or simplifying it to only cover tasks not automated elsewhere. For an MVP, it might be safe to remove if it's primarily a developer convenience for local workflows that are replicated in CI.
    *   **LoRA Configuration (`__main___loras.json`)**: Review the LoRA loading mechanism. The current approach using `__main___loras.json` and `TWAT_GENAI_LORA_LIB` environment variable is flexible. For an MVP, ensure this is clearly documented and robust. No immediate change planned unless issues are found.
6.  **Testing:**
    *   Ensure existing tests pass after changes.
    *   Add new tests or modify existing ones if significant logic changes occur (e.g., output directory handling).
7.  **Final Review and Submission:**
    *   Perform a final review of all changes.
    *   Update `PLAN.md` and `TODO.md` to reflect completed tasks.
    *   Update `CHANGELOG.md` with a summary of changes.
    *   Submit the changes.
