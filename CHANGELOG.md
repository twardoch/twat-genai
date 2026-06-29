---
this_file: CHANGELOG.md
---

# Changelog

All notable changes to the twat-genai project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `STYLE_GUIDE.md`: developer and user reference explaining all providers (FAL, Chutes, OpenAI, Gemini), operations, LoRA syntax, prompt syntax, and code conventions for non-AI-experts and contributors.
- `.github/workflows/ci.yml`: GitHub Actions CI running pytest across Python 3.10–3.12 plus ruff lint/format checks.

### Fixed

- `cli.py` `parse_image_size`: detect `NxM` format (e.g. `100x200`) and raise a targeted `"For custom image sizes use 'width,height'…"` error instead of the generic unknown-preset message.
- `engines/fal/client.py` outpaint error message: `{e!r}` → `{e}` so the message contains the readable error text rather than the `repr()` form.
- `engines/fal/client.py` `_process_generic`: moved `build_lora_arguments` call inside the `try` block so LoRA resolution errors are wrapped in `RuntimeError("Processing {model} job failed: …")`.
- Tests (`tests/test_fal_client.py`): fixed disconnected `mock_submit_job` and `mock_build_lora_arguments` fixtures that caused patches to be silently ignored; corrected stale assertions for TTI, I2I, and outpaint job parameters; fixed `num_outputs` → `num_images` key in outpaint assertions.
- Tests (`tests/test_cli.py`): made `test_cli_text_basic` synchronous (avoids nesting `asyncio.run()` inside pytest-asyncio's event loop); rewrote `test_cli_text_with_options` to use `TwatGenAiCLI` class directly.
- Tests (`tests/test_image_utils.py`): replaced plain `async def` mock functions with `AsyncMock`; fixed undefined variable references in download tests.

### Changed

- Full test suite green: **70 tests pass**, 0 failures, 0 errors (all network calls mocked for offline execution).

## [v1.7.5] - 2025-02-15

### Changed

- Improved error message formatting across the codebase
- Updated type hints to use modern Python syntax (e.g., `list[str]` instead of `List[str]`)
- Simplified union type hints using `|` operator (e.g., `str | None` instead of `Optional[str]`)

### Fixed

- Fixed circular imports in FAL engine modules
- Improved error handling and messaging in LoRA processing

## [v1.7.3] - 2025-02-15

### Added

- New FAL-specific image input handling
- Added `FALImageInput` class for better FAL API integration

### Changed

- Refactored image input processing for better type safety
- Updated dependency requirements for better compatibility

## [v1.6.2] - 2025-02-06

### Changed

- Updated dependency versions:
  - `twat>=1.0.0`
  - `twat-image>=1.0.0`

### Fixed

- Package dependency issues

## [v1.6.1] - 2025-02-06

### Changed

- Reorganized module exports in `__init__.py`
- Improved code organization and imports
- Enhanced type annotations throughout the codebase

### Fixed

- Various import and circular dependency issues
- Code style and formatting improvements

## [v1.6.0] - 2025-02-06

### Added

- Initial public release with core functionality
- Support for text-to-image generation
- Support for image-to-image transformation
- LoRA integration with FAL.ai
- Command-line interface
- Python API
- Configuration management
- Image processing utilities

### Features

- Multiple model support through FAL.ai
- Flexible prompt expansion system
- LoRA configuration management
- Image size presets and custom sizes
- Output directory management
- File naming conventions
- Environment variable configuration

## [v1.0.0] - 2025-02-06

### Added

- Initial project structure
- Basic package setup
- Core dependencies
- Development environment configuration

[v1.7.5]: https://github.com/twardoch/twat-genai/compare/v1.7.3...v1.7.5
[v1.7.3]: https://github.com/twardoch/twat-genai/compare/v1.6.2...v1.7.3
[v1.6.2]: https://github.com/twardoch/twat-genai/compare/v1.6.1...v1.6.2
[v1.6.1]: https://github.com/twardoch/twat-genai/compare/v1.6.0...v1.6.1
[v1.6.0]: https://github.com/twardoch/twat-genai/compare/v1.0.0...v1.6.0
[v1.0.0]: https://github.com/twardoch/twat-genai/releases/tag/v1.0.0 
