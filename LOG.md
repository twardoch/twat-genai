---
this_file: LOG.md
---

# Changelog

All notable changes to the twat-genai project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
