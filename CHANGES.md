# Changes

## 1. Fixed Issues

### 1.1. Directory Creation with Tilde (~) Character
Fixed issue where running `python -m twat_genai outpaint` would create a literal "~" directory with "Pictures/twat_genai" and "tmp/twat" subdirectories. This occurred because path expansion wasn't correctly handling tilde characters.

**Changes made:**
- Updated paths.toml to use `${HOME}` instead of `~` for paths that were causing issues
- Improved the path expansion logic in `PathConfig.expand_path` method to ensure proper order of operations:
  1. Expand environment variables first
  2. Then expand user home directory (~ character)
  3. Finally convert to absolute path
- Enhanced `format_path` method to follow the same expansion order
- Updated `GenAIConfig` to use proper path expansion and added null checks

### 1.2. Pydantic Model Definition Error in ImageInput
Fixed issue where attempting to use the `outpaint` or `upscale` commands with an image would fail with: 
`ImageInput is not fully defined; you should define Image, then call ImageInput.model_rebuild()`

**Changes made:**
- Modified PIL Image import to be non-conditional, removing the TYPE_CHECKING condition
- Properly imported PIL.Image where needed in the core/config.py file
- Updated the `to_url` method signature in `ImageInput` to include an optional `client` parameter
- Updated the `FALImageInput.to_url` implementation to accept the client parameter
- Refactored the `_prepare_image_input` method in FAL engine to better handle image processing

## 2. Test Scripts
- Added a test script (`test_outpaint.py`) to verify fixes without running the full outpaint command 