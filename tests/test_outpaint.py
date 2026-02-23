#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pydantic", "Pillow", "loguru", "fire"]
# ///
"""
Test script for twat_genai outpaint command.
This script can be used to verify the fixed outpaint functionality.
"""

import sys
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="DEBUG")


def test_outpaint(input_image: str | None = None):
    """
    Test the outpaint functionality without running the actual command.

    Args:
        input_image: Path to an image file (optional)
    """
    from twat_genai.cli import TwatGenAiCLI

    logger.info(f"Testing outpaint with input_image: {input_image}")

    cli = TwatGenAiCLI(verbose=True)

    if input_image:
        try:
            # Test the _prepare_image_input function directly
            input_img = cli._prepare_image_input(input_image)
            logger.info(f"Successfully prepared input image: {input_img}")

            # Check the path exists
            if input_img and input_img.path:
                logger.info(f"Path exists: {input_img.path.exists()}")

            return "Success: Image input preparation is working correctly!"
        except Exception as e:
            logger.error(f"Error preparing image: {e}")
            return f"Error: {e}"
    else:
        logger.warning("No input_image provided. Checking directory creation only.")
        # Just check if the image directory is created properly
        from twat_os.paths import PathManager

        paths = PathManager.for_package("twat_genai")
        logger.info(f"twat_genai paths: {paths}")
        logger.info(f"GenAI output directory: {paths.genai.output_dir}")
        return f"Output directory would be: {paths.genai.output_dir}"


if __name__ == "__main__":
    import fire

    fire.Fire(test_outpaint)
