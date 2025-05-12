#!/usr/bin/env -S uv run
# /// script
# dependencies = ["fire", "loguru"]
# ///
# this_file: src/twat_genai/__main__.py

"""
Main entry point for the twat-genai CLI when run as a module (`python -m twat_genai`).
Delegates directly to the main CLI class defined in `cli.py`.
"""

import sys
import fire
from loguru import logger

# Import the actual CLI class from the dedicated module
from twat_genai.cli import TwatGenAiCLI

# Configure logging minimally for the entry point
# The main configuration happens within TwatGenAiCLI.__init__
logger.remove()
logger.add(sys.stderr, level="WARNING")  # Default level


if __name__ == "__main__":
    logger.debug("Running twat-genai via __main__.py entry point")
    fire.Fire(TwatGenAiCLI)
