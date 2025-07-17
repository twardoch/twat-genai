"""Test version handling and semver compliance."""

import re
from twat_genai import __version__


def test_version_exists():
    """Test that version is available."""
    assert __version__ is not None
    assert isinstance(__version__, str)


def test_version_format():
    """Test that version follows semver format."""
    # Allow for development versions like 1.0.0.dev0+hash
    semver_pattern = r'^(\d+)\.(\d+)\.(\d+)([a-zA-Z0-9\.\+\-]*)?$'
    assert re.match(semver_pattern, __version__), f"Version {__version__} does not match semver pattern"


def test_version_components():
    """Test version components are valid."""
    # Split version to get basic components
    base_version = __version__.split('+')[0].split('.dev')[0]
    parts = base_version.split('.')
    
    # Should have at least 3 parts (major.minor.patch)
    assert len(parts) >= 3, f"Version {__version__} should have at least 3 components"
    
    # First three parts should be numeric
    for i, part in enumerate(parts[:3]):
        assert part.isdigit(), f"Version component {i} ({part}) should be numeric"


def test_version_not_placeholder():
    """Test that version is not a placeholder."""
    assert __version__ != "0.0.0"
    assert __version__ != "unknown"
    assert __version__ != ""