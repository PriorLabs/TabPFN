"""Tests for the settings module."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from tabpfn.settings import (
    BaseSettingsWithFilteredEmptyStrings,
)


def test_empty_string_converts_to_default_with_string_field() -> None:
    """Test that empty strings convert to default values for string fields."""

    class TestSettings(BaseSettingsWithFilteredEmptyStrings):
        my_string: str = Field(default="default_value")

    # Test with empty string
    settings = TestSettings(my_string="")
    assert settings.my_string == "default_value"

    # Test with whitespace-only string
    settings = TestSettings(my_string="   ")
    assert settings.my_string == "default_value"

    # Test with tab and newline
    settings = TestSettings(my_string="\t\n")
    assert settings.my_string == "default_value"


def test_non_empty_string_preserved() -> None:
    """Test that non-empty strings are not converted."""

    class TestSettings(BaseSettingsWithFilteredEmptyStrings):
        my_string: str = Field(default="default_value")
        my_path: Path | None = Field(default=None)

    # Test with actual values
    settings = TestSettings(my_string="actual_value", my_path="/some/path")
    assert settings.my_string == "actual_value"
    assert settings.my_path == Path("/some/path")


def test_field_without_default_preserves_empty_string() -> None:
    """Test that empty strings for required fields are preserved as-is."""

    class TestSettings(BaseSettingsWithFilteredEmptyStrings):
        required_field: str  # No default value

    # Empty string should be preserved for required fields without defaults
    settings = TestSettings(required_field="")
    assert settings.required_field == ""

    # But whitespace-only strings are also preserved
    settings = TestSettings(required_field="   ")
    assert settings.required_field == "   "


def test_non_string_types_not_affected() -> None:
    """Test that non-string types pass through unchanged."""

    class TestSettings(BaseSettingsWithFilteredEmptyStrings):
        my_int: int = Field(default=42)
        my_bool: bool = Field(default=True)
        my_float: float = Field(default=3.14)

    # Test with various non-string types
    settings = TestSettings(my_int=0, my_bool=False, my_float=0.0)
    assert settings.my_int == 0
    assert settings.my_bool is False
    assert settings.my_float == 0.0
