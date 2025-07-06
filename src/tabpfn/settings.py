"""Settings module for TabPFN configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import Field, ValidationInfo, field_validator
from pydantic_core import PydanticUndefined
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseSettingsWithFilteredEmptyStrings(BaseSettings):
    """Base settings class that handles empty strings consistently."""

    @field_validator("*", mode="before")
    @classmethod
    def empty_to_default(cls, v: Any, info: ValidationInfo) -> Any:
        """Convert empty strings to default values.

        Only if the field has a default value.
        """
        if (
            isinstance(v, str)
            and not v.strip()
            and info.field_name
            and (field_info := cls.model_fields.get(info.field_name))
            and field_info.default is not PydanticUndefined
        ):
            return field_info.default
        return v


class TabPFNSettings(BaseSettingsWithFilteredEmptyStrings):
    """Configuration settings for TabPFN.

    These settings can be configured via environment variables or a .env file.

    Prefixed by ``TABPFN_`` in environment variables.
    """

    model_config = SettingsConfigDict(env_prefix="TABPFN_", env_file=".env")

    # Model Configuration
    model_cache_dir: Optional[Path] = Field(
        default=None,
        description="Custom directory for caching downloaded TabPFN models. "
        "If not set, uses platform-specific user cache directory.",
    )

    # Performance/Memory Settings
    allow_cpu_large_dataset: bool = Field(
        default=False,
        description="Allow running TabPFN on CPU with large datasets (>1000 samples). "
        "Set to True to override the CPU limitation.",
    )

    pytorch_cuda_alloc_conf: str = Field(
        default="max_split_size_mb:512",
        description="PyTorch CUDA memory allocation configuration. "
        "Used to optimize GPU memory usage.",
    )


class TestingSettings(BaseSettingsWithFilteredEmptyStrings):
    """Testing/Development Settings."""

    force_consistency_tests: bool = Field(
        default=False,
        description="Force consistency tests to run regardless of platform. "
        "Set to True to run tests on non-reference platforms.",
    )

    ci: bool = Field(
        default=False,
        description="Indicates if running in continuous integration environment. "
        "Typically set by CI systems (e.g., GitHub Actions).",
    )


class Settings(BaseSettings):
    """Global settings instance."""

    tabpfn: TabPFNSettings = Field(default_factory=TabPFNSettings)
    testing: TestingSettings = Field(default_factory=TestingSettings)


settings = Settings()
