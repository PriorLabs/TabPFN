"""Settings module for TabPFN configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from tabpfn.constants import ModelVersion


class TabPFNSettings(BaseSettings):
    """Configuration settings for TabPFN.

    These settings can be configured via environment variables or a .env file.

    Prefixed by ``TABPFN_`` in environment variables.
    """

    # Set extra="ignore" so that unknown keys in the .env file, for example, entries for
    # other applications, do not cause validation errors.
    model_config = SettingsConfigDict(
        env_prefix="TABPFN_", env_file=".env", extra="ignore"
    )

    # Model Configuration
    model_cache_dir: Path | None = Field(
        default=None,
        description="Custom directory for caching downloaded TabPFN models. "
        "If not set, uses platform-specific user cache directory.",
    )
    model_version: ModelVersion = Field(
        default=ModelVersion.V2_5,
        description="The version of the TabPFN model to use by default.",
    )

    # Performance/Memory Settings
    allow_cpu_large_dataset: bool = Field(
        default=False,
        description="Allow running TabPFN on CPU with large datasets (>1000 samples). "
        "Set to True to override the CPU limitation.",
    )


class PytorchSettings(BaseSettings):
    """PyTorch settings for TabPFN."""

    pytorch_cuda_alloc_conf: str = Field(
        default="max_split_size_mb:512",
        description="PyTorch CUDA memory allocation configuration. "
        "Used to optimize GPU memory usage.",
    )


class TestingSettings(BaseSettings):
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

    @field_validator("ci", mode="before")
    @classmethod
    def _parse_ci(cls, value: Any) -> bool:
        """Interpret any non-empty environment value as ``True``.

        Some CI providers set the ``CI`` environment variable to a non-boolean
        string (e.g., ``"azure"``).  Treat any non-empty string other than
        common falsy values as ``True`` so importing TabPFN works seamlessly in
        those environments.
        """
        if isinstance(value, str):
            value_lower = value.strip().lower()
            return value_lower not in {"", "0", "false", "no", "off"}
        return bool(value)


class Settings(BaseSettings):
    """Global settings instance."""

    tabpfn: TabPFNSettings = Field(default_factory=TabPFNSettings)
    testing: TestingSettings = Field(default_factory=TestingSettings)
    pytorch: PytorchSettings = Field(default_factory=PytorchSettings)


settings = Settings()
