#  Copyright (c) Prior Labs GmbH 2026.

"""Keep SKILL.md, the usage guide for coding agents, in sync with the code.

SKILL.md names constructor arguments, `inference_config` keys, environment
variables and repository files. These tests fail when one of them is renamed or
removed, so the guide is updated in the same pull request.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path

import pytest

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.inference_config import InferenceConfig
from tabpfn.settings import PytorchSettings, TabPFNSettings

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL = (REPO_ROOT / "SKILL.md").read_text(encoding="utf-8")


def _constructor_args() -> set[str]:
    return {
        name
        for estimator in (TabPFNClassifier, TabPFNRegressor)
        for name in inspect.signature(estimator.__init__).parameters
    }


def test__skill__has_frontmatter_with_name_and_description() -> None:
    match = re.match(r"^---\n(.*?)\n---\n", SKILL, re.DOTALL)
    assert match, "SKILL.md must start with a YAML frontmatter block"
    assert re.search(r"^name: \S+", match.group(1), re.MULTILINE)
    assert re.search(r"^description: \S+", match.group(1), re.MULTILINE)


def test__skill__keyword_arguments_exist() -> None:
    # Every `name=` written in SKILL.md must be an estimator constructor
    # argument or a `predict` argument.
    named = set(re.findall(r"\b([a-z_]+)=(?!=)", SKILL))
    predict_args = {
        name
        for estimator in (TabPFNClassifier, TabPFNRegressor)
        for name in inspect.signature(estimator.predict).parameters
    }
    unknown = named - _constructor_args() - predict_args
    assert named, "expected keyword arguments in SKILL.md"
    assert not unknown, f"SKILL.md names unknown keyword arguments: {unknown}"


def test__skill__inference_config_keys_exist() -> None:
    keys = set(re.findall(r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b", SKILL))
    env_vars = {k for k in keys if k.startswith(("TABPFN_", "PYTORCH_"))}
    config_fields = {f.name for f in dataclasses.fields(InferenceConfig)}
    unknown = keys - env_vars - config_fields
    assert not unknown, f"SKILL.md names unknown inference_config keys: {unknown}"


def test__skill__environment_variables_exist() -> None:
    names = set(re.findall(r"\b((?:TABPFN|PYTORCH)_[A-Z0-9_]+)\b", SKILL))
    settings_vars = {f"TABPFN_{name.upper()}" for name in TabPFNSettings.model_fields}
    settings_vars |= {name.upper() for name in PytorchSettings.model_fields}
    source = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (REPO_ROOT / "src" / "tabpfn").rglob("*.py")
    )
    unknown = {n for n in names if n not in settings_vars and n not in source}
    assert not unknown, f"SKILL.md names unknown environment variables: {unknown}"


@pytest.mark.parametrize(
    "link", sorted(set(re.findall(r"\]\(((?!https?://)[^)#]+)\)", SKILL)))
)
def test__skill__relative_links_exist(link: str) -> None:
    assert (REPO_ROOT / link).exists(), f"SKILL.md links to missing path {link}"


def test__skill__predict_output_types_exist() -> None:
    from tabpfn.regressor import OutputType  # noqa: PLC0415

    allowed = set(OutputType.__args__)
    section = SKILL.split("output_type=...")[1].split("\n- ")[0]
    named = set(re.findall(r'`"([a-z]+)"`', section))
    assert named, "expected output types listed after output_type=..."
    assert named <= allowed, f"unknown output types: {named - allowed}"


def test__skill__regressor_presets_exist() -> None:
    from tabpfn.regressor import REGRESSION_PRESETS  # noqa: PLC0415

    named = set(re.findall(r'presets="([a-z_]+)"', SKILL))
    assert named <= set(REGRESSION_PRESETS), named - set(REGRESSION_PRESETS)
