"""Unit tests for `tabpfn.telemetry`."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import jwt
import pytest
from tabpfn_common_utils.telemetry.core.events import DatasetEvent, PingEvent

import tabpfn.telemetry as telemetry_module


def _product_telemetry_factory(mock_service: MagicMock) -> Callable[..., MagicMock]:
    def _factory(*_args: object, **_kwargs: object) -> MagicMock:
        return mock_service

    return _factory


@pytest.fixture
def mock_config() -> dict[str, str]:
    return {"project_token": "test-project-token", "api_host": "https://example.com"}


class TestGetTelemetryConfig:
    def test_returns_none_when_download_config_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(telemetry_module, "download_config", lambda: None)
        assert telemetry_module._get_telemetry_config() is None

    def test_returns_none_when_project_token_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "tabpfn.telemetry.download_config",
            lambda: {"project_token": None, "api_host": "https://h.example"},
        )
        assert telemetry_module._get_telemetry_config() is None

    def test_returns_none_when_api_host_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "tabpfn.telemetry.download_config",
            lambda: {"project_token": "tok", "api_host": None},
        )
        assert telemetry_module._get_telemetry_config() is None

    def test_returns_config_when_complete(
        self, monkeypatch: pytest.MonkeyPatch, mock_config: dict[str, str]
    ) -> None:
        monkeypatch.setattr("tabpfn.telemetry.download_config", lambda: mock_config)
        cfg = telemetry_module._get_telemetry_config()
        assert cfg is not None
        assert cfg.project_token == mock_config["project_token"]
        assert cfg.api_host == mock_config["api_host"]


class TestGetUserId:
    def test_returns_none_when_no_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("tabpfn.telemetry.get_cached_auth_token", lambda: None)
        assert telemetry_module._get_user_id() is None

    def test_returns_none_when_token_invalid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "tabpfn.telemetry.get_cached_auth_token", lambda: "not-a-jwt"
        )
        assert telemetry_module._get_user_id() is None

    def test_returns_user_claim_when_token_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = {"user": "user-123", "sub": "ignored"}
        token = jwt.encode(
            payload,
            "x" * 32,
            algorithm="HS256",
        )
        monkeypatch.setattr("tabpfn.telemetry.get_cached_auth_token", lambda: token)
        assert telemetry_module._get_user_id() == "user-123"


class TestPatchClient:
    def test_clears_posthog_when_no_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(telemetry_module, "_get_telemetry_config", lambda: None)
        instance = MagicMock()
        monkeypatch.setattr(
            telemetry_module.telemetry_service,
            "ProductTelemetry",
            lambda: instance,
        )
        telemetry_module._patch_client()
        assert instance._posthog_client is None

    def test_sets_posthog_when_config_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_config: dict[str, str],
    ) -> None:
        monkeypatch.setattr(
            telemetry_module,
            "_get_telemetry_config",
            lambda: telemetry_module._TelemetryConfig(
                project_token=mock_config["project_token"],
                api_host=mock_config["api_host"],
            ),
        )
        instance = MagicMock()
        instance.HOST = "https://eu.i.posthog.com"
        monkeypatch.setattr(
            telemetry_module.telemetry_service,
            "ProductTelemetry",
            lambda: instance,
        )
        fake_posthog = MagicMock()
        monkeypatch.setattr(telemetry_module, "Posthog", fake_posthog)
        telemetry_module._patch_client()
        fake_posthog.assert_called_once()
        call_kw = fake_posthog.call_args.kwargs
        assert call_kw["project_api_key"] == mock_config["project_token"]
        assert instance._posthog_client == fake_posthog.return_value


class TestCaptureEventWithUserId:
    def test_returns_early_when_no_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(telemetry_module, "_get_telemetry_config", lambda: None)
        mock_pt = MagicMock()
        monkeypatch.setattr(telemetry_module, "ProductTelemetry", mock_pt)
        telemetry_module._capture_event_with_user_id(PingEvent())
        mock_pt.assert_not_called()

    def test_skips_non_passthrough_when_no_user(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_config: dict[str, str],
    ) -> None:
        monkeypatch.setattr(
            telemetry_module,
            "_get_telemetry_config",
            lambda: telemetry_module._TelemetryConfig(
                project_token=mock_config["project_token"],
                api_host=mock_config["api_host"],
            ),
        )
        monkeypatch.setattr(telemetry_module, "_get_user_id", lambda: None)
        mock_pt = MagicMock()
        monkeypatch.setattr(telemetry_module, "ProductTelemetry", mock_pt)
        telemetry_module._capture_event_with_user_id(
            DatasetEvent(task="classification", role="train")
        )
        mock_pt.assert_not_called()

    def test_allows_ping_without_user(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_config: dict[str, str],
    ) -> None:
        monkeypatch.setattr(
            telemetry_module,
            "_get_telemetry_config",
            lambda: telemetry_module._TelemetryConfig(
                project_token=mock_config["project_token"],
                api_host=mock_config["api_host"],
            ),
        )
        monkeypatch.setattr(telemetry_module, "_get_user_id", lambda: None)
        mock_service = MagicMock()
        monkeypatch.setattr(
            telemetry_module,
            "ProductTelemetry",
            _product_telemetry_factory(mock_service),
        )
        telemetry_module._capture_event_with_user_id(PingEvent())
        mock_service.capture.assert_called_once()
        call_kw = mock_service.capture.call_args.kwargs
        assert "distinct_id" not in call_kw

    def test_passes_distinct_id_when_user_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_config: dict[str, str],
    ) -> None:
        monkeypatch.setattr(
            telemetry_module,
            "_get_telemetry_config",
            lambda: telemetry_module._TelemetryConfig(
                project_token=mock_config["project_token"],
                api_host=mock_config["api_host"],
            ),
        )
        monkeypatch.setattr(telemetry_module, "_get_user_id", lambda: "uid-42")
        mock_service = MagicMock()
        monkeypatch.setattr(
            telemetry_module,
            "ProductTelemetry",
            _product_telemetry_factory(mock_service),
        )
        telemetry_module._capture_event_with_user_id(PingEvent())
        mock_service.capture.assert_called_once()
        assert mock_service.capture.call_args.kwargs["distinct_id"] == "uid-42"


class TestInit:
    def test_calls_patch_ping_and_capture_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def mark(name: str) -> None:
            calls.append(name)

        monkeypatch.setattr(
            telemetry_module,
            "_patch_client",
            lambda: mark("_patch_client"),
        )
        monkeypatch.setattr(telemetry_module, "ping", lambda: mark("ping"))
        monkeypatch.setattr(
            telemetry_module,
            "capture_session",
            lambda: mark("capture_session"),
        )
        telemetry_module.init()
        assert calls == ["_patch_client", "ping", "capture_session"]
