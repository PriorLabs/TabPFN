from __future__ import annotations

import pytest
import sklearn.datasets
import torch

from tabpfn import TabPFNClassifier
from tabpfn.utils import infer_device_and_type


def test_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    if getattr(torch.backends, "mps", None):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    device = infer_device_and_type("auto")
    assert device.type == "cuda"


def test_auto_prefers_mps_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if getattr(torch.backends, "mps", None):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    else:
        pytest.skip("MPS backend not available in torch")
    device = infer_device_and_type("auto")
    assert device.type == "mps"


def test_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if getattr(torch.backends, "mps", None):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    device = infer_device_and_type("auto")
    assert device.type == "cpu"


def test_mps_string_returns_device():
    device = infer_device_and_type("mps")
    assert isinstance(device, torch.device)
    assert device.type == "mps"


def test_model_runs_on_mps_if_available():
    if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        pytest.skip("MPS backend not available in torch")

    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X, y = X[:10], y[:10]

    model = TabPFNClassifier(n_estimators=1, device="mps")
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
