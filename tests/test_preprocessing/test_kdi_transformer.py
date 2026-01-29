from __future__ import annotations

import numpy as np
import torch

from tabpfn.preprocessing.steps import KDITransformerWithNaN


def test_kdi_transformer_with_nan_integration():
    """Tests KDITransformerWithNaN handles NaNs and maintains mask."""
    # Create data with NaNs and a torch tensor to test both features
    X = torch.tensor(
        [[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [np.nan, 8.0, 9.0]],
        dtype=torch.float32,
    )

    transformer = KDITransformerWithNaN(alpha=1.0, output_distribution="normal")

    # Test fit
    transformer.fit_transform(X)
    assert hasattr(transformer, "imputation_values_")

    # Test transform
    Xt = transformer.transform(X)

    # Verify type and shape
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == X.shape

    # Verify NaNs are preserved in the exact same positions
    mask = torch.isnan(X).numpy()
    assert np.all(np.isnan(Xt) == mask)

    # Verify non-NaN values are actual numbers (transformed)
    assert np.all(np.isfinite(Xt[~mask]))
