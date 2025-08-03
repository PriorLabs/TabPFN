from __future__ import annotations


def test__packages_can_still_be_imported_from_old_location() -> None:
    """Test modules in the base architecture can still be imported from old location.

    We moved the packages from tabpfn.model to tabpfn.architectures.base.
    """
    import tabpfn.model.attention
    import tabpfn.model.bar_distribution
    import tabpfn.model.config
    import tabpfn.model.encoders
    import tabpfn.model.layer
    import tabpfn.model.loading
    import tabpfn.model.memory
    import tabpfn.model.mlp
    import tabpfn.model.preprocessing
    import tabpfn.model.transformer

    assert hasattr(tabpfn.model.attention, "Attention")
    assert hasattr(tabpfn.model.bar_distribution, "BarDistribution")
    assert hasattr(tabpfn.model.config, "ModelConfig")
    assert hasattr(tabpfn.model.encoders, "InputEncoder")
    assert hasattr(tabpfn.model.layer, "LayerNorm")
    assert hasattr(tabpfn.model.memory, "MemoryUsageEstimator")
    assert hasattr(tabpfn.model.mlp, "MLP")
    assert hasattr(tabpfn.model.preprocessing, "SequentialFeatureTransformer")
    assert hasattr(tabpfn.model.transformer, "PerFeatureTransformer")
