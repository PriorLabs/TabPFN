from __future__ import annotations


def test__packages_can_still_be_imported_from_old_location() -> None:
    """Test modules in the base architecture can still be imported from old location.

    We moved the packages from tabpfn.model to tabpfn.architectures.base.
    """
    import tabpfn.model.attention  # noqa: PLC0415
    import tabpfn.model.bar_distribution  # noqa: PLC0415
    import tabpfn.model.config  # noqa: PLC0415
    import tabpfn.model.encoders  # noqa: PLC0415
    import tabpfn.model.layer  # noqa: PLC0415
    import tabpfn.model.loading  # noqa: PLC0415
    import tabpfn.model.memory  # noqa: PLC0415
    import tabpfn.model.mlp  # noqa: PLC0415
    import tabpfn.model.preprocessing  # noqa: PLC0415
    import tabpfn.model.transformer  # noqa: PLC0415

    assert hasattr(tabpfn.model.attention, "Attention")
    assert hasattr(tabpfn.model.bar_distribution, "BarDistribution")
    assert hasattr(tabpfn.model.config, "ModelConfig")
    assert hasattr(tabpfn.model.encoders, "InputEncoder")
    assert hasattr(tabpfn.model.layer, "LayerNorm")
    assert hasattr(tabpfn.model.mlp, "MLP")
    assert hasattr(tabpfn.model.preprocessing, "SequentialFeatureTransformer")
    assert hasattr(tabpfn.model.transformer, "PerFeatureTransformer")
