"""Data model for the preprocessing pipeline."""

from __future__ import annotations

import dataclasses
from enum import Enum

import pandas as pd


class FeatureModality(str, Enum):
    """The modality of a feature.

    Here we move between the way the data is stored, and what it actually
    represents. For instance, a numerical dtype could represent numerical
    and categorical features, while a string could represent categorical
    or text features.
    """

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    CONSTANT = "constant"


@dataclasses.dataclass(frozen=True)
class DatasetView:
    """A view of a dataset split by feature modalities."""

    X: pd.DataFrame
    columns_by_modality: dict[FeatureModality, list[str]]

    @property
    def x_num(self) -> pd.DataFrame:
        """Returns the numerical features as a pd.DataFrame."""
        return self._get_modality(FeatureModality.NUMERICAL)

    @property
    def x_cat(self) -> pd.DataFrame:
        """Returns the categorical features as a pd.DataFrame."""
        return self._get_modality(FeatureModality.CATEGORICAL)

    @property
    def x_num_and_cat(self) -> pd.DataFrame:
        """Returns the numerical and categorical features as a pd.DataFrame."""
        return pd.concat([self.x_num, self.x_cat], axis=1)

    @property
    def x_txt(self) -> pd.DataFrame:
        """Returns the text features as a pd.DataFrame."""
        return self._get_modality(FeatureModality.TEXT)

    def _get_modality(self, modality: FeatureModality) -> pd.DataFrame:
        return self.X.loc[:, self.columns_by_modality[modality]]
