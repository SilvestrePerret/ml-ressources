# standard imports
from typing import Optional

# external imports
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class Resampler(BaseEstimator):
    """
    WARNING: This should be a sklearn transformer if sklearn allowed transformers to change both X and y.
    >.<
    """

    def __init__(
        self,
        model: BaseEstimator,
        max_samples_per_class: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.model = model
        self.max_samples_per_class = max_samples_per_class
        self.random_state = random_state

    @property
    def classes_(self):
        if hasattr(self, "model_"):
            return self.model_.classes_
        else:
            raise NotFittedError

    def fit(self, X: pd.DataFrame, y: pd.Series):
        data = X.copy()
        data[y.name] = y
        data = self.subsample(data, y.name)
        self.model_ = self.model.fit(data.drop(columns=y.name), data[y.name])
        return self

    def predict(self, *args, **kwargs):
        return self.model_.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return self.model_.predict_proba(*args, **kwargs)

    def predict_with_confidence(self, *args, **kwargs):
        return self.model_.predict_with_confidence(*args, **kwargs)

    def subsample(self, df: pd.DataFrame, output_col: str) -> pd.DataFrame:
        if self.max_samples_per_class is None:
            return df
        else:
            return (
                df.groupby(output_col)
                .apply(
                    lambda df: df.sample(
                        n=min(self.max_samples_per_class, len(df)), random_state=self.random_state
                    )
                )
                .reset_index(0, drop=True)
            )
