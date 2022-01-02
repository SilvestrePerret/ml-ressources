# standard imports
from typing import Union

# external imports
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfSearchEngine(BaseEstimator):
    def __init__(self, min_df: Union[float, int] = 0.0, max_df: Union[float, int] = 1.0):
        self.min_df = min_df
        self.max_df = max_df

    def fit(self, raw_documents: pd.DataFrame, y: pd.Series):
        self.tfidf_ = TfidfVectorizer(min_df=self.min_df, max_df=self.max_df, lowercase=False).fit(
            raw_documents.iloc[:, 0]
        )
        self.vectors_ = self._tfidf_transform(raw_documents).groupby(np.asarray(y)).mean()
        self.classes_ = self.vectors_.index
        return self

    def predict_proba(self, raw_documents: pd.DataFrame) -> np.ndarray:
        return cosine_similarity(self._tfidf_transform(raw_documents), self.vectors_)

    def predict_top_matches(self, raw_documents: pd.DataFrame, n_top: int) -> np.ndarray:
        proba = self.predict_proba(raw_documents)
        top_pred_classes_idx = np.argpartition(proba, -n_top, axis=1)[:, -n_top:]
        return self.classes_[top_pred_classes_idx]

    def predict(self, raw_documents: pd.DataFrame) -> np.ndarray:
        return self.predict_top_matches(raw_documents, n_top=1)

    def _tfidf_transform(self, raw_documents: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.tfidf_.transform(raw_documents.iloc[:, 0]).todense(),
            columns=self.tfidf_.get_feature_names(),
            index=raw_documents.index,
        ).fillna(0.0)