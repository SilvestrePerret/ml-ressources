# standard imports
from typing import Iterable, List, Tuple
import logging

# external imports
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone


logger = logging.getLogger(__name__)


def _parallel_clone_and_fit(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
    model_idx: int,
    n_models: int,
    verbose: int = 0,
) -> BaseEstimator:
    if verbose > 1:
        logger.info(f"creating specialized agents ({model_idx}/{n_models}).")
    new_model = clone(model)
    if hasattr(new_model, "name"):
        setattr(new_model, "name", name)
    elif hasattr(new_model, "model") and hasattr(new_model.model, "name"):
        setattr(new_model.model, "name", name)
    new_model.fit(X, y)
    return new_model


class HierarchicalModel(BaseEstimator):
    def __init__(
        self,
        global_model: BaseEstimator,
        specialized_model: BaseEstimator,
        X_global: Iterable[str],
        y_global: str,
        n_jobs: int = 1,
        verbose: int = 0,
        use_production_names: bool = False,
    ):
        self.global_model = global_model
        self.specialized_model = specialized_model
        self.X_global = X_global
        self.y_global = y_global
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.use_production_names = use_production_names

    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError

    def _fit_specialized_models(self, groups: List[Tuple[str, pd.DataFrame]], y_col: str):
        models = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_clone_and_fit)(
                self.specialized_model,
                gdf.drop(columns=[y_col, self.y_global]),
                gdf[y_col],
                f"{name}_agent" if self.use_production_names else "",
                idx + 1,
                len(groups),
                self.verbose,
            )
            for idx, (name, gdf) in enumerate(groups)
        )
        self.specialized_models_ = {name: model for (name, _), model in zip(groups, models)}
        self.classes_ = np.sort(
            np.unique(np.concatenate([sm.classes_ for sm in self.specialized_models_.values()]))
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_pred_global = self.global_model_.predict(X[self.X_global])
        predictions = []
        for name, gdf in X.groupby(y_pred_global):
            predictions.append(
                pd.Series(
                    self.specialized_models_[name].predict(gdf.drop(columns=self.y_global)),
                    index=gdf.index,
                )
            )
        return pd.concat(predictions, axis=0).reindex(X.index).values

    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        predictions = self.predict_with_context(X)
        predictions.loc[:, "y_pred"] = predictions["y_pred_spec"]
        predictions.loc[:, "confidence"] = (
            predictions.loc[:, "confidence_global"] * predictions.loc[:, "confidence_spec"]
        )
        return predictions

    def predict_with_context(self, X: pd.DataFrame) -> pd.DataFrame:
        pred_with_conf_global = self.global_model_.predict_with_confidence(X[self.X_global])
        pred_with_conf_spec = []
        for name, gdf in X.groupby(pred_with_conf_global["y_pred"]):
            pred_with_conf_spec.append(
                self.specialized_models_[name].predict_with_confidence(
                    gdf.drop(columns=self.y_global)
                )
            )
        pred_with_conf = pd.concat(
            [
                pred_with_conf_global.add_suffix("_global"),
                pd.concat(pred_with_conf_spec, axis=0).reindex(X.index).add_suffix("_spec"),
            ],
            axis=1,
        )
        return pred_with_conf

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        y_pred_global = self.global_model_.predict(X[self.X_global])
        proba = []
        for name, gdf in X.groupby(y_pred_global):
            proba.append(
                pd.DataFrame(
                    self.specialized_models_[name].predict_proba(gdf.drop(columns=self.y_global)),
                    columns=self.specialized_models_[name].classes_,
                    index=gdf.index,
                )
            )
        return (
            pd.concat(proba, axis=0)
            .fillna(0.0)
            .reindex(index=X.index, columns=self.classes_)
            .values
        )

    def _clone_and_rename(self, model, name):
        new_model = clone(model)
        if self.use_production_names:
            if hasattr(new_model, "name"):
                new_model.name = name
            elif hasattr(new_model, "model") and hasattr(new_model.model, "name"):
                setattr(new_model.model, "name", name)
        return new_model


class HierarchicalModelAlpha(HierarchicalModel):
    def fit(self, X: pd.DataFrame, y: pd.Series):
        df = X.copy()
        df[y.name] = y
        if self.verbose > 1:
            logger.info("creating global agent .")
        self.global_model_ = self._clone_and_rename(self.global_model, "global").fit(
            df[self.X_global], df[self.y_global]
        )
        self._fit_specialized_models(df.groupby(self.y_global), y.name)
        return self


class HierarchicalModelBeta(HierarchicalModel):
    def fit(self, X: pd.DataFrame, y: pd.Series):
        df = X.copy()
        df[y.name] = y
        if self.verbose > 1:
            logger.info("creating global agent .")
        self.global_model_ = self._clone_and_rename(self.global_model, "global_agent").fit(
            df[self.X_global], df[self.y_global]
        )
        y_pred_global = self.global_model_.predict(df[self.X_global])
        self._fit_specialized_models(df.groupby(y_pred_global), y.name)
        return self
