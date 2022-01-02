# standard imports
import re
import time

# external imports
import numpy as np
import pandas as pd

from sklearn import __version__
from sklearn.base import clone, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV

# internal imports
from .evaluator import Evaluator


class SklearnEvaluator(Evaluator):
    """
    SklearnEvaluator instances contains the predictive sklearn model to test on the benchmark.
    
    Attributes:
        name (str): name of the model.
        version (str): sklearn version.
        is_regressor (boolean): True if the model is a regression predictive model, False if the model is a classifier predictive model.
        base_model (sklearn model): must be a predictive model from the scikit-learn module
        best_models (dict): dictionary of models with best performances.
        param_grid (dict): contains a list of values for each base_model parameter to try.
        seeded (boolean, default=True): by default the models are seeded in order to have deterministic benchmark results. We can disable the option by setting it to False.
        
    Example:
    -------
    Constructing a SklearnEvaluator from a sklearn model and a param_grid

    >>> from sklearn.tree import DecisionTreeRegressor
    >>> sklearn_evaluator = SklearnEvaluator(
    ...            DecisionTreeRegressor(),
    ...            {
    ...                # The maximum depth of the tree.
    ...                "max_depth": [2, 4, 8, 20],
    ...                # The minimum number of samples required to split an internal node:
    ...                "min_samples_split": [2],
    ...                # The minimum number of samples in a leaf
    ...                # If float, then minimal proportion of the training set in a leaf
    ...                "min_samples_leaf": [2, 6, 10, 20, 50, 100],
    ...                # The number of features to consider when looking for the best split:
    ...                # If float, fraction of the features to consider.
    ...                # If ‘auto‘ or ‘sqrt‘, max_features = np.sqrt(n_features).
    ...                # If ‘log2‘, max_features = np.log2(n_features).
    ...                # If None, all features are considered at each split.
    ...                "max_features": [None],
    ...                # Max number of leaves
    ...                "max_leaf_nodes": [None],
    ...            },
    ...        )
    >>> ds = Dataset.from_name("synth1_long")
    >>> best_params, validation_scores, total_time = sklearn_evaluator.find_best_hyperparams(ds)
    >>> best_params, validation_scores, total_time
    ({'r2': {'max_depth': 8,
       'max_features': None,
       'max_leaf_nodes': None,
       'min_samples_leaf': 20,
       'min_samples_split': 2},
      'mae': {'max_depth': 8,
       'max_features': None,
       'max_leaf_nodes': None,
       'min_samples_leaf': 20,
       'min_samples_split': 2}},
     {'r2': array([0.5771617441337398, 0.593850035097715, 0.6916585797174823,
             0.5659524323673955], dtype=object),
      'mae': array([0.8031995343872846, 0.722283295958568, 0.7241403498486527,
             0.8368799186570904], dtype=object)},
     1.2461683750152588)
    """

    def __init__(self, base_model, param_grid, seeded=True):
        """
        Main constructor

        Args:
            base_model (sklearn model): must be a predictive model from the scikit-learn module
            param_grid (dict): contains a list of values for each base_model parameter to try.
            seeded (boolean, default=True): by default the models are seeded in order to have deterministic benchmark results. We can disable the option by setting it to False.
        """
        self.base_model = base_model
        self.best_models = {}
        self.seeded = seeded
        # Set the seed param only if the base_model contains a random_state parameter
        if seeded and "random_state" in base_model.get_params().keys():
            self.base_model.set_params(random_state=4321)
        self.param_grid = param_grid

    @property
    def name(self):
        return type(self.base_model).__name__

    @property
    def is_regressor(self):
        return is_regressor(self.base_model)

    def find_best_hyperparams(self, dataset):
        # initialization
        scoring = {metric.name: metric.scorer for metric in dataset.metrics}
        X, y = self.get_X_y(dataset)
        gs = GridSearchCV(
            self.base_model,
            self.param_grid,
            scoring=scoring,
            refit=False,
            n_jobs=None,
            cv=dataset.cv.train,
            return_train_score=False,
        )
        # grid search run
        start_time = time.time()
        gs.fit(X, y)
        total_time = time.time() - start_time
        cv_results = pd.DataFrame(gs.cv_results_)

        # extract relevant infos (best scores, best_params)
        # and save models into self.models
        best_params = {}
        validation_scores = {}

        for metric in dataset.metrics:
            best_params[metric.name] = self.get_best_params(cv_results, metric)
            validation_scores[metric.name] = self.get_best_test_scores(
                cv_results, metric
            )
            model = clone(self.base_model)
            model.set_params(**best_params[metric.name])
            self.best_models[metric.name] = model

        return best_params, validation_scores, total_time

    def evaluate_on_test(self, dataset, return_predictions=False):
        # initialization
        self.check_if_fit()

        test_scores = {m.name: [] for m in dataset.metrics}
        if return_predictions:
            test_predictions = {m.name: [] for m in dataset.metrics}
        mean_fit_time = []

        # cross val (on test)
        for metric in dataset.metrics:
            cv_scores = self.cross_validate(
                self.best_models[metric.name],
                dataset,
                is_test=True,
                return_predictions=return_predictions,
            )

            if return_predictions:
                cv_scores, cv_predictions = cv_scores
                test_predictions[metric.name] = cv_predictions
            test_scores[metric.name] = cv_scores[metric.name]
            mean_fit_time += list(cv_scores["fit_time"])

        mean_fit_time = np.mean(mean_fit_time)
        if return_predictions:
            return test_scores, mean_fit_time, test_predictions
        return test_scores, mean_fit_time

    def cross_validate(self, model, dataset, is_test=False, return_predictions=False):
        """WARNING: this method is not used for the gridsearch."""
        # initialization
        metrics = dataset.metrics
        X, y = self.get_X_y(dataset)
        cv = dataset.cv.test if is_test else dataset.cv.train
        cv_scores = {
            **{m.name: [] for m in metrics},
            **{"fit_time": []},
        }
        if return_predictions:
            cv_predictions = []
        # cross-validation run here
        for idx, (train_index, test_index) in enumerate(cv.split(X)):
            X_train = X.iloc[train_index, :]
            y_train = y.iloc[train_index]
            X_test = X.iloc[test_index, :]
            y_test = y.iloc[test_index]
            scores = self._fit_and_score(
                model, X_train, y_train, X_test, y_test, metrics
            )

            for m in metrics:
                cv_scores[m.name].append(scores[m.name])
            cv_scores["fit_time"].append(scores["fit_time"])

            if return_predictions:
                cv_predictions.append(
                    {k: list(scores[k]) for k in ["y_true", "y_pred"]}
                )

        for m in metrics:
            cv_scores["mean_{}".format(m.name)] = np.mean(cv_scores[m.name])

        if return_predictions:
            return cv_scores, cv_predictions
        return cv_scores

    @staticmethod
    def _fit_and_score(model, X_train, y_train, X_test, y_test, metrics):
        start_time = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_time

        y_pred = model.predict(X_test)
        scores = {m.name: m.scoring_fn(y_test, y_pred) for m in metrics}
        scores["fit_time"] = fit_time
        scores["y_true"] = y_test
        scores["y_pred"] = y_pred
        return scores

    @staticmethod
    def get_X_y(dataset):
        X = dataset.sklearn_data.drop(columns=dataset.output)
        y = dataset.sklearn_data[dataset.output[0]]
        return X, y

    @staticmethod
    def get_best_params(cv_results, metric):
        best_index = np.argmax(cv_results["mean_test_{}".format(metric.name)])
        return cv_results.loc[best_index, "params"]

    @staticmethod
    def get_best_test_scores(cv_results, metric):
        best_index = np.argmax(cv_results["mean_test_{}".format(metric.name)])
        metric_col_pattern = re.compile("^split[0-9]+_test_{}$".format(metric.name))
        metric_cols = filter(metric_col_pattern.match, cv_results.columns)

        scores = cv_results.loc[best_index, metric_cols].values

        if not metric.greater_is_better:
            scores *= -1

        return list(scores)

    def check_if_fit(self):
        if self.best_models == {}:
            raise NotFittedError(
                "This SklearnEvaluator({}) is not fitted yet.".format(self.name)
            )
