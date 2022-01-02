# standard imports
from abc import ABC


class Evaluator(ABC):
    """Abstract mother class of all evaluators."""

    def find_best_hyperparams(self, dataset):
        """
        Find best hyperparameters on :param:dataset.
        Returns stats on the fitting process.
        """
        pass

    def evaluate_on_test(self, dataset):
        """
        Score itself on the :param:dataset test part.
        Returns stats on the scoring process.
        """
        pass

    def check_if_fit(self):
        """
        raise NotFittedError if the evaluator is not fitted.
        """
        pass
