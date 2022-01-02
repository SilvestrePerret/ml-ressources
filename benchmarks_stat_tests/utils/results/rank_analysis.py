"""
Method and parts of the code come from https://github.com/hfawaz/cd-diagram.
"""
# standard imports
import warnings

# external imports
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare

# internal imports
from .utils import from_long_format_to_matrix


def _filter_evaluators_according_to_nb_of_datasets(result_long_df):
    """filter out the evaluators which don't have values for every datasets."""
    # count the number of tested datasets per classifier
    nb_dataset_per_eval = result_long_df.groupby(["evaluator"]).size()
    # get the maximum number of tested datasets
    max_nb_dataset = nb_dataset_per_eval.max()
    # get the list of evaluators who have been tested on nb_max_datasets
    evaluators = list(nb_dataset_per_eval[nb_dataset_per_eval == max_nb_dataset].index)
    if len(evaluators) != len(nb_dataset_per_eval):
        ignored_evaluators = list(set(nb_dataset_per_eval.index) - set(evaluators))
        warnings.warn(
            (
                "Some evaluator don't have score for every dataset and therefore will be ignored."
                f"Ignored evaluators: {ignored_evaluators}"
            )
        )
    return result_long_df.loc[result_long_df["evaluator"].isin(evaluators), :]


def _check_if_friedmanchisquare_passed(alpha, results_per_eval):
    """test the null hypothesis using friedman"""
    friedman_p_value = friedmanchisquare(
        *(results_per_eval[k]["value"].values for k in results_per_eval)
    )[1]
    if friedman_p_value >= alpha:
        # then the null hypothesis over the entire classifiers cannot be rejected
        raise ValueError(
            "the null hypothesis over the entire classifiers cannot be rejected {}".format(
                friedman_p_value
            )
        )


class WilcoxonTestResult:
    """Simple class used to represent test result."""

    def __init__(self, name_1, name_2, p_value):
        self.name_1 = name_1
        self.name_2 = name_2
        self.p_value = p_value
        self.is_significant = False


def wilcoxon_holm(result_long_df, metric_name, greater_is_better, alpha=0.05):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm
    and then use Holm to reject the null's hypothesis

    param: result_long_df (pd.DataFrame): benchmarks results in long format.
                                          should contain the columns 'dataset', 'evaluator', 'metric', 'value'.
    param: metric_name (string): name of the metric to consider.
    param: alpha (float): level of (statistical) signifiance, i.e. 0.05 means 'significant at 95%'.

    returns: wilcoxon_test_results (list of WilcoxonTestResults): the list contains one element
                                                                  for each pair of evaluators.
    """
    # variables initialisation
    # filter out result_long_df
    result_long_df = result_long_df[
        (result_long_df["metric"] == metric_name) & (result_long_df["is_test"])
    ]
    result_long_df = _filter_evaluators_according_to_nb_of_datasets(result_long_df)
    if not greater_is_better:
        result_long_df["metric"] *= -1
    result_long_df
    # other variables
    nb_evaluators = result_long_df["evaluator"].nunique()
    results_per_eval = {n: df for n, df in result_long_df.groupby("evaluator")}
    evaluators = list(results_per_eval.keys())

    # test the null hypothesis using friedman before doing a post-hoc analysis
    _check_if_friedmanchisquare_passed(alpha, results_per_eval)

    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    wilcoxon_test_results = []
    # loop through the algorithms to compare pairwise
    for i in range(nb_evaluators - 1):
        # get the name of classifier one
        evaluator_1 = evaluators[i]
        # get the performance of classifier one
        perf_1 = results_per_eval[evaluator_1]["value"].values
        for j in range(i + 1, nb_evaluators):
            # get the name of the second classifier
            evaluator_2 = evaluators[j]
            # get the performance of classifier one
            perf_2 = results_per_eval[evaluator_2]["value"].values
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method="pratt")[1]
            # appen to the list
            wilcoxon_test_results.append(
                WilcoxonTestResult(evaluator_1, evaluator_2, p_value)
            )

    # HOLM CORRECTION
    # https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
    # get the number of hypothesis
    nb_hypotheses = len(wilcoxon_test_results)
    # sort the list in acsending manner of p-value
    wilcoxon_test_results.sort(key=lambda r: r.p_value)
    # loop through the hypothesis
    for hyp_idx, result in enumerate(wilcoxon_test_results):
        # correct alpha with holm
        new_alpha = float(alpha / (nb_hypotheses + 1 - hyp_idx))
        # test if significant after holm's correction of alpha
        if result.p_value <= new_alpha:
            result.is_significant = True
        else:
            # we reject all other test results, see link above.
            break

    return wilcoxon_test_results


def mean_rank(result_long_df, metric_name, greater_is_better):
    """
    Compute the average ranks per evaluator.

    param: result_long_df (pd.DataFrame): benchmarks results in long format.
                                          should contain the columns 'dataset', 'evaluator', 'metric', 'value'.
    param: metric_name (string): name of the metric to consider.

    returns: pd.Series: containing mean rank per evaluator (used as index).
    """
    # convert long format to matrix
    result_matrix = from_long_format_to_matrix(result_long_df, metric_name, True)
    # replace value with rank
    rank_table = result_matrix.rank(axis=0, ascending=not greater_is_better)
    # average over datasets
    return rank_table.mean(axis=1).sort_values(ascending=False)
