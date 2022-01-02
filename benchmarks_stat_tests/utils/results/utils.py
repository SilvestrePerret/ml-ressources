# standard imports

# external imports
import numpy as np
import pandas as pd


def from_result_dict_to_long_format(result_dict):
    """
    Transform the benchmark results under dict form to long dataframe form.

    param: result_dict (dict): benchmarks results, should have the following structure
                               {
                                   dataset_name: {
                                       evaluator_name: {
                                           "test_scores": {
                                               metric_name: [...],
                                               ...
                                           },
                                           ...
                                       },
                                       ...
                                   },
                                   ...
                               }

    returns: result_long_df (pd.DataFrame): benchmarks results, contains the columns
                                            'evaluator', 'dataset', 'metric', 'is_test', 'fold_idx', 'value'
    """
    # initialisation
    columns = ["evaluator", "dataset", "metric", "is_test", "fold_idx", "value"]
    result_long = []
    for dataset in result_dict:
        for evaluator in result_dict[dataset]:
            for key in result_dict[dataset][evaluator]:
                if "time" in key:
                    value = result_dict[dataset][evaluator][key]
                    result_long.append([evaluator, dataset, key, np.nan, np.nan, value])
                elif "scores" in key:
                    is_test = "test" in key
                    for metric in result_dict[dataset][evaluator][key]:
                        scores = result_dict[dataset][evaluator][key][metric]
                        for fold_idx, value in enumerate(scores):
                            result_long.append(
                                [evaluator, dataset, metric, is_test, fold_idx, value]
                            )
    return pd.DataFrame(data=result_long, columns=columns)


def from_result_dict_to_param_long_format(result_dict):
    # initialisation
    columns = ["evaluator", "dataset", "metric", "param_name", "param_value"]
    result_long = []
    for dataset in result_dict:
        for evaluator in result_dict[dataset]:
            for metric in result_dict[dataset][evaluator]["best_params"]:
                best_params = result_dict[dataset][evaluator]["best_params"][metric]
                for name, value in best_params.items():
                    result_long.append(
                        [
                            evaluator,
                            dataset,
                            metric,
                            name,
                            value if value is not None else np.nan,
                        ]
                    )
    return pd.DataFrame(data=result_long, columns=columns)


def from_long_format_to_matrix(result_long_df, metric_name, is_test=True):
    """
    Transform the benchmark results under long dataframe form to matrix form.

    param: result_long_df (pd.DataFrame): benchmarks results, contains the columns
                                          'evaluator', 'dataset', 'metric' and 'value'
    partam: metric_name (string): name of the metric to consider

    returns: pd.DataFrame: contains a columns for each dataset, a row for each evaluator,
                           value in cells are the param:metric_name of this evaluator on this dataset.
    """
    df = result_long_df[
        (result_long_df["metric"] == metric_name) & (result_long_df["is_test"])
    ]
    return pd.pivot_table(df, values="value", index="evaluator", columns="dataset",)
