import json
import pickle as pkl
import warnings
from pathlib import Path

from loguru import logger
from sklearn.metrics import roc_auc_score as metric
from tqdm import tqdm

from bin.utils import (
    get_datasets,
    get_logistic_regression_grid,
    get_predefined_logistic_regression,
    put_results,
)
from meta_tuner.searchers.hpo_searchers import RandomSearch

MIN_BEST: bool = False
TRUNCATE_NUMBER: int = 167


def main():
    logger.info("Loading datasets")
    dataloaders = get_datasets("data/mini_holdout")

    logger.info("Start searching")
    progress_bar = tqdm((key, data_tuple) for key, data_tuple in dataloaders.items())

    logger.warning(f"Number of tasks: {len(dataloaders.keys())}")

    for dir_name, data_tuple in progress_bar:
        if int(dir_name[-5:]) >= TRUNCATE_NUMBER:
            continue

        grid = get_logistic_regression_grid()
        model = get_predefined_logistic_regression()
        searcher = RandomSearch(model, grid)

        train_X, train_y, test_X, test_y = (
            data_tuple[0].iloc[:, :-1],
            data_tuple[0].iloc[:, -1],
            data_tuple[1].iloc[:, :-1],
            data_tuple[1].iloc[:, -1],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            searcher.search_holdout(
                train_X, train_y, test_X, test_y, metric, n_iter=200
            )

        put_results(
            "data/mini_holdout/logistic_scores.pkl",
            dir_name,
            str(model.__class__),
            searcher.search_results["score"],
        )

    with open("data/mini_holdout/logistic_hpo.pkl", "wb") as f:
        pkl.dump(searcher.search_results["hpo"], f)


if __name__ == "__main__":
    main()
