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


def main():
    logger.info("Loading datasets")
    dataloaders = get_datasets("data/warmstart_d2v_data/train/train_split")

    logger.info("Start searching")
    progress_bar = ((key, data_tuple) for key, data_tuple in dataloaders.items())

    with open("data/warmstart_d2v_data/logistic_scores.pkl", "rb") as f:
        obj = pkl.load(f)
    already_calc = list(
        obj["<class 'sklearn.linear_model._logistic.LogisticRegression'>"].keys()
    )

    logger.warning(f"Number of tasks: {len(dataloaders.keys())}")

    for dir_name, data_tuple in progress_bar:

        if dir_name in already_calc:
            logger.warning(f"ignore task {dir_name}")
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

        logger.info(f"Start training {dir_name}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            searcher.search_holdout(
                train_X, train_y, test_X, test_y, metric, n_iter=100
            )

        put_results(
            "data/warmstart_d2v_data/logistic_scores.pkl",
            dir_name,
            str(model.__class__),
            searcher.search_results["score"],
        )

    with open("data/warmstart_d2v_data/logistic_hpo.pkl", "wb") as f:
        pkl.dump(searcher.search_results["hpo"], f)


if __name__ == "__main__":
    main()
