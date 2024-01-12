import warnings

from loguru import logger
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from bin.utils import (
    get_datasets,
    get_logistic_regression_grid,
    get_predefined_logistic_regression,
    put_results,
)
from meta_tuner.searchers.hpo_searchers import RandomSearch

MIN_BEST = False


def main():
    logger.info("Loading datasets")
    dataloaders = get_datasets("data/mini_holdout")

    logger.info("Start searching")
    progress_bar = tqdm((key, data_tuple) for key, data_tuple in dataloaders.items())

    logger.warning(len(dataloaders.keys()))

    for dir_name, data_tuple in progress_bar:
        logger.info(f"Dataset: {dir_name}")

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
            searcher.search_holdout(train_X, train_y, test_X, test_y, roc_auc_score)

        put_results(dir_name, {"best_hpo": searcher.get_best_hpo(min_best=MIN_BEST)})


if __name__ == "__main__":
    main()
