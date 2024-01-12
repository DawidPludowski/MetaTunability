import warnings

from loguru import logger
from sklearn.metrics import roc_auc_score as metric
from tqdm import tqdm

from bin.utils import get_datasets, get_predefined_svc, get_svc_grid, put_results
from meta_tuner.searchers.hpo_searchers import RandomSearch

MIN_BEST = False


def main():
    logger.info("Loading datasets")
    dataloaders = get_datasets("data/mini_holdout")

    logger.info("Start searching")
    progress_bar = tqdm((key, data_tuple) for key, data_tuple in dataloaders.items())

    logger.warning(f"Number of tasks: {len(dataloaders.keys())}")

    for dir_name, data_tuple in progress_bar:
        grid = get_svc_grid()
        model = get_predefined_svc()
        searcher = RandomSearch(model, grid)

        train_X, train_y, test_X, test_y = (
            data_tuple[0].iloc[:, :-1],
            data_tuple[0].iloc[:, -1],
            data_tuple[1].iloc[:, :-1],
            data_tuple[1].iloc[:, -1],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            searcher.search_holdout(train_X, train_y, test_X, test_y, metric)

        put_results(
            dir_name,
            str(model.__class__),
            {
                "best_hpo": searcher.get_best_hpo(min_best=MIN_BEST),
                "scoring_func": str(metric.__name__),
                "best_score": min(searcher.search_results["score"])
                if MIN_BEST
                else max(searcher.search_results["score"]),
                "early_stopping": str(searcher.early_stopping.__class__),
                "is_min_score_best": MIN_BEST,
            },
        )


if __name__ == "__main__":
    main()
