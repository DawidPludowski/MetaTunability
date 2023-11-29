import typer
import pickle as pkl
import warnings

from pathlib import Path
from typing_extensions import Annotated
from typing import Dict, List
from sklearn.metrics import roc_auc_score
from loguru import logger
from tqdm import tqdm

from utils import (
    get_logistic_regression_grid,
    get_predefined_logistic_regression,
    get_openml_bin_task_ids,
)
from meta_tuner.searchers.search_grid import ConditionalGrid
from meta_tuner.data.factory import PandasDatasetsFactory
from meta_tuner.searchers.preprocessors import wrap_model_with_preprocessing
from meta_tuner.searchers.hpo_searchers import RandomSearch

app = typer.Typer()
warnings.filterwarnings("ignore")


def perform_experiment_bin_openml(
    id_: int, model: any, grid: ConditionalGrid, n_iter: int, cv: int
) -> Dict[str, List[any]]:
    datasets = PandasDatasetsFactory.create_from_openml(ids=[id_])
    model_wrapped = wrap_model_with_preprocessing(model)
    df = datasets[0]

    X, y = df.iloc[:, :-1], df.iloc[:, [-1]]

    grid.reset_seed()
    search = RandomSearch(model_wrapped, grid)
    search.search(X, y, roc_auc_score, n_iter=n_iter, cv=cv, encode_y=True)

    return search.search_results


def is_search_completed(data_id: int, results_dir: str | Path) -> bool:
    files = Path(results_dir).glob(f"*id={data_id}.pkl")

    return len(list(files)) != 0


def save_results(
    results: Dict[str, List[any]],
    file_name: str,
    results_dir: str | Path,
) -> None:
    results_dir = Path(results_dir)
    with open(results_dir / file_name, "wb") as f:
        pkl.dump(results, f)


def init_directory(results_dir: Path | str) -> None:
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)


@app.command(help="run logistic regression experiment on roc-auc score.")
def main(
    results_dir: Annotated[
        Path, typer.Option(..., help="path to directory to store results.")
    ] = "results/linear/classification/bin/roc-auc",
    n_iter: Annotated[
        int, typer.Option(..., help="Number of iteration in random search.")
    ] = 500,
    cv: Annotated[
        int, typer.Option(..., help="Number of folds in cross validation.")
    ] = 3,
    seed: Annotated[
        int,
        typer.Option(..., help="Seed. If not provided, each experiment will be random"),
    ] = None,
) -> None:
    logger.info("Create init objects.")
    init_directory(results_dir)
    ids = get_openml_bin_task_ids()
    model = get_predefined_logistic_regression()
    grid = get_logistic_regression_grid(seed)

    logger.info("Start experiment.")
    progress_bar = tqdm(ids)
    for id_, name_ in progress_bar:
        if not is_search_completed(id_, results_dir):
            logger.info(f"Dataset id={id_}, name={name_}")
            try:
                file_name = f"name={name_}-id={id_}.pkl"
                results = perform_experiment_bin_openml(
                    id_, model, grid, n_iter=n_iter, cv=cv
                )
                save_results(results, file_name, results_dir)
            except Exception:
                logger.error(f"Dataset id={id_} results in error and skipped.")
        else:
            logger.warning(f"Dataset with id={id_} is already computed.")


if __name__ == "__main__":
    main()
