import json
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from meta_tuner.searchers.search_grid import ConditionalGrid, CubeGrid


def get_logistic_regression_grid(init_seed: int = None) -> ConditionalGrid:
    """
    Create random grid for logistic regression in sklearn package.
    Hyperparameters are set as in docs:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    Returns:
        ConditionalGrid: random grid for logistic regression.
    """
    grid_base = CubeGrid()
    grid_base.add(
        "tol", values=[0.0001, 0.001], space="real", distribution="loguniform"
    )
    grid_base.add("C", values=[0.0001, 10000], space="real", distribution="loguniform")
    grid_base.add(
        "solver",
        values=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        space="cat",
    )

    grid_liblinear = CubeGrid()
    grid_liblinear.add("intercept_scaling", values=[0.001, 1], space="real")
    grid_liblinear.add("penalty", values=["l1", "l2"], space="cat")

    grid_liblinear_ext = CubeGrid()
    grid_liblinear_ext.add("dual", values=[True, False], space="cat")

    grid_saga = CubeGrid()
    grid_saga.add("penalty", values=["elasticnet", "l1", "l2", None], space="cat")
    grid_saga.add("l1_ratio", values=[0, 1], space="real")

    grid_others = CubeGrid()
    grid_others.add("penalty", values=["l2", None], space="cat")

    cond_grid = ConditionalGrid(init_seed=init_seed)
    cond_grid.add_cube(grid_base)
    cond_grid.add_cube(grid_liblinear, lambda hpo: hpo["solver"] == "liblinear")
    cond_grid.add_cube(
        grid_liblinear_ext,
        lambda hpo: hpo["solver"] == "liblinear" and hpo["penalty"] == "l2",
    )
    cond_grid.add_cube(grid_saga, lambda hpo: hpo["solver"] == "saga")
    cond_grid.add_cube(
        grid_others, lambda hpo: hpo["solver"] not in ("liblinear", "saga")
    )

    return cond_grid


def get_predefined_logistic_regression() -> LogisticRegression:
    model = LogisticRegression(random_state=123, max_iter=500)
    return model


def get_gaussian_process_grid() -> ConditionalGrid:
    base_grid = CubeGrid()
    base_grid.add("n_restarts_optimizer", values=[1, 10], space="int")
    base_grid.add("max_iter_predict", values=[10, 200], space="int")
    base_grid.add("warm_start", values=[False, True], space="cat")

    cond_grid = ConditionalGrid()
    cond_grid.add_cube(base_grid)

    return cond_grid


def get_predefined_gaussian_classifier() -> GaussianProcessClassifier:
    model = GaussianProcessClassifier(random_state=123)
    return model


def get_svc_grid() -> ConditionalGrid:
    base_grid = CubeGrid()
    base_grid.add("C", values=[0.001, 1000], space="real", distribution="loguniform")
    base_grid.add("kernel", values=["linear", "poly", "rbf", "sigmoid"], space="cat")
    base_grid.add("gamma", values=["auto", "scale"], space="cat")
    base_grid.add("shrinking", values=[True, False], space="cat")
    base_grid.add("probability", values=[True, False], space="cat")
    base_grid.add("tol", values=[0.0001, 0.01], space="real")

    coef_grid = CubeGrid()
    coef_grid.add("coef0", values=[0.0001, 1], space="real")

    cond_grid = ConditionalGrid()
    cond_grid.add_cube(base_grid)
    cond_grid.add_cube(cond_grid, lambda hpo: hpo["kernel"] in ["poly", "sigmoid"])

    return cond_grid


def get_predefined_svc() -> SVC:
    model = SVC(random_state=123)
    return model


def get_datasets(path: str | Path) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    root_path = Path(path)

    directories = root_path.glob("*/*")
    data_tuples = {}

    for path in directories:
        df_train = pd.read_csv(path / "train.csv")
        df_test = pd.read_csv(path / "test.csv")

        data_tuples[str(path)] = (df_train, df_test)

    return data_tuples


def put_results(path: Path, model: str, data: Dict[str, any]) -> None:
    path = Path(path) / "results.pkl"
    if path.is_file():
        with open(path, "rb") as f:
            obj = pkl.load(f)
            obj[model] = data
        with open(path, "wb") as f:
            pkl.dump(obj, f)

    else:
        with open(path, "wb") as f:
            pkl.dump({model: data}, f)
