from sklearn.linear_model import LogisticRegression
from typing import List, Tuple
from openml import tasks

from meta_tuner.searchers.search_grid import CubeGrid, ConditionalGrid


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
    grid_saga.add("lr_raito", values=[0, 1], space="real")

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


def get_openml_bin_task_ids(rows_treshold: int = None) -> List[Tuple[int, str]]:
    classification = tasks.list_tasks(
        task_type=tasks.TaskType.SUPERVISED_CLASSIFICATION
    )

    data_ids = []

    for task in classification.values():
        if task.get("NumberOfClasses") is None:
            continue
        if rows_treshold is not None and task["NumberOfInstances"] > rows_treshold:
            continue
        if task["NumberOfClasses"] == 2:
            data_ids.append((task["did"], task["name"]))

    return data_ids
