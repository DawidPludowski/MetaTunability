import pandas as pd
import shutil

from pytest import fixture
from pathlib import Path

from meta_tuner.data.datasets import (
    PandasDatasets,
    OpenmlPandasDatasets,
    LazyPandasDatasets,
)
from meta_tuner.searchers.search_grid import CubeGrid, ConditionalGrid


@fixture(scope="session")
def new_dir():
    return Path("tests") / "resources" / "temp_dir"


@fixture(scope="session")
def resource_path():
    return Path("tests") / "resources"


@fixture(scope="session")
def test_datasets(resource_path):
    datasets_paths = Path(resource_path).glob("*.csv")
    datasets = []

    for path in datasets_paths:
        datasets.append(pd.read_csv(path))

    return datasets


@fixture(scope="session")
def test_datasets_names(resource_path):
    datasets_paths = Path.glob(resource_path, "*.csv")
    datasets_names = list(map(lambda x: x.stem, datasets_paths))
    return datasets_names


@fixture(scope="function")
def pandas_datasets(test_datasets, test_datasets_names):
    return PandasDatasets(test_datasets, test_datasets_names)


@fixture(scope="session")
def openml_datasets(test_datasets, test_datasets_names):
    return OpenmlPandasDatasets(
        test_datasets, [100, 101, 102, 103], test_datasets_names
    )


@fixture(scope="function")
def lazy_datasets(resource_path):
    data_paths = Path(resource_path).glob("*.csv")
    datasets = LazyPandasDatasets(data_paths, download_datasets=False)
    return datasets


@fixture(autouse=True)
def run_before_and_after_tests(new_dir):
    """Fixture to execute asserts before and after a test is run"""
    # Setup: fill with any logic you want

    yield  # this is where the testing happens

    # Teardown : fill with any logic you want
    shutil.rmtree(new_dir, ignore_errors=True)


@fixture(scope="function")
def search_result():
    dir_1 = {"hpo": [{"a": 1}, {"a": 2}], "mean_score": [200, 99]}
    dir_2 = {"hpo": [{"a": 1}, {"a": 2}], "mean_score": [0, 10]}
    dir_3 = {"hpo": [{"a": 1}, {"a": 2}], "mean_score": [5, 4]}

    return [dir_1, dir_2, dir_3]


@fixture(scope="function")
def logistic_grid():
    grid_1 = CubeGrid()
    grid_1.add("solver", ("lbfgs", "liblinear", "newton-cg"), "cat")
    grid_1.add("C", [0.001, 100], "real")

    grid_2 = CubeGrid()
    grid_2.add("penalty", ("l1"), "cat")

    grid_3 = CubeGrid()
    grid_3.add("penalty", ("l2"), "cat")

    cond_grid = ConditionalGrid(init_seed=123)
    cond_grid.add_cube(grid_1)
    cond_grid.add_cube(grid_2, lambda hpo: hpo["solver"] == "liblinear")
    cond_grid.add_cube(grid_3, lambda hpo: hpo["solver"] != "liblinear")

    return cond_grid


@fixture(scope="function")
def naive_logistic_grid():
    grid_1 = CubeGrid()
    grid_1.add("solver", ("liblinear", "newton-cg"), "cat")

    cond_grid = ConditionalGrid(init_seed=123)
    cond_grid.add_cube(grid_1)

    return cond_grid
