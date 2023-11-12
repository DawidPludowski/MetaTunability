import pandas as pd
import shutil

from pytest import fixture
from pathlib import Path

from meta_tuner.data.datasets import PandasDatasets, OpenmlPandasDatasets


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


@fixture(autouse=True)
def run_before_and_after_tests(new_dir):
    """Fixture to execute asserts before and after a test is run"""
    # Setup: fill with any logic you want

    yield  # this is where the testing happens

    # Teardown : fill with any logic you want
    shutil.rmtree(new_dir, ignore_errors=True)
