import pandas as pd

from pytest import fixture
from pathlib import Path

from meta_tuner.data.datasets import PandasDatasets


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
