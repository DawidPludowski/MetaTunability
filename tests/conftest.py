import pandas as pd

from pytest import fixture
from pathlib import Path

resource_path = Path("tests") / "rescources"


@fixture(scope="session")
def test_datasets():
    datasets_paths = Path(resource_path).glob("*.csv")
    datasets = []

    for path in datasets_paths:
        datasets.append(pd.read_csv(path))

    return datasets


@fixture(scope="session")
def test_datasets_names():
    datasets_paths = Path.glob(resource_path, "*.csv")
    datasets_paths = list(map(lambda x: x.stem, datasets_paths))
    return datasets_paths
