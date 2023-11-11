import pandas as pd

from pathlib import Path
from meta_tuner.data.datasets import PandasDatasets
from typing import List
from openml import datasets


class PandasDatasetsFactory:
    """
    Factory to automize creation of PandasDatasets from
    various sources.
    """

    @staticmethod
    def create_from_dir(path: str | Path) -> PandasDatasets:
        """
        Args:
            path (str | Path): directory path from which all *.csv files
            will be downloaded to PandasDatasets

        Returns:
            PandasDatasets: datasets wrapper with all data from directory.
        """
        datasets_paths = list(Path.glob(path, "*.csv"))
        data_names = list(map(lambda x: x.stem, datasets_paths))
        datasets = list(map(lambda x: pd.read_csv(x), datasets_paths))

        pandas_datasets = PandasDatasets(datasets, data_names)

        return pandas_datasets

    @staticmethod
    def create_from_openml(id: int | List[int]) -> PandasDatasets:
        """
        Args:
            id (int | List[int]): id (or array of ids) refering to
            datasets from OpenML site (https://www.openml.org/)

        Returns:
            PandasDatasets: datasets wrapper with all data specified
            by ids.
        """
        pass
