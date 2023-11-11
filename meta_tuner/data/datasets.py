import pandas as pd

from typing import List, Generator


class PandasDatasets:
    def __init__(
        self,
        datasets: List[pd.DataFrame],
        datasets_names: pd.DataFrame | List[dict] = None,
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.datasets_names = datasets_names

    def __iter__(self) -> Generator[pd.DataFrame, None, None]:
        for dataset in self.datasets:
            yield dataset

    def __getitem__(
        self, items: int | str | slice | List[int | str]
    ) -> pd.DataFrame | List[pd.DataFrame]:
        if isinstance(items, slice):
            return self.datasets[items]
        elif isinstance(items, int):
            return self.datasets[items]
        elif isinstance(items, str):
            return self.datasets[self.datasets_names.index(items)]
        elif isinstance(items, list):
            if isinstance(items[0], int):
                return [self.datasets[i] for i in items]
            elif isinstance(items[0], str):
                return [self.datasets[self.datasets_names.index(i)] for i in items]
