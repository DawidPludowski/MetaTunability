import pandas as pd

from typing import Any, List, Generator


class PandasDatasets:
    """
    Wrapper to list of the datasets. Allows to refer to
    items by slices, list of integers and list of datasets names.
    """

    def __init__(
        self,
        datasets: List[pd.DataFrame],
        datasets_names: List[str] = None,
    ) -> None:
        """
        Args:
            datasets (List[pd.DataFrame]): list of pandas datasets
            datasets_names (List[str], optional): names of datasets, optional. Defaults to None.
        """
        super().__init__()
        self.datasets = datasets
        self.datasets_names = datasets_names

    def __iter__(self) -> Generator[pd.DataFrame, None, None]:
        for dataset in self.datasets:
            yield dataset

    def __getitem__(
        self, items: int | str | slice | List[int | str]
    ) -> pd.DataFrame | List[pd.DataFrame]:
        try:
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
        except IndexError:
            raise IndexError("Provided index is out of range.")
        except ValueError:
            raise IndexError("Provided dataset name does not exist.")

    def __len__(self):
        return len(self.datasets)


class OpenmlPandasDatasets(PandasDatasets):
    def __init__(
        self,
        datasets: List[pd.DataFrame],
        openml_ids: List[int],
        datasets_names: List[str] = None,
    ) -> None:
        super().__init__(datasets, datasets_names)
        self.openml_ids = openml_ids

    def __getitem__(
        self, items: int | str | slice | List[int | str]
    ) -> pd.DataFrame | List[pd.DataFrame]:
        return super().__getitem__(items)

    class _IndexWrapper:
        def __init__(self, obj, openml_ids) -> None:
            self._obj = obj
            self._openml_ids = openml_ids

        def __getitem__(self, items):
            return self._obj[self._openml_ids.index(items)]

    @property
    def oml_loc(self):
        return self._IndexWrapper(self, self.openml_ids)
