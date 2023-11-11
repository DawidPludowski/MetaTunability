import pandas as pd

from typing import List, Generator, override


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

        self.__check_init()

    def __check_init(self):
        if self.datasets_names:
            assert len(self.datasets) == len(self.datasets_names)

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
    """
    Extension to PandasWrapper. Implements refering
    to datasets by their OpenML indexes.
    """

    def __init__(
        self,
        datasets: List[pd.DataFrame],
        openml_ids: List[int],
        datasets_names: List[str] = None,
    ) -> None:
        """
        Args:
            datasets (List[pd.DataFrame]): list of pandas dataframes
            openml_ids (List[int]): list of OpenML ids
            datasets_names (List[str], optional): names of datasets, optional. Defaults to None.
        """
        super().__init__(datasets, datasets_names)
        self.openml_ids = openml_ids
        self.__check_init()

    @override
    def __check_init(self):
        assert len(self.openml_ids) == len(self.datasets)
        if self.datasets_names:
            assert len(self.datasets) == len(self.datasets_names)

    def __getitem__(
        self, items: int | str | slice | List[int | str]
    ) -> pd.DataFrame | List[pd.DataFrame]:
        return super().__getitem__(items)

    class _IndexWrapper:
        """
        Index wrapper to perform multiindexing in
        pandas way.
        """

        def __init__(self, obj, openml_ids) -> None:
            self._obj = obj
            self._openml_ids = openml_ids

        def __getitem__(self, items):
            try:
                if isinstance(items, int):
                    return self._obj[self._openml_ids.index(items)]
                elif isinstance(items, list):
                    if isinstance(items[0], int):
                        return self._obj[[self._openml_ids.index(i) for i in items]]
            except ValueError:
                raise IndexError("Provided index is not in present.")

    @property
    def oml_loc(self) -> List[pd.DataFrame] | pd.DataFrame:
        """
        Property-like object to implement pandas-like
        solution to multiple indexing functionality.

        Returns:
            List[pd.DataFrame] | pd.DataFrame: subset of datasets.
        """
        return self._IndexWrapper(self, self.openml_ids)
