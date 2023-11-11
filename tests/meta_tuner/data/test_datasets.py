import pandas as pd

from meta_tuner.data.datasets import PandasDatasets


def test_pandas_datasets_get_int(test_datasets, test_datasets_names):
    datasets = PandasDatasets(test_datasets, test_datasets_names)

    assert isinstance(datasets[0], pd.DataFrame)


def test_pandas_datasets_get_str(test_datasets, test_datasets_names):
    datasets = PandasDatasets(test_datasets, test_datasets_names)

    assert isinstance(datasets["credit-g"], pd.DataFrame)


def test_pandas_datasets_get_slice(test_datasets, test_datasets_names):
    datasets = PandasDatasets(test_datasets, test_datasets_names)

    datasets_slice = datasets[0:3]
    assert isinstance(datasets_slice, list)
    assert isinstance(datasets_slice[0], pd.DataFrame)
    assert len(datasets_slice) == 3


def test_pandas_datasets_get_int_list(test_datasets, test_datasets_names):
    datasets = PandasDatasets(test_datasets, test_datasets_names)

    dataset_list = datasets[[0, 2]]
    assert isinstance(dataset_list, list)
    assert isinstance(dataset_list[0], pd.DataFrame)
    assert len(dataset_list) == 2


def test_pandas_datasets_get_str_list(test_datasets, test_datasets_names):
    datasets = PandasDatasets(test_datasets, test_datasets_names)

    dataset_list = datasets[["credit-g", "kr-vs-kp"]]
    assert isinstance(dataset_list, list)
    assert isinstance(dataset_list[0], pd.DataFrame)
    assert len(dataset_list) == 2
