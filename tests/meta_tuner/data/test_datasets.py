import pandas as pd
import pytest

from meta_tuner.data.datasets import PandasDatasets, OpenmlPandasDatasets


def test_pandas_datasets_get_int(pandas_datasets):
    assert isinstance(pandas_datasets[0], pd.DataFrame)


def test_pandas_datasets_get_str(pandas_datasets):
    assert isinstance(pandas_datasets["credit-g"], pd.DataFrame)


def test_pandas_datasets_get_slice(pandas_datasets):
    datasets_slice = pandas_datasets[0:3]
    assert isinstance(datasets_slice, list)
    assert isinstance(datasets_slice[0], pd.DataFrame)
    assert len(datasets_slice) == 3


def test_pandas_datasets_get_int_list(pandas_datasets):
    dataset_list = pandas_datasets[[0, 2]]
    assert isinstance(dataset_list, list)
    assert isinstance(dataset_list[0], pd.DataFrame)
    assert len(dataset_list) == 2


def test_pandas_datasets_get_str_list(pandas_datasets):
    dataset_list = pandas_datasets[["credit-g", "kr-vs-kp"]]
    assert isinstance(dataset_list, list)
    assert isinstance(dataset_list[0], pd.DataFrame)
    assert len(dataset_list) == 2


def test_pandas_datasets_iterator(pandas_datasets):
    cnt = 0
    for _ in pandas_datasets:
        cnt += 1

    assert cnt == 4


def test_exception_raisining_wrong_int(pandas_datasets):
    with pytest.raises(IndexError) as e:
        pandas_datasets[10]

    with pytest.raises(IndexError) as e:
        pandas_datasets[[0, 1, 10]]


def test_exception_raisining_wrong_str(pandas_datasets):
    with pytest.raises(IndexError) as e:
        pandas_datasets["not exists"]

    with pytest.raises(IndexError) as e:
        pandas_datasets[["not exists", "credit-g"]]


def test_exception_raising_different_len(test_datasets):
    with pytest.raises(AssertionError) as e:
        PandasDatasets(test_datasets, ["a", "b", "c"])


def test_exception_raising_different_len_openml(test_datasets):
    with pytest.raises(AssertionError) as e:
        OpenmlPandasDatasets(test_datasets, [1, 2, 3])


def test_exception_wrong_id(openml_datasets):
    with pytest.raises(IndexError):
        openml_datasets.oml_loc[999]


def test_openml_indexing(openml_datasets):
    openml_datasets.oml_loc[101]
    openml_datasets.oml_loc[[101, 103]]
