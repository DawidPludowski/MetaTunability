import pandas as pd
import pytest

from pathlib import Path

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


def test_to_dir(pandas_datasets, new_dir):
    pandas_datasets.to_dir(new_dir)

    assert Path(new_dir).is_dir
    assert len(list(Path.glob(new_dir, "*.csv"))) == 4


def test_to_dir_already_exists(pandas_datasets, new_dir):
    pandas_datasets.to_dir(new_dir)

    with pytest.raises(OSError) as e:
        pandas_datasets.to_dir(new_dir)


def test_to_dir_without_names(pandas_datasets, new_dir):
    pandas_datasets.datasets_names = None
    pandas_datasets.to_dir(new_dir)

    assert Path(new_dir).is_dir
    assert len(list(Path.glob(new_dir, "data*.csv"))) == 4


def test_to_dir_lazy(lazy_datasets, new_dir):
    lazy_datasets.to_dir(new_dir)

    assert Path(new_dir).is_dir
    assert len(list(Path.glob(new_dir, "*.csv"))) == 4


def test_download_lazy(lazy_datasets):
    lazy_datasets.download_datasets = True
    lazy_datasets[0]

    assert lazy_datasets.datasets[0] is not None


def test_lazy_datasets_get_int(lazy_datasets):
    assert isinstance(lazy_datasets[0], pd.DataFrame)


def test_lazy_datasets_get_slice(lazy_datasets):
    datasets_slice = lazy_datasets[0:3]
    assert isinstance(datasets_slice, list)
    assert isinstance(datasets_slice[0], pd.DataFrame)
    assert len(datasets_slice) == 3


def test_lazy_datasets_get_int_list(lazy_datasets):
    dataset_list = lazy_datasets[[0, 2]]
    assert isinstance(dataset_list, list)
    assert isinstance(dataset_list[0], pd.DataFrame)
    assert len(dataset_list) == 2


def test_lazy_datasets_get_str(lazy_datasets):
    assert isinstance(lazy_datasets["credit-g"], pd.DataFrame)


def test_lazy_datasets_get_str_list(lazy_datasets):
    dataset_list = lazy_datasets[["credit-g", "kr-vs-kp"]]
    assert isinstance(dataset_list, list)
    assert isinstance(dataset_list[0], pd.DataFrame)
    assert len(dataset_list) == 2


def test_lazy_datasets_load_already_downloaded_data(lazy_datasets):
    lazy_datasets.download_datasets = True
    lazy_datasets[0]
    assert id(lazy_datasets[0]) == id(lazy_datasets[0])


def test_lazy_datasets_wrong_indexes(lazy_datasets):
    with pytest.raises(IndexError) as e:
        lazy_datasets["not_exists"]
    with pytest.raises(IndexError) as e:
        lazy_datasets[99]
