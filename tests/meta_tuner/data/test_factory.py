from meta_tuner.data.factory import PandasDatasetsFactory


def test_factory_from_dir(resource_path):
    datasets = PandasDatasetsFactory.create_from_dir(resource_path)
    assert len(datasets) == 4


def test_factory_from_openml():
    datasets = PandasDatasetsFactory.create_from_openml([31, 1504, 3, 1494])

    assert "credit-g" in datasets.datasets_names
    assert "kr-vs-kp" in datasets.datasets_names
    assert "qsar-biodeg" in datasets.datasets_names
    assert "steel-plates-fault" in datasets.datasets_names

    assert len(datasets) == 4


def test_factory_single_id():
    datasets = PandasDatasetsFactory.create_from_openml(31)
    assert "credit-g" in datasets.datasets_names
