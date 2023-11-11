from meta_tuner.data.factory import PandasDatasetsFactory


def test_factory_from_dir(resource_path):
    datasets = PandasDatasetsFactory.create_from_dir(resource_path)
    assert len(datasets) == 4
