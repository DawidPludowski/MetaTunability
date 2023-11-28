import pytest
import numpy as np

from meta_tuner.searchers.search_grid import CubeGrid, ConditionalGrid
from collections import Counter


def test_cube_grid_pass_int():
    grid = CubeGrid()
    grid.add("param", 999, space="int")
    rand_values = [grid.pick()["param"] == 999 for _ in range(100)]

    assert all(rand_values)

    grid = CubeGrid()
    grid.add("param", 999, space="real")
    rand_values = [grid.pick()["param"] == 999 for _ in range(100)]

    assert all(rand_values)


def test_cube_grid_pass_str():
    grid = CubeGrid()
    grid.add("param", "category", space="cat")
    rand_values = [grid.pick()["param"] == "category" for _ in range(100)]

    assert all(rand_values)


def test_cube_grid_pass_int_space():
    grid = CubeGrid()
    grid.add("param", [0, 1], space="int")
    rand_values = [grid.pick()["param"] for _ in range(100)]

    counter = Counter(rand_values)
    counter_dir = dict(counter.items())

    assert counter_dir[0] > 0
    assert counter_dir[1] > 0

    with pytest.raises(KeyError):
        assert counter_dir[2]


def test_cube_grid_pass_real_space():
    n = 10_000
    grid = CubeGrid()
    grid.add("param", [0, 1], space="real")
    rand_values = np.array([grid.pick()["param"] for _ in range(n)])

    theoretical_mean = 0.5
    theoretical_std = 1 / np.sqrt(12)

    assert np.abs(rand_values.mean() - theoretical_mean).item() < 1e-2
    assert np.abs(rand_values.std() - theoretical_std).item() < 1e-2


def test_cube_grid_pass_real_logspace():
    n = 10_000
    grid = CubeGrid()
    grid.add("param", [1, 2], space="real", distribution="loguniform")
    rand_values = np.array([grid.pick()["param"] for _ in range(n)])

    theoretical_mean = 1 / np.log(2)
    theoretical_std = np.sqrt((3 / (2 * np.log(2))) - np.power(1 / np.log(2), 2))

    assert np.abs(rand_values.mean() - theoretical_mean).item() < 1e-1
    assert np.abs(rand_values.std() - theoretical_std).item() < 1e-1


def test_cube_grid_wrong_space():
    grid = CubeGrid()

    with pytest.raises(ValueError) as e:
        grid.add("param", [1, 2], space="notExists")

    with pytest.raises(ValueError) as e:
        grid.add("param", 1, space="cat")


def test_cube_grid_pass_cat():
    n = 10_000
    grid = CubeGrid()
    grid.add("param", ("a", "b", "c"), space="cat")
    rand_values = np.array([grid.pick()["param"] for _ in range(n)])

    counter = Counter(rand_values)
    counter_dir = dict(counter.items())

    assert counter_dir["a"] > 0
    assert counter_dir["b"] > 0
    assert counter_dir["c"] > 0

    with pytest.raises(KeyError):
        assert counter_dir["d"]


def test_cube_grid_reset_seed():
    grid = CubeGrid(init_seed=123)
    grid.add("param_1", [0, 1], space="real")
    grid.add("param_2", [0, 1], space="real")

    values = grid.pick()
    value_1, value_2 = values["param_1"], values["param_2"]

    for _ in range(10):
        grid.reset_seed()
        new_values = grid.pick()
        new_value_1, new_value_2 = new_values["param_1"], new_values["param_2"]

        assert value_1 == new_value_1
        assert value_2 == new_value_2


def test_cond_grid_reset_seed():
    grid_1 = CubeGrid()
    grid_1.add("param_1", [0, 1], space="real")

    grid_2 = CubeGrid()
    grid_2.add("param_2", [0, 1], space="real")

    grid_3 = CubeGrid()
    grid_3.add("param_2", [1, 2], space="real")

    cond_grid = ConditionalGrid(init_seed=123)
    cond_grid.add_cube(grid_1)
    cond_grid.add_cube(grid_2, lambda hpo: hpo["param_1"] > 0.5)
    cond_grid.add_cube(grid_2, lambda hpo: hpo["param_1"] <= 0.5)

    values = cond_grid.pick()
    value_1, value_2 = values["param_1"], values["param_2"]

    for _ in range(10):
        cond_grid.reset_seed()
        new_values = cond_grid.pick()
        new_value_1, new_value_2 = new_values["param_1"], new_values["param_2"]

        assert value_1 == new_value_1
        assert value_2 == new_value_2


def test_cond_grid_condition():
    grid_1 = CubeGrid()
    grid_1.add("param_1", [0, 0.5], space="real")

    grid_2 = CubeGrid()
    grid_2.add("param_2", 1, space="int")

    grid_3 = CubeGrid()
    grid_3.add("param_2", 2, space="int")

    cond_grid = ConditionalGrid(init_seed=123)
    cond_grid.add_cube(grid_1)
    cond_grid.add_cube(grid_2, lambda hpo: hpo["param_1"] > 0.5)
    cond_grid.add_cube(grid_2, lambda hpo: hpo["param_1"] <= 0.5)

    for _ in range(100):
        assert cond_grid.pick()["param_2"] == 1


def test_cond_grind_wrong_condition():
    grid_1 = CubeGrid()
    grid_1.add("param_1", [0, 1], space="real")

    grid_2 = CubeGrid()
    grid_2.add("param_2", 1, space="int")

    grid_3 = CubeGrid()
    grid_3.add("param_2", 2, space="int")

    cond_grid = ConditionalGrid(init_seed=123)
    cond_grid.add_cube(grid_1)
    cond_grid.add_cube(grid_2, lambda hpo: hpo["notExists"] > 0.5)
    cond_grid.add_cube(grid_2, lambda hpo: hpo["notExists"] <= 0.5)

    with pytest.raises(KeyError) as e:
        cond_grid.pick()
