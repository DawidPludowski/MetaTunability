import numpy as np

from typing import Dict, Tuple, List, Callable
from functools import partial
from copy import deepcopy
from abc import ABC, abstractmethod


class RandomGrid(ABC):
    @abstractmethod
    def pick() -> Dict[str, any]:
        ...


class CubeGrid(RandomGrid):
    """
    Interface to generate random values based on
    cube, i.e. when variables' values do not depends
    on others.
    """

    def __init__(self, init_seed: int = None) -> None:
        """
        Args:
            init_seed (int, optional): Random seed. Defaults to None.
        """
        self.init_seed = init_seed

        self.names: List[str] = []
        self.rngs: List[Callable[..., np.ndarray]] = []

    def add(
        self,
        name: str,
        values: int | str | Tuple[int | str],
        space: str = None,
        distribution: str = "uniform",
    ) -> None:
        """
        Add new dimension to cube.

        Args:
            name (str): name of dimension
            range (int | str | Tuple[str]): range of value. If int|str:
                fixed value, if tuple|list of two numbers: range(low, high),
                if tuple|list of string: distinct values.
            values (str): Type of space. Valid values: ("real", "int", "cat")
            distribution (str, optional): Type of distribution if real numbers.
                Valid values: ("uniform", "loguniform"). Defaults to "uniform".
        """

        rng = np.random.default_rng(seed=self.init_seed)
        rng_len = len(self.rngs)

        if space == "real":
            if isinstance(values, int):
                self.rngs.append(lambda: values)
            if isinstance(values, list):
                if distribution == "uniform":
                    self.rngs.append(lambda: rng.uniform(values[0], values[1]))
                elif distribution == "loguniform":
                    self.rngs.append(lambda: self.__lognuniform(values[0], values[1]))
        elif space == "int":
            if isinstance(values, int):
                self.rngs.append(lambda: values)
            if isinstance(values, list):
                self.rngs.append(lambda: rng.integers(values[0], values[1]))
        elif space == "cat":
            if isinstance(values, str):
                self.rngs.append(lambda: values)
            if isinstance(values, (list, tuple)):
                self.rngs.append(lambda: rng.choice(values, replace=True))
        else:
            raise ValueError(
                f'For arg "values" only "real", "int", "cat" are allowed. {values} provided'
            )

        if len(self.rngs) == rng_len:
            raise ValueError("Wrong parameters provided.")

        self.names.append(name)

    def pick(self) -> Dict[str, any]:
        """
        Generate random point from the grid.

        Returns:
            Dict[str, any]: List of points from grid.
        """
        random_coordinates = [rng() for rng in self.rngs]
        dict_coordinates = {
            name: coordinates
            for name, coordinates in zip(self.names, random_coordinates)
        }
        return dict_coordinates

    def __lognuniform(self, low=0, high=1, base=np.e):
        rng = np.random.default_rng(seed=self.init_seed)
        rng_ = partial(rng.uniform, **{"low": low, "high": high})

        return np.power(base, rng_()) / base


class ConditionalGrid(RandomGrid):
    """
    Interface to generate random values from conditional
    space, i.e. when space of one dimension depends on
    other.
    """

    def __init__(self, init_seed: int = None) -> None:
        """
        Args:
            init_seed (int, optional): If provided, override init_seed
            on each cube that is added to it. Defaults to None.
        """
        self.init_seed = init_seed

        self.cubes: List[CubeGrid] = []
        self.conditions: List[Callable[..., bool]] = []

    def add_cube(
        self, cube: CubeGrid, condition: Callable[..., bool] = lambda _: True
    ) -> None:
        """Add cube to sequence. Random values are generated in sequence
        of adding them, starting from first.

        Args:
            cube (CubeGrid): CubeGrid object from which random values will be picked.
            condition (Callable, optional): Condition function to determine if cube
                will be used in picking. Funciton's input is dictionary with already generated
                values. Function should return True or False. Defaults to lambda_:True.
        """
        cube_ = deepcopy(cube)
        if self.init_seed:
            cube_.init_seed = self.init_seed
        self.cubes.append(cube_)
        self.conditions.append(condition)

    def pick(self) -> Dict[str, any]:
        """
        Generate random point from the conditional grid.

        Returns:
            Dict[str, any]: List of points from grid.
        """
        pick = {}

        for cube, condition in zip(self.cubes, self.conditions):
            try:
                if condition(pick):
                    pick |= cube.pick()
            except KeyError as e:
                raise KeyError(
                    f'In condition, key "{e.args[0]}" was used but not present in previous cubes.'
                )

        return pick
