import numpy as np

from typing import Dict, Tuple, List, Callable
from functools import partial


class CubeGrid:
    def __init__(self, init_seed: int = None) -> None:
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
                    self.rngs.append(
                        lambda size: rng.uniform(values[0], values[1], size=size)
                    )
                elif distribution == "loguniform":
                    self.rngs.append(
                        lambda size: self.__lognuniform(values[0], values[1], size=size)
                    )
        elif space == "int":
            if isinstance(values, int):
                self.rngs.append(lambda size: np.repeat([values], repeats=size))
            if isinstance(values, list):
                self.rngs.append(
                    lambda size: rng.integers(values[0], values[1], size=size)
                )
        elif space == "cat":
            if isinstance(values, str):
                self.rngs.append(lambda size: np.repeat([values], repeats=size))
            if isinstance(values, (list, tuple)):
                self.rngs.append(
                    lambda size: rng.choice(values, size=size, replace=True)
                )
        else:
            raise ValueError(
                f'For arg "values" only "real", "int", "cat" are allowed. {values} provided'
            )

        if len(self.rngs) == rng_len:
            raise ValueError("Wrong parameters provided.")

        self.names.append(name)

    def pick(self, size: int = 1) -> Dict[str, np.ndarray[any]]:
        """
        Generate random point(s) from the grid.
        Args:
            size (int, optional): Number of points. Defaults to 1.

        Returns:
            Dict[str, np.ndarray[any]]: List of points from grid.
        """
        random_coordinates = [rng(size=size) for rng in self.rngs]
        dict_coordinates = {
            name: coordinates
            for name, coordinates in zip(self.names, random_coordinates)
        }
        return dict_coordinates

    def __lognuniform(self, low=0, high=1, size=None, base=np.e):
        rng = np.random.default_rng(seed=self.init_seed)
        rng_ = partial(rng.uniform, **{"low": low, "high": high})

        return np.power(base, rng_(size=size)) / base


class ConditionalGrid:
    pass
