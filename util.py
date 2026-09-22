from edtrace import link
from typing import Callable, Any
from altair import Chart, Data
import functools
import numpy as np
import random
import torch


def article_link(url):
    return link(url, title="[article]")


def make_plot(title: str,
              xlabel: str,
              ylabel: str,
              f: Callable[[float], float] | None,
              xrange: tuple[float, float] = (-3, 3),
              points: list[dict] | None = None) -> dict:
    to_show = []

    if f is not None:
        values = [{xlabel: x, ylabel: f(x)} for x in np.linspace(xrange[0], xrange[1], 30)]
        line = Chart(Data(values=values)).properties(title=title).mark_line().encode(x=f"{xlabel}:Q", y=f"{ylabel}:Q")
        to_show.append(line)

    if points is not None:
        points = Chart(Data(values=points)).mark_point().encode(x=f"{xlabel}:Q", y=f"{ylabel}:Q", color="color:N")
        to_show.append(points)

    chart = functools.reduce(lambda c1, c2: c1 + c2, to_show)
    return chart.to_dict()


class Vocabulary:
    """Maps strings to integers."""
    def __init__(self):
        self.index_to_string: list[str] = []
        self.string_to_index: dict[str, int] = {}

    def get_index(self, string: str) -> int:  # @inspect string
        index = self.string_to_index.get(string)  # @inspect index
        if index is None:  # New string
            index = len(self.index_to_string)  # @inspect index
            self.index_to_string.append(string)
            self.string_to_index[string] = index
        return index

    def get_string(self, index: int) -> str:
        return self.index_to_string[index]

    def __len__(self):
        return len(self.index_to_string)

    def asdict(self):
        return {
            "index_to_string": self.index_to_string,
            "string_to_index": self.string_to_index,
        }


def set_random_seed(seed: int):
    """Set all random seeds for deterministic behavior."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def one_hot(index: int, length: int) -> torch.Tensor:
    """Create a one-hot vector of the given `length` with a 1 at the `index` position."""
    vector = torch.zeros(length)
    vector[index] = 1
    return vector


def sample_dict(choices: dict[Any, float]) -> Any:
    """Sample a key from a dictionary of choices based on their probabilities (values)."""
    return np.random.choice(list(choices.keys()), p=list(choices.values()))


def normalize_dict(choices: dict[Any, float]) -> dict[Any, float]:
    """Normalize a dictionary of choices based on their probabilities (values)."""
    total_prob = sum(choices.values())
    return {key: prob / total_prob for key, prob in choices.items()}