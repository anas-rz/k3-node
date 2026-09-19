import copy
from abc import ABC, abstractmethod
from typing import Any, Callable


class BaseTransform(ABC):
    r"""An abstract base class for writing transforms.

    Transforms are a general way to modify and customize
    :class:`~k3_node.data.Data` or :class:`~k3_node.data.HeteroData` objects.
    """

    def __call__(self, data: Any) -> Any:
        # Shallow-copy the data to prevent in-place modification of caller's object
        return self.forward(copy.copy(data))

    @abstractmethod
    def forward(self, data: Any) -> Any:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def functional_transform(name: str) -> Callable:
    r"""Decorator for functional transforms."""

    def wrapper(cls: Any) -> Any:
        return cls

    return wrapper

