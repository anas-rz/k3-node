from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class _FieldStatus(Enum):
    UNSET = None


@dataclass
class TensorAttr:
    """Defines the attributes of a FeatureStore tensor."""

    group_name: Optional[Any] = _FieldStatus.UNSET
    attr_name: Optional[str] = _FieldStatus.UNSET
    index: Optional[Any] = _FieldStatus.UNSET

    def is_set(self, key: str) -> bool:
        return getattr(self, key) != _FieldStatus.UNSET

    def is_fully_specified(self) -> bool:
        return all(self.is_set(k) for k in ("group_name", "attr_name", "index"))


class FeatureStore(ABC):
    """Abstract base class for feature stores."""

    def __init__(self):
        self._feat_dict: Dict[Tuple[Any, str], Any] = {}

    @abstractmethod
    def _put_tensor(self, tensor: Any, attr: TensorAttr) -> bool:
        pass

    @abstractmethod
    def _get_tensor(self, attr: TensorAttr) -> Optional[Any]:
        pass

    @abstractmethod
    def _remove_tensor(self, attr: TensorAttr) -> bool:
        pass

    def put_tensor(self, tensor: Any, group_name: Any = None, attr_name: Optional[str] = None, index: Any = None) -> bool:
        attr = TensorAttr(group_name=group_name, attr_name=attr_name, index=index)
        return self._put_tensor(tensor, attr)

    def get_tensor(self, group_name: Any = None, attr_name: Optional[str] = None, index: Any = None) -> Optional[Any]:
        attr = TensorAttr(group_name=group_name, attr_name=attr_name, index=index)
        return self._get_tensor(attr)

    def remove_tensor(self, group_name: Any = None, attr_name: Optional[str] = None, index: Any = None) -> bool:
        attr = TensorAttr(group_name=group_name, attr_name=attr_name, index=index)
        return self._remove_tensor(attr)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple):
            group_name, attr_name = key[:2]
            index = key[2] if len(key) > 2 else None
            return self.get_tensor(group_name=group_name, attr_name=attr_name, index=index)
        return self.get_tensor(group_name=key)

    def __setitem__(self, key: Any, value: Any):
        if isinstance(key, tuple):
            group_name, attr_name = key[:2]
            index = key[2] if len(key) > 2 else None
            self.put_tensor(value, group_name=group_name, attr_name=attr_name, index=index)
        else:
            self.put_tensor(value, group_name=key)

