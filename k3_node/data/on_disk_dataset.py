import os
import os.path as osp
from typing import Any, Callable, List, Optional, Union

from k3_node.data.data import BaseData
from k3_node.data.database import Database, RocksDatabase, SQLiteDatabase, Schema
from k3_node.data.dataset import Dataset


class OnDiskDataset(Dataset):
    """Dataset base class for out-of-core graph datasets using a Database backend."""

    BACKENDS = {
        "sqlite": SQLiteDatabase,
        "rocksdb": RocksDatabase,
    }

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        backend: str = "sqlite",
        schema: Schema = object,
        log: bool = True,
    ):
        if backend not in self.BACKENDS:
            raise ValueError(f"Database backend must be one of {set(self.BACKENDS.keys())}, got '{backend}'")

        self.backend = backend
        self.schema = schema
        self._db: Optional[Database] = None

        super().__init__(root, transform, pre_filter=pre_filter, log=log)

    @property
    def processed_file_names(self) -> str:
        return f"{self.backend}.db"

    @property
    def db(self) -> Database:
        if self._db is not None:
            return self._db

        cls = self.BACKENDS[self.backend]
        os.makedirs(self.processed_dir, exist_ok=True)
        path = osp.join(self.processed_dir, self.processed_file_names)
        self._db = cls(path=path, schema=self.schema)
        return self._db

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None

    def serialize(self, data: BaseData) -> Any:
        return data

    def deserialize(self, data: Any) -> BaseData:
        return data

    def len(self) -> int:
        return len(self.db)

    def get(self, idx: int) -> BaseData:
        return self.deserialize(self.db[idx])

    def append(self, data: BaseData):
        idx = len(self)
        self.db[idx] = self.serialize(data)

    def extend(self, data_list: List[BaseData]):
        start = len(self)
        indices = list(range(start, start + len(data_list)))
        serialized = [self.serialize(d) for d in data_list]
        self.db.multi_insert(indices, serialized)

