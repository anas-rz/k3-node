import io
import pickle
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

Schema = Any


class Database(ABC):
    """Base class for key/value and index-based graph databases."""

    def __init__(self, schema: Schema = object):
        self.schema = schema

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def insert(self, index: int, data: Any):
        pass

    def multi_insert(self, indices: Sequence[int], data_list: Sequence[Any]):
        for idx, d in zip(indices, data_list):
            self.insert(idx, d)

    @abstractmethod
    def get(self, index: int) -> Any:
        pass

    def multi_get(self, indices: Sequence[int]) -> List[Any]:
        return [self.get(idx) for idx in indices]

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, int):
            return self.get(idx)
        elif isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop or len(self)
            step = idx.step or 1
            return self.multi_get(range(start, stop, step))
        elif isinstance(idx, (list, tuple)):
            return self.multi_get(idx)
        else:
            return self.get(int(idx))

    def __setitem__(self, idx: Any, value: Any):
        if isinstance(idx, int):
            self.insert(idx, value)
        elif isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop or len(self)
            step = idx.step or 1
            indices = list(range(start, stop, step))
            self.multi_insert(indices, value)
        elif isinstance(idx, (list, tuple)):
            self.multi_insert(idx, value)
        else:
            self.insert(int(idx), value)


class SQLiteDatabase(Database):
    """SQLite-backed persistent database."""

    def __init__(self, path: str, name: str = "data", schema: Schema = object):
        super().__init__(schema)
        self.path = path
        self.name = name
        self._conn: Optional[sqlite3.Connection] = None
        self.connect()

    def connect(self):
        self._conn = sqlite3.connect(self.path)
        with self._conn:
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.name} (id INTEGER PRIMARY KEY, val BLOB)"
            )

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def insert(self, index: int, data: Any):
        buf = io.BytesIO()
        pickle.dump(data, buf)
        raw = buf.getvalue()
        with self._conn:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {self.name} (id, val) VALUES (?, ?)",
                (index, raw),
            )

    def multi_insert(self, indices: Sequence[int], data_list: Sequence[Any]):
        rows = []
        for idx, d in zip(indices, data_list):
            buf = io.BytesIO()
            pickle.dump(d, buf)
            rows.append((idx, buf.getvalue()))
        with self._conn:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO {self.name} (id, val) VALUES (?, ?)",
                rows,
            )

    def get(self, index: int) -> Any:
        cursor = self._conn.execute(
            f"SELECT val FROM {self.name} WHERE id = ?", (index,)
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Index {index} not found in database")
        return pickle.loads(row[0])

    def multi_get(self, indices: Sequence[int]) -> List[Any]:
        return [self.get(idx) for idx in indices]

    def __len__(self) -> int:
        cursor = self._conn.execute(f"SELECT COUNT(*) FROM {self.name}")
        return cursor.fetchone()[0]


class RocksDatabase(Database):
    """RocksDB-backed database stub."""

    def __init__(self, path: str, schema: Schema = object):
        super().__init__(schema)
        self.path = path
        raise NotImplementedError("RocksDatabase requires rocksdb C++ binding; use SQLiteDatabase instead.")

    def connect(self):
        pass

    def close(self):
        pass

    def insert(self, index: int, data: Any):
        pass

    def get(self, index: int) -> Any:
        pass

    def __len__(self) -> int:
        return 0

