import os
import tempfile
import numpy as np
import pytest
from keras import ops

from k3_node.data import (
    Data,
    EdgeAttr,
    EdgeLayout,
    FeatureStore,
    GraphStore,
    InMemoryDataset,
    OnDiskDataset,
    SQLiteDatabase,
    TensorAttr,
)


class MyInMemoryDataset(InMemoryDataset):
    def __init__(self, data_list, root=None):
        super().__init__(root)
        self._data_list = data_list


def test_in_memory_dataset():
    d1 = Data(x=ops.convert_to_tensor([[1.0]]), edge_index=ops.convert_to_tensor([[0], [0]], dtype="int64"))
    d2 = Data(x=ops.convert_to_tensor([[2.0]]), edge_index=ops.convert_to_tensor([[0], [0]], dtype="int64"))

    ds = MyInMemoryDataset([d1, d2])
    assert len(ds) == 2
    assert ops.convert_to_numpy(ds[0].x).item() == 1.0
    assert ops.convert_to_numpy(ds[1].x).item() == 2.0

    # Slicing
    sub_ds = ds[1:]
    assert len(sub_ds) == 1
    assert ops.convert_to_numpy(sub_ds[0].x).item() == 2.0


def test_sqlite_database_and_on_disk_dataset():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        db = SQLiteDatabase(path=db_path)
        d1 = Data(x=ops.convert_to_tensor([[1.0]]))
        d2 = Data(x=ops.convert_to_tensor([[2.0]]))
        db[0] = d1
        db[1] = d2
        assert len(db) == 2
        rec1 = db[0]
        assert ops.convert_to_numpy(rec1.x).item() == 1.0
        db.close()

        # Test OnDiskDataset
        ds = OnDiskDataset(root=tmp_dir, backend="sqlite")
        ds.append(d1)
        ds.append(d2)
        assert len(ds) == 2
        rec = ds[0]
        assert ops.convert_to_numpy(rec.x).item() == 1.0
        ds.close()


class SimpleFeatureStore(FeatureStore):
    def _put_tensor(self, tensor, attr):
        self._feat_dict[(attr.group_name, attr.attr_name)] = tensor
        return True

    def _get_tensor(self, attr):
        return self._feat_dict.get((attr.group_name, attr.attr_name))

    def _remove_tensor(self, attr):
        return self._feat_dict.pop((attr.group_name, attr.attr_name), None) is not None


class SimpleGraphStore(GraphStore):
    def __init__(self):
        self._store = {}

    def _put_edge_index(self, edge_index, edge_attr):
        self._store[(edge_attr.edge_type, edge_attr.layout)] = edge_index
        return True

    def _get_edge_index(self, edge_attr):
        return self._store.get((edge_attr.edge_type, edge_attr.layout))

    def _remove_edge_index(self, edge_attr):
        return self._store.pop((edge_attr.edge_type, edge_attr.layout), None) is not None


def test_feature_and_graph_store():
    fs = SimpleFeatureStore()
    x = ops.convert_to_tensor([[1.0, 2.0]])
    fs.put_tensor(x, group_name="user", attr_name="feat")
    ret = fs.get_tensor(group_name="user", attr_name="feat")
    assert ops.convert_to_numpy(ret).shape == (1, 2)

    gs = SimpleGraphStore()
    ei = ops.convert_to_tensor([[0, 1], [1, 0]], dtype="int64")
    gs.put_edge_index(ei, edge_type=("u", "follows", "u"), layout=EdgeLayout.COO)
    ret_ei = gs.get_edge_index(edge_type=("u", "follows", "u"), layout=EdgeLayout.COO)
    assert ops.convert_to_numpy(ret_ei).shape == (2, 2)

