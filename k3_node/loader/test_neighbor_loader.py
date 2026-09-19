import numpy as np

try:
    import torch
except ImportError:
    torch = None

from k3_node.data import Data, HeteroData
from k3_node.loader import HGTLoader, LinkNeighborLoader, NeighborLoader


def get_homo_graph():
    # 6 nodes connected in a ring: 0->1->2->3->4->5->0 and some cross edges
    edge_index = np.array([
        [0, 1, 2, 3, 4, 5, 0, 2],
        [1, 2, 3, 4, 5, 0, 3, 5],
    ], dtype=np.int64)
    x = np.random.randn(6, 16).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)

    if torch is not None:
        edge_index = torch.from_numpy(edge_index)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

    return Data(x=x, edge_index=edge_index, y=y)


def get_hetero_graph():
    data = HeteroData()
    data['paper'].x = np.random.randn(10, 8).astype(np.float32)
    data['author'].x = np.random.randn(5, 8).astype(np.float32)

    data['author', 'writes', 'paper'].edge_index = np.array([
        [0, 1, 2, 3, 4, 0, 1],
        [0, 1, 2, 3, 4, 5, 6],
    ], dtype=np.int64)
    data['paper', 'cites', 'paper'].edge_index = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ], dtype=np.int64)

    if torch is not None:
        data['paper'].x = torch.from_numpy(data['paper'].x)
        data['author'].x = torch.from_numpy(data['author'].x)
        data['author', 'writes', 'paper'].edge_index = torch.from_numpy(data['author', 'writes', 'paper'].edge_index)
        data['paper', 'cites', 'paper'].edge_index = torch.from_numpy(data['paper', 'cites', 'paper'].edge_index)

    return data


def test_homo_neighbor_loader():
    data = get_homo_graph()
    input_nodes = [0, 1]
    if torch is not None:
        input_nodes = torch.tensor(input_nodes, dtype=torch.long)

    loader = NeighborLoader(
        data,
        num_neighbors=[2, 2],
        batch_size=2,
        input_nodes=input_nodes,
        shuffle=False,
    )

    batch = next(iter(loader))
    assert batch.batch_size == 2
    assert hasattr(batch, 'n_id')
    assert hasattr(batch, 'e_id')
    assert hasattr(batch, 'num_sampled_nodes')
    assert hasattr(batch, 'num_sampled_edges')
    assert batch.x.shape[0] == len(batch.n_id)
    assert batch.edge_index.shape[0] == 2


def test_hetero_neighbor_loader():
    data = get_hetero_graph()
    loader = NeighborLoader(
        data,
        num_neighbors=[2, 2],
        batch_size=2,
        input_nodes=('paper', [0, 1]),
        shuffle=False,
    )

    batch = next(iter(loader))
    assert batch['paper'].batch_size == 2
    assert hasattr(batch['paper'], 'n_id')
    assert hasattr(batch['author', 'writes', 'paper'], 'edge_index')


def test_link_neighbor_loader():
    data = get_homo_graph()
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[2, 2],
        batch_size=2,
        neg_sampling_ratio=1.0,
        shuffle=False,
    )

    batch = next(iter(loader))
    assert hasattr(batch, 'edge_label_index')
    assert hasattr(batch, 'edge_label')
    assert hasattr(batch, 'n_id')
    assert batch.edge_label.shape[0] == 4  # 2 positive + 2 negative


def test_hgt_loader():
    data = get_hetero_graph()
    loader = HGTLoader(
        data,
        num_samples=[4, 4],
        input_nodes=('paper', [0, 1]),
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(loader))
    assert batch['paper'].batch_size == 2
    assert hasattr(batch['paper'], 'n_id')

