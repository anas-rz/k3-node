import logging
import os
import os.path as osp
from collections import Counter
from typing import Any, Callable, List, Optional
import numpy as np
from keras import ops

from k3_node.data import Data, HeteroData, InMemoryDataset
from k3_node.io import fs


class Entities(InMemoryDataset):
    r"""The relational entities networks "AIFB", "MUTAG", "BGS" and "AM"."""

    url = "https://data.dgl.ai/dataset/{}.tgz"

    def __init__(
        self,
        root: str,
        name: str,
        hetero: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.name = name.lower()
        self.hetero = hetero
        assert self.name in ["aifb", "am", "mutag", "bgs"]
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def num_relations(self) -> int:
        return int(ops.convert_to_numpy(self._data.edge_type).max()) + 1

    @property
    def num_classes(self) -> int:
        return int(ops.convert_to_numpy(self._data.train_y).max()) + 1

    @property
    def raw_file_names(self) -> List[str]:
        return [
            f"{self.name}_stripped.nt.gz",
            "completeDataset.tsv",
            "trainingSet.tsv",
            "testSet.tsv",
        ]

    @property
    def processed_file_names(self) -> str:
        return "hetero_data.pt" if self.hetero else "data.pt"

    def download(self):
        tgz_path = osp.join(self.root, f"{self.name}.tgz")
        fs.cp(self.url.format(self.name), tgz_path, extract=True)
        if osp.exists(tgz_path):
            fs.rm(tgz_path)

    def process(self):
        import gzip
        import rdflib as rdf

        graph_file, task_file, train_file, test_file = self.raw_paths

        g = rdf.Graph()
        with gzip.open(graph_file, "rb") as f:
            g.parse(file=f, format="nt")

        freq = Counter(g.predicates())
        relations = sorted(set(g.predicates()), key=lambda p: -freq.get(p, 0))
        subjects = set(g.subjects())
        objects = set(g.objects())
        nodes = list(subjects.union(objects))

        N = len(nodes)
        R = 2 * len(relations)

        relations_dict = {rel: i for i, rel in enumerate(relations)}
        nodes_dict = {str(node): i for i, node in enumerate(nodes)}

        edges = []
        for s, p, o in g.triples((None, None, None)):
            src, dst = nodes_dict[str(s)], nodes_dict[str(o)]
            rel = relations_dict[p]
            edges.append([src, dst, 2 * rel])
            edges.append([dst, src, 2 * rel + 1])

        edge = np.array(edges, dtype=np.int64).T
        sort_key = N * R * edge[0] + R * edge[1] + edge[2]
        perm = np.argsort(sort_key)
        edge = edge[:, perm]

        edge_index, edge_type = edge[:2], edge[2]

        if self.name == "am":
            label_header = "label_cateogory"
            nodes_header = "proxy"
        elif self.name == "aifb":
            label_header = "label_affiliation"
            nodes_header = "person"
        elif self.name == "mutag":
            label_header = "label_mutagenic"
            nodes_header = "bond"
        elif self.name == "bgs":
            label_header = "label_lithogenesis"
            nodes_header = "rock"

        import pandas as pd

        labels_df = pd.read_csv(task_file, sep="\t")
        labels_set = set(labels_df[label_header].values.tolist())
        labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}

        train_labels_df = pd.read_csv(train_file, sep="\t")
        train_indices, train_labels = [], []
        for nod, lab in zip(train_labels_df[nodes_header].values, train_labels_df[label_header].values):
            train_indices.append(nodes_dict[nod])
            train_labels.append(labels_dict[lab])

        train_idx = np.array(train_indices, dtype=np.int64)
        train_y = np.array(train_labels, dtype=np.int64)

        test_labels_df = pd.read_csv(test_file, sep="\t")
        test_indices, test_labels = [], []
        for nod, lab in zip(test_labels_df[nodes_header].values, test_labels_df[label_header].values):
            test_indices.append(nodes_dict[nod])
            test_labels.append(labels_dict[lab])

        test_idx = np.array(test_indices, dtype=np.int64)
        test_y = np.array(test_labels, dtype=np.int64)

        data = Data(
            edge_index=ops.convert_to_tensor(edge_index, dtype="int64"),
            edge_type=ops.convert_to_tensor(edge_type, dtype="int64"),
            train_idx=ops.convert_to_tensor(train_idx, dtype="int64"),
            train_y=ops.convert_to_tensor(train_y, dtype="int64"),
            test_idx=ops.convert_to_tensor(test_idx, dtype="int64"),
            test_y=ops.convert_to_tensor(test_y, dtype="int64"),
            num_nodes=N,
        )

        if self.hetero:
            data = data.to_heterogeneous(node_type_names=["v"])

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name.upper()}{self.__class__.__name__}()"

