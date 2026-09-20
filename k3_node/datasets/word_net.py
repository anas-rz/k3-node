from typing import Callable, List, Optional
import numpy as np
from keras import ops

from k3_node.data import Data, InMemoryDataset
from k3_node.io import fs


class WordNet18(InMemoryDataset):
    r"""The WordNet18 dataset containing 40,943 entities, 18 relations and 151,442 fact triplets."""

    url = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18/original"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["train.txt", "valid.txt", "test.txt"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        for filename in self.raw_file_names:
            fs.cp(f"{self.url}/{filename}", self.raw_dir)

    def process(self):
        srcs, dsts, edge_types = [], [], []
        for path in self.raw_paths:
            with open(path) as f:
                edges = [int(x) for x in f.read().split()[1:]]
                edge = np.array(edges, dtype=np.int64)
                srcs.append(edge[::3])
                dsts.append(edge[1::3])
                edge_types.append(edge[2::3])

        src = np.concatenate(srcs, axis=0)
        dst = np.concatenate(dsts, axis=0)
        edge_type = np.concatenate(edge_types, axis=0)

        n_train = len(srcs[0])
        n_val = len(srcs[1])
        n_test = len(srcs[2])

        train_mask = np.zeros(len(src), dtype=bool)
        train_mask[:n_train] = True
        val_mask = np.zeros(len(src), dtype=bool)
        val_mask[n_train : n_train + n_val] = True
        test_mask = np.zeros(len(src), dtype=bool)
        test_mask[n_train + n_val :] = True

        num_nodes = int(max(src.max(), dst.max())) + 1
        perm = np.argsort(num_nodes * src + dst)

        edge_index = np.stack([src[perm], dst[perm]], axis=0)
        edge_type = edge_type[perm]
        train_mask = train_mask[perm]
        val_mask = val_mask[perm]
        test_mask = test_mask[perm]

        data = Data(
            edge_index=ops.convert_to_tensor(edge_index, dtype="int64"),
            edge_type=ops.convert_to_tensor(edge_type, dtype="int64"),
            train_mask=ops.convert_to_tensor(train_mask, dtype="bool"),
            val_mask=ops.convert_to_tensor(val_mask, dtype="bool"),
            test_mask=ops.convert_to_tensor(test_mask, dtype="bool"),
            num_nodes=num_nodes,
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])


class WordNet18RR(InMemoryDataset):
    r"""The WordNet18RR dataset."""

    url = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original"

    edge2id = {
        "_also_see": 0,
        "_derivationally_related_form": 1,
        "_has_part": 2,
        "_hypernym": 3,
        "_instance_hypernym": 4,
        "_member_meronym": 5,
        "_member_of_domain_region": 6,
        "_member_of_domain_usage": 7,
        "_similar_to": 8,
        "_synset_domain_topic_of": 9,
        "_verb_group": 10,
    }

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["train.txt", "valid.txt", "test.txt"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        for filename in self.raw_file_names:
            fs.cp(f"{self.url}/{filename}", self.raw_dir)

    def process(self):
        srcs, dsts, edge_types = [], [], []
        for path in self.raw_paths:
            with open(path) as f:
                lines = f.read().split("\n")[:-1]
                src = [int(line.split()[0]) for line in lines]
                rel = [self.edge2id[line.split()[1]] for line in lines]
                dst = [int(line.split()[2]) for line in lines]
                srcs.append(np.array(src, dtype=np.int64))
                dsts.append(np.array(dst, dtype=np.int64))
                edge_types.append(np.array(rel, dtype=np.int64))

        src = np.concatenate(srcs, axis=0)
        dst = np.concatenate(dsts, axis=0)
        edge_type = np.concatenate(edge_types, axis=0)

        n_train = len(srcs[0])
        n_val = len(srcs[1])

        train_mask = np.zeros(len(src), dtype=bool)
        train_mask[:n_train] = True
        val_mask = np.zeros(len(src), dtype=bool)
        val_mask[n_train : n_train + n_val] = True
        test_mask = np.zeros(len(src), dtype=bool)
        test_mask[n_train + n_val :] = True

        num_nodes = int(max(src.max(), dst.max())) + 1
        perm = np.argsort(num_nodes * src + dst)

        edge_index = np.stack([src[perm], dst[perm]], axis=0)
        edge_type = edge_type[perm]
        train_mask = train_mask[perm]
        val_mask = val_mask[perm]
        test_mask = test_mask[perm]

        data = Data(
            edge_index=ops.convert_to_tensor(edge_index, dtype="int64"),
            edge_type=ops.convert_to_tensor(edge_type, dtype="int64"),
            train_mask=ops.convert_to_tensor(train_mask, dtype="bool"),
            val_mask=ops.convert_to_tensor(val_mask, dtype="bool"),
            test_mask=ops.convert_to_tensor(test_mask, dtype="bool"),
            num_nodes=num_nodes,
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

