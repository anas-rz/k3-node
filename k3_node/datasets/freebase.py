from typing import Callable, List, Optional
import numpy as np
from keras import ops

from k3_node.data import Data, InMemoryDataset
from k3_node.io import fs


class FB15k_237(InMemoryDataset):
    r"""The FB15K237 dataset containing 14,541 entities, 237 relations and 310,116 fact triples.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): "train", "val", or "test". (default: "train")
        transform (callable, optional): Transform function.
        pre_transform (callable, optional): Pre-transform function.
        force_reload (bool, optional): Whether to re-process the dataset.
    """

    url = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split argument (got {split})")
        self.split = split
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        idx = ["train", "val", "test"].index(split)
        self.load(self.processed_paths[idx])

    @property
    def raw_file_names(self) -> List[str]:
        return ["train.txt", "valid.txt", "test.txt"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["train_data.pt", "val_data.pt", "test_data.pt"]

    def download(self):
        for filename in self.raw_file_names:
            fs.cp(f"{self.url}/{filename}", self.raw_dir)

    def process(self):
        # Map entities and relations to integer IDs
        entities, relations = {}, {}
        for path in self.raw_paths:
            with open(path) as f:
                lines = f.read().split("\n")[:-1]
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 3:
                        s, r, d = parts[0], parts[1], parts[2]
                        if s not in entities:
                            entities[s] = len(entities)
                        if d not in entities:
                            entities[d] = len(entities)
                        if r not in relations:
                            relations[r] = len(relations)

        for in_path, out_path in zip(self.raw_paths, self.processed_paths):
            srcs, dsts, rels = [], [], []
            with open(in_path) as f:
                lines = f.read().split("\n")[:-1]
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 3:
                        srcs.append(entities[parts[0]])
                        rels.append(relations[parts[1]])
                        dsts.append(entities[parts[2]])

            edge_index = np.array([srcs, dsts], dtype=np.int64)
            edge_type = np.array(rels, dtype=np.int64)

            data = Data(
                edge_index=ops.convert_to_tensor(edge_index, dtype="int64"),
                edge_type=ops.convert_to_tensor(edge_type, dtype="int64"),
                num_nodes=len(entities),
            )

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            self.save([data], out_path)

