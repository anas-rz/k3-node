import os
import os.path as osp
import re
import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from keras import ops

from k3_node.data import InMemoryDataset, download_url, extract_gz
from k3_node.utils.smiles import from_smiles as default_from_smiles


class MoleculeNet(InMemoryDataset):
    r"""The `MoleculeNet <http://moleculenet.org/datasets-1>`_ benchmark collection
    from the `"MoleculeNet: A Benchmark for Molecular Machine Learning"
    <https://arxiv.org/abs/1703.00564>`_ paper, containing datasets from physical
    chemistry, biophysics and physiology.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"ESOL"`, :obj:`"FreeSolv"`,
            :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`, :obj:`"HIV"`,
            :obj:`"BACE"`, :obj:`"BBBP"`, :obj:`"Tox21"`, :obj:`"ToxCast"`,
            :obj:`"SIDER"`, :obj:`"ClinTox"`).
        transform (callable, optional): A function/transform that takes in a
            :obj:`k3_node.data.Data` object and returns a transformed version.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in a
            :obj:`k3_node.data.Data` object and returns a transformed version.
            (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :obj:`k3_node.data.Data` object and returns a boolean value,
            indicating whether the data object should be included in the final
            dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        from_smiles (callable, optional): A custom function that takes a SMILES
            string and outputs a :obj:`k3_node.data.Data` object.
            (default: :obj:`None`)
    """

    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}"

    # Format: name: (display_name, url_name, csv_name, smiles_idx, y_idx)
    names: Dict[str, Tuple[str, str, str, int, Union[int, slice]]] = {
        "esol": ("ESOL", "delaney-processed.csv", "delaney-processed", -1, -2),
        "freesolv": ("FreeSolv", "SAMPL.csv", "SAMPL", 1, 2),
        "lipo": ("Lipophilicity", "Lipophilicity.csv", "Lipophilicity", 2, 1),
        "pcba": ("PCBA", "pcba.csv.gz", "pcba", -1, slice(0, 128)),
        "muv": ("MUV", "muv.csv.gz", "muv", -1, slice(0, 17)),
        "hiv": ("HIV", "HIV.csv", "HIV", 0, -1),
        "bace": ("BACE", "bace.csv", "bace", 0, 2),
        "bbbp": ("BBBP", "BBBP.csv", "BBBP", -1, -2),
        "tox21": ("Tox21", "tox21.csv.gz", "tox21", -1, slice(0, 12)),
        "toxcast": ("ToxCast", "toxcast_data.csv.gz", "toxcast_data", 0, slice(1, 618)),
        "sider": ("SIDER", "sider.csv.gz", "sider", 0, slice(1, 28)),
        "clintox": ("ClinTox", "clintox.csv.gz", "clintox", 0, slice(1, 3)),
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        from_smiles: Optional[Callable] = None,
    ) -> None:
        self.name = name.lower()
        if self.name not in self.names:
            raise ValueError(
                f"Unknown dataset name '{name}'. Available names: {list(self.names.keys())}"
            )
        self.from_smiles = from_smiles or default_from_smiles
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            force_reload=force_reload,
        )
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> str:
        return f"{self.names[self.name][2]}.csv"

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1].endswith("gz"):
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self) -> None:
        with open(self.raw_paths[0], "r", encoding="utf-8") as f:
            dataset = f.read().split("\n")[1:-1]
            dataset = [x for x in dataset if len(x) > 0]

        data_list = []
        for line in dataset:
            line = re.sub(r'".*"', "", line)  # Replace quoted substrings
            values = line.split(",")

            smiles = values[self.names[self.name][3]]
            labels = values[self.names[self.name][4]]
            labels = labels if isinstance(labels, list) else [labels]

            ys = [float(y) if len(y) > 0 else float("nan") for y in labels]
            y = ops.convert_to_tensor(np.array(ys, dtype=np.float32).reshape(1, -1), dtype="float32")

            data = self.from_smiles(smiles)
            data.y = y

            if data.num_nodes == 0:
                warnings.warn(
                    f"Skipping molecule '{smiles}' since it resulted in zero atoms",
                    stacklevel=2,
                )
                continue

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.names[self.name][0]}({len(self)})"

