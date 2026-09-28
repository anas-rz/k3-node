import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np

from k3_node.data import Data, InMemoryDataset
from k3_node.data.download import download_url
from k3_node.data.extract import extract_zip

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414
CONVERSION = np.array([1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
                       1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.], dtype=np.float32)
ATOMREFS = {
    6: [0., 0., 0., 0., 0.],
    7: [-13.61312172, -1029.86312267, -1485.30251237, -2042.61123593, -2713.48485589],
    8: [-13.5745904, -1029.82456413, -1485.26398105, -2042.5727046, -2713.44632457],
    9: [-13.54887564, -1029.79887659, -1485.2382935, -2042.54701705, -2713.42063702],
    10: [-13.90303183, -1030.25891228, -1485.71166277, -2043.01812778, -2713.88796536],
    11: [0., 0., 0., 0., 0.],
}


class QM9(InMemoryDataset):
    r"""The QM9 dataset: about 130,000 small organic molecules with their 3D structure and 19
    regression targets (dipole moment, HOMO/LUMO energies, internal energy, ...), as in PyG.

    Every molecule has 11 atom features ``x`` (one-hot H/C/N/O/F, atomic number, aromatic,
    sp/sp2/sp3 hybridization, number of hydrogens), atomic numbers ``z``, positions ``pos``,
    one-hot bond types ``edge_attr`` and the targets ``y`` of shape ``[1, 19]`` (energies in eV).
    Requires RDKit to process the raw files.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function applied to each graph when it is accessed.
        pre_transform (callable, optional): A function applied to each graph before saving.
        pre_filter (callable, optional): A function deciding which graphs to keep.
        force_reload (bool, optional): Whether to re-process the dataset. (default: ``False``)
    """

    raw_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip'
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'

    def __init__(self, root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, force_reload: bool = False):
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        return float(np.asarray(self._data.y)[:, target].mean())

    def std(self, target: int) -> float:
        return float(np.asarray(self._data.y)[:, target].std(ddof=1))

    def atomref(self, target: int) -> Optional[np.ndarray]:
        r"""Per-element reference energies (a ``[100, 1]`` array indexed by atomic number), or
        ``None`` for targets without them."""
        if target not in ATOMREFS:
            return None
        out = np.zeros((100, 1), dtype=np.float32)
        out[[1, 6, 7, 8, 9], 0] = ATOMREFS[target]
        return out

    @property
    def raw_file_names(self) -> List[str]:
        return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data_v3.pt'

    def download(self):
        path = download_url(self.raw_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        download_url(self.raw_url2, self.raw_dir)
        os.rename(osp.join(self.raw_dir, '3195404'), osp.join(self.raw_dir, 'uncharacterized.txt'))

    def process(self):
        from rdkit import Chem, RDLogger
        from rdkit.Chem.rdchem import BondType as BT
        from rdkit.Chem.rdchem import HybridizationType as HT

        RDLogger.DisableLog('rdApp.*')
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1]) as f:
            target = np.array([[float(x) for x in line.split(',')[1:20]] for line in f.read().split('\n')[1:-1]],
                              dtype=np.float32)
        target = np.concatenate([target[:, 3:], target[:, :3]], axis=-1) * CONVERSION
        with open(self.raw_paths[2]) as f:
            skip = {int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]}

        data_list = []
        for i, mol in enumerate(Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)):
            if i in skip:
                continue
            N = mol.GetNumAtoms()
            pos = mol.GetConformer().GetPositions().astype(np.float32)
            atoms = list(mol.GetAtoms())
            z = np.array([a.GetAtomicNum() for a in atoms], dtype=np.int64)
            rows, cols, edge_types = [], [], []
            for bond in mol.GetBonds():
                s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [s, e]
                cols += [e, s]
                edge_types += 2 * [bonds[bond.GetBondType()]]
            edge_index = np.array([rows, cols], dtype=np.int64).reshape(2, -1)
            edge_type = np.array(edge_types, dtype=np.int64)
            perm = np.argsort(edge_index[0] * N + edge_index[1], kind="stable")
            edge_index, edge_type = edge_index[:, perm], edge_type[perm]

            num_hs = np.zeros(N, dtype=np.float32)
            np.add.at(num_hs, edge_index[1], (z == 1).astype(np.float32)[edge_index[0]])
            hybrid = [a.GetHybridization() for a in atoms]
            x = np.concatenate([
                np.eye(len(types), dtype=np.float32)[[types[a.GetSymbol()] for a in atoms]],
                np.stack([z, [a.GetIsAromatic() for a in atoms], [h == HT.SP for h in hybrid],
                          [h == HT.SP2 for h in hybrid], [h == HT.SP3 for h in hybrid], num_hs], axis=1).astype(np.float32),
            ], axis=1)
            data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
                        edge_attr=np.eye(len(bonds), dtype=np.float32)[edge_type].reshape(-1, len(bonds)),
                        y=target[i][None], smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
                        name=mol.GetProp('_Name'), idx=i)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        self.save(data_list, self.processed_paths[0])
