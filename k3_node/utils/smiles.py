import re
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from keras import ops


x_map: Dict[str, List[Any]] = {
    "atomic_num": list(range(0, 119)),
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "CHI_TETRAHEDRAL",
        "CHI_ALLENE",
        "CHI_SQUAREPLANAR",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
    ],
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 7)),
    "num_hs": list(range(0, 9)),
    "num_radical_electrons": list(range(0, 5)),
    "hybridization": [
        "UNSPECIFIED",
        "S",
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "OTHER",
    ],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}

e_map: Dict[str, List[Any]] = {
    "bond_type": [
        "UNSPECIFIED",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "QUADRUPLE",
        "QUINTUPLE",
        "HEXTUPLE",
        "ONEANDAHALF",
        "TWOANDAHALF",
        "THREEANDAHALF",
        "FOURANDAHALF",
        "FIVEANDAHALF",
        "AROMATIC",
        "IONIC",
        "HYDROGEN",
        "THREECENTER",
        "DATIVEONE",
        "DATIVE",
        "DATIVEL",
        "DATIVER",
        "OTHER",
        "ZERO",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOANY",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
    ],
    "is_conjugated": [False, True],
}


def from_rdmol(mol: Any) -> "Any":
    r"""Converts an :class:`rdkit.Chem.Mol` instance to a :class:`k3_node.data.Data` instance."""
    from rdkit import Chem
    from k3_node.data.data import Data

    assert isinstance(mol, Chem.Mol)

    xs: List[List[int]] = []
    for atom in mol.GetAtoms():
        row: List[int] = []
        row.append(x_map["atomic_num"].index(atom.GetAtomicNum()))
        row.append(x_map["chirality"].index(str(atom.GetChiralTag())))
        row.append(x_map["degree"].index(atom.GetTotalDegree()))
        row.append(x_map["formal_charge"].index(atom.GetFormalCharge()))
        row.append(x_map["num_hs"].index(atom.GetTotalNumHs()))
        row.append(x_map["num_radical_electrons"].index(atom.GetNumRadicalElectrons()))
        row.append(x_map["hybridization"].index(str(atom.GetHybridization())))
        row.append(x_map["is_aromatic"].index(atom.GetIsAromatic()))
        row.append(x_map["is_in_ring"].index(atom.IsInRing()))
        xs.append(row)

    if len(xs) > 0:
        x_np = np.array(xs, dtype=np.int64).reshape(-1, 9)
    else:
        x_np = np.empty((0, 9), dtype=np.int64)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map["bond_type"].index(str(bond.GetBondType())))
        e.append(e_map["stereo"].index(str(bond.GetStereo())))
        e.append(e_map["is_conjugated"].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    if len(edge_indices) > 0:
        edge_index_np = np.array(edge_indices, dtype=np.int64).T.reshape(2, -1)
        edge_attr_np = np.array(edge_attrs, dtype=np.int64).reshape(-1, 3)

        # Sort indices matching PyG canonical ordering
        perm = (edge_index_np[0] * x_np.shape[0] + edge_index_np[1]).argsort()
        edge_index_np = edge_index_np[:, perm]
        edge_attr_np = edge_attr_np[perm]
    else:
        edge_index_np = np.empty((2, 0), dtype=np.int64)
        edge_attr_np = np.empty((0, 3), dtype=np.int64)

    x = ops.convert_to_tensor(x_np, dtype="int64")
    edge_index = ops.convert_to_tensor(edge_index_np, dtype="int64")
    edge_attr = ops.convert_to_tensor(edge_attr_np, dtype="int64")

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def from_smiles(
    smiles: str,
    with_hydrogen: bool = False,
    kekulize: bool = False,
) -> Any:
    r"""Converts a SMILES string to a :class:`k3_node.data.Data` instance."""
    try:
        from rdkit import Chem, RDLogger
    except ImportError as e:
        raise ImportError(
            "from_smiles requires 'rdkit'. Please install it via 'pip install rdkit'."
        ) from e

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles("")
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    data = from_rdmol(mol)
    data.smiles = smiles
    return data


def to_rdmol(
    data: Any,
    kekulize: bool = False,
) -> Any:
    r"""Converts a :class:`k3_node.data.Data` instance to an :class:`rdkit.Chem.Mol` instance."""
    try:
        from rdkit import Chem
    except ImportError as e:
        raise ImportError(
            "to_rdmol requires 'rdkit'. Please install it via 'pip install rdkit'."
        ) from e

    mol = Chem.RWMol()

    assert data.x is not None
    assert data.num_nodes is not None
    assert data.edge_index is not None
    assert data.edge_attr is not None

    x_np = ops.convert_to_numpy(data.x)
    edge_index_np = ops.convert_to_numpy(data.edge_index)
    edge_attr_np = ops.convert_to_numpy(data.edge_attr)

    for i in range(data.num_nodes):
        atom = Chem.Atom(int(x_np[i, 0]))
        atom.SetChiralTag(Chem.rdchem.ChiralType.values[int(x_np[i, 1])])
        atom.SetFormalCharge(x_map["formal_charge"][int(x_np[i, 3])])
        atom.SetNumExplicitHs(x_map["num_hs"][int(x_np[i, 4])])
        atom.SetNumRadicalElectrons(x_map["num_radical_electrons"][int(x_np[i, 5])])
        atom.SetHybridization(Chem.rdchem.HybridizationType.values[int(x_np[i, 6])])
        atom.SetIsAromatic(bool(x_np[i, 7]))
        mol.AddAtom(atom)

    edges = [tuple(edge_index_np[:, idx]) for idx in range(edge_index_np.shape[1])]
    visited = set()

    for idx, (src, dst) in enumerate(edges):
        src, dst = int(src), int(dst)
        if tuple(sorted((src, dst))) in visited:
            continue

        bond_type = Chem.BondType.values[int(edge_attr_np[idx, 0])]
        mol.AddBond(src, dst, bond_type)

        stereo = Chem.rdchem.BondStereo.values[int(edge_attr_np[idx, 1])]
        if stereo != Chem.rdchem.BondStereo.STEREONONE:
            db = mol.GetBondBetweenAtoms(src, dst)
            db.SetStereoAtoms(dst, src)
            db.SetStereo(stereo)

        is_conjugated = bool(edge_attr_np[idx, 2])
        mol.GetBondBetweenAtoms(src, dst).SetIsConjugated(is_conjugated)

        visited.add(tuple(sorted((src, dst))))

    mol = mol.GetMol()
    if kekulize:
        Chem.Kekulize(mol)

    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol)
    return mol


def to_smiles(
    data: Any,
    kekulize: bool = False,
) -> str:
    r"""Converts a :class:`k3_node.data.Data` instance to a SMILES string."""
    from rdkit import Chem

    mol = to_rdmol(data, kekulize=kekulize)
    return Chem.MolToSmiles(mol, isomericSmiles=True)
