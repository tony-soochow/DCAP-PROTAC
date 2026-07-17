import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pad_sequence


SYBYL_ATOM_TYPE = [
    'C.3', 'C.2', 'C.1', 'C.ar', 'C.cat',
    'N.3', 'N.2', 'N.1', 'N.ar', 'N.am', 'N.pl3', 'N.4',
    'O.3', 'O.2', 'O.co2',
    'S.3', 'S.2', 'S.o', 'S.o2',
    'P.3',
    'F', 'Cl', 'Br', 'I', 'H',
]
NUM_ATOM_TYPES = len(SYBYL_ATOM_TYPE) + 1  # +1 for unknown types
FASTA_CHAR = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
              "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
              "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
              "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}
EDGE_ATTR = {'1': 1, '2': 2, '3': 3, 'ar': 4, 'am': 5}

DEFAULT_PLM_DIM = 1280
DEFAULT_LINKER_CONF_DIM = 64


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def mol2graph(path, ATOM_TYPE, geometric=False):
    with open(path) as f:
        lines = f.readlines()
    try:
        atom_end_line = lines.index('@<TRIPOS>UNITY_ATOM_ATTR\n')
    except ValueError:
        atom_end_line = lines.index('@<TRIPOS>BOND\n')

    atom_lines = lines[lines.index('@<TRIPOS>ATOM\n') + 1:atom_end_line]
    bond_lines = lines[lines.index('@<TRIPOS>BOND\n') + 1:]
    atoms = []
    positions = []
    for atom in atom_lines:
        sybyl_type = atom.split()[5]  
        atoms.append(ATOM_TYPE.index(sybyl_type)
                     if sybyl_type in ATOM_TYPE
                     else len(ATOM_TYPE))
        positions.append([eval(atom.split()[2]), eval(atom.split()[3]), eval(atom.split()[4])])
    edge_1 = [int(i.split()[1]) - 1 for i in bond_lines]
    edge_2 = [int(i.split()[2]) - 1 for i in bond_lines]
    edge_attr = [EDGE_ATTR[i.split()[3]] for i in bond_lines]
    x = torch.tensor(atoms)
    edge_idx = torch.tensor([edge_1 + edge_2, edge_2 + edge_1])
    edge_attr = torch.tensor(edge_attr + edge_attr)
    if not geometric:
        positions = torch.tensor(positions)
        tdEdge = to_scipy_sparse_matrix(edge_idx, edge_attr).todense()
        tdEdge = torch.from_numpy(np.array(tdEdge, dtype=np.float32).flatten())
        graph = Data(x=x, pos=positions, edge=tdEdge)
    else:
        graph = Data(x=x, edge_index=edge_idx, edge_attr=edge_attr)
    return graph


class PROTACDataset(Dataset):
    def __init__(self, data_dir, clean_data, geometric=False, sequence = False, feature_root='data/features'):
        self.data_dir = data_dir
        self.data = pd.read_csv(clean_data)
        self.geometric = geometric
        self.sequence = sequence
        self.feature_root = Path(feature_root) if feature_root is not None else None

    def _load_optional_feature(self, feature_group, compound_id):
        if self.feature_root is None:
            return None
        feature_path = self.feature_root / feature_group / f'{compound_id}.npz'
        if not feature_path.exists():
            return None
        return np.load(feature_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        compound_id, target_sequence, e3_ligase_sequence, label = row['compound id'], row['target_sequence'], row[
            'e3_ligase_sequence'], row['label']
        self.label = int(label)
        warhead_file = self.data_dir + 'warhead_{}.mol2'.format(compound_id)
        ligase_ligand_file = self.data_dir + 'e3_ligand_{}.mol2'.format(compound_id)
        linker_file = self.data_dir + 'linker_{}.mol2'.format(compound_id)

        warhead = mol2graph(warhead_file, SYBYL_ATOM_TYPE, self.geometric)
        ligase_ligand = mol2graph(ligase_ligand_file, SYBYL_ATOM_TYPE, self.geometric)
        linker = mol2graph(linker_file, SYBYL_ATOM_TYPE, self.geometric)
        target_sequence_str = str(target_sequence)
        e3_ligase_sequence_str = str(e3_ligase_sequence)
        target_sequence = label_sequence(target_sequence_str, FASTA_CHAR)
        e3_ligase_sequence = label_sequence(e3_ligase_sequence_str, FASTA_CHAR)

        plm_npz = self._load_optional_feature('plm', compound_id)
        if plm_npz is not None and 'target_plm' in plm_npz and 'e3_plm' in plm_npz:
            target_plm = torch.tensor(plm_npz['target_plm'], dtype=torch.float32)
            e3_plm = torch.tensor(plm_npz['e3_plm'], dtype=torch.float32)
        else:
            target_len = min(len(target_sequence_str), 1000)
            e3_len = min(len(e3_ligase_sequence_str), 1000)
            target_plm = torch.zeros((target_len, DEFAULT_PLM_DIM), dtype=torch.float32)
            e3_plm = torch.zeros((e3_len, DEFAULT_PLM_DIM), dtype=torch.float32)

        linker_npz = self._load_optional_feature('linker', compound_id)
        if linker_npz is not None and 'linker_conf' in linker_npz:
            linker_conf = torch.tensor(linker_npz['linker_conf'], dtype=torch.float32)
        else:
            linker_conf = torch.zeros((DEFAULT_LINKER_CONF_DIM,), dtype=torch.float32)

        smiles_npz = self._load_optional_feature('smiles', compound_id)
        if smiles_npz is None:
            raise FileNotFoundError(
                f"缺少ChemBERTa特征文件: {self.feature_root / 'smiles' / f'{compound_id}.npz'}"
            )
        if 'smiles_chemberta' not in smiles_npz:
            raise KeyError(f"smiles特征文件缺少字段 smiles_chemberta: compound_id={compound_id}")
        smiles_chemberta = torch.tensor(smiles_npz['smiles_chemberta'], dtype=torch.float32)

        sample = {
            "ligase_ligand": ligase_ligand,
            "ligase": e3_ligase_sequence,
            "target_ligand": warhead,
            "target": target_sequence,
            "linker": linker,
            "target_plm": target_plm,
            "e3_plm": e3_plm,
            "linker_conf": linker_conf,
            "smiles_chemberta": smiles_chemberta,
            "label": int(label),
        }
        return sample


def collater(data_list):
    batch = {}
    ligase_ligand = [x["ligase_ligand"] for x in data_list]
    ligase = [x["ligase"] for x in data_list]
    target_ligand = [x["target_ligand"] for x in data_list]
    target = [x["target"] for x in data_list]
    linker = [x["linker"] for x in data_list]
    target_plm = [x["target_plm"] for x in data_list]
    e3_plm = [x["e3_plm"] for x in data_list]
    linker_conf = [x["linker_conf"] for x in data_list]
    smiles_chemberta = [x["smiles_chemberta"] for x in data_list]
    label = [x["label"] for x in data_list]

    batch["ligase_ligand"] = Batch.from_data_list(ligase_ligand)
    batch["ligase"] = torch.from_numpy(np.array(ligase))
    batch["target_ligand"] = Batch.from_data_list(target_ligand)
    batch["target"] = torch.from_numpy(np.array(target))
    batch["linker"] = Batch.from_data_list(linker)
    batch["target_plm"] = pad_sequence(target_plm, batch_first=True)
    batch["e3_plm"] = pad_sequence(e3_plm, batch_first=True)
    batch["linker_conf"] = torch.stack(linker_conf, dim=0)
    batch["smiles_chemberta"] = torch.stack(smiles_chemberta, dim=0)
    batch["label"] = torch.tensor(label)
    return batch
