import pickle
from collections import Counter, defaultdict
import torch
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.utils import compute_class_weight
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
import os
import json
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from itertools import repeat
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
import random
import warnings
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")


multi_atoms = ['Br', 'Cl', 'Si', 'Se', 'As', 'Te', 'At', 'Li', 'Na', 'Rb', 'Cs', 'Fr', 'Be', 'Mg', 'Ca', 'Sr', 'Ba',
               'Ra', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'Sn', 'Pb', 'Ag', 'Cd', 'In', 'Sb',
               'Hg', 'Tl', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
               'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Al', 'Ar', 'He', 'Ne', 'Kr', 'Xe', 'Rn',
               'Au', 'Pt', 'Pd', 'Ir', 'Re', 'Os', 'Hf', 'Ta', 'Re', 'Os', 'Hf', 'Ta', 'Sn', 'Pb', 'Bi', 'Po', 'At',
               'Zr', 'Ru', 'Rh', 'Ag', 'Cd', 'In', 'Np', 'Th', 'Pa', 'Ac']
descriptor_names = [d[0] for d in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)


class DynamicSmilesTokenizer:
    def __init__(self, path, max_length):
        self.path = path
        self.max_length = max_length
        if os.path.exists(path):
            self.load_vocab()
        else:
            self.vocab = ['[PAD]', '[UNK]']
            self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
            self.idx_to_token = {idx: token for idx, token in enumerate(self.vocab)}
            self.current_idx = len(self.vocab)

    def tokenize_smiles(self, smi):
        tokens = []
        i = 0
        while i < len(smi):
            if i + 1 < len(smi) and smi[i:i+2] in multi_atoms:
                tokens.append(smi[i:i+2])
                i += 2
            else:
                tokens.append(smi[i])
                i += 1
        return tokens

    def add_token(self, token):
        self.token_to_idx[token] = self.current_idx
        self.idx_to_token[self.current_idx] = token
        self.vocab.append(token)
        self.current_idx += 1

    def encode(self, tokens):
        for i, token in enumerate(tokens):
            if token not in self.token_to_idx:
                self.add_token(token)
        encoded = []
        for i, token in enumerate(tokens):
            encoded.append(self.token_to_idx[token])
        return encoded

    def process_smiles(self, smiles_list):
        encoded_list = []
        for smi in smiles_list:
            tokens = self.tokenize_smiles(smi)
            encoded = self.encode(tokens)
            if len(encoded) > self.max_length:
                encoded = encoded[:self.max_length]
            else:
                encoded += [self.token_to_idx['[PAD]']] * (self.max_length - len(encoded))
            encoded_list.append(encoded)
        return encoded_list

    def get_vocab_size(self):
        return len(self.vocab)

    def save_vocab(self):
        with open(self.path, "w") as f:
            for token in self.vocab:
                f.write(f"{token}\n")
        print(f"vocab save at: {self.path}")

    def load_vocab(self):
        with open(self.path, "r") as f:
            self.vocab = [line.strip() for line in f.readlines()]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        self.current_idx = len(self.vocab)


def save_symbol2idx(symbol2idx, path):
    with open(path, 'w') as f:
        json.dump(dict(symbol2idx), f)


def get_atom_features(atom, use_chirality=True):
    symbol = atom.GetSymbol()
    degree = atom.GetTotalDegree()
    formal_charge = atom.GetFormalCharge()
    num_hydrogens = atom.GetTotalNumHs(includeNeighbors=True)
    hybridization = atom.GetHybridization().real
    aromatic = int(atom.GetIsAromatic())
    chirality = atom.GetChiralTag().real if use_chirality else 0
    return [
        symbol,
        degree,
        formal_charge,
        num_hydrogens,
        hybridization,
        aromatic,
        chirality
    ]


def smiles_to_graph(mol, symbol2idx=None, current_idx=None, use_chirality=True):
    if not mol:
        return None

    if symbol2idx is None:
        symbol2idx = {}
        current_idx = 0

    features = []
    symbols = []
    for atom in mol.GetAtoms():
        feat = get_atom_features(atom, use_chirality=use_chirality)
        symbol = feat[0]
        symbols.append(symbol)
        if symbol not in symbol2idx:
            symbol2idx[symbol] = current_idx
            current_idx += 1
        feat[0] = symbol2idx[symbol]
        features.append(feat)

    x = torch.tensor(features, dtype=torch.float)
    edge_index = []
    edge_attr = []

    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }

    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        btype = bond_type_map.get(bond.GetBondType(), 0)
        edge_index.extend([[start, end], [end, start]])
        edge_attr.extend([btype, btype])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    fp = torch.tensor(
        np.concatenate([np.array(ecfp, dtype=np.float32), np.array(maccs, dtype=np.float32)]),
        dtype=torch.float
    ).unsqueeze(0)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, fp=fp), symbol2idx, current_idx


def get_severity(diliseverity, smiles):
    severity = diliseverity.loc[diliseverity['SMILES'] == smiles, 'diliseverity']
    sev = torch.zeros(9)
    if len(severity) > 0:
        sev[severity.values[0]] = 1
    return sev.unsqueeze(0)


class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, task, sequence_max_length, dataset='tox21', empty=False):

        self.dataset = dataset
        self.root = root + '/' + dataset + "/new/" + str(task + 1)
        self.tokenizer = DynamicSmilesTokenizer(os.path.join(root, 'vocab.txt'), max_length=sequence_max_length)
        self.diliseverity = pd.read_csv(os.path.join(root, 'DILIseverity.csv'))
        self.symbol2idx_path = os.path.join(root, 'symbol2idx.json')
        if os.path.exists(self.symbol2idx_path):
            with open(self.symbol2idx_path, 'r') as f:
                loaded = json.load(f)
            self.symbol2idx = loaded
            self._current_idx = len(loaded)
        else:
            self.symbol2idx = {}
            self._current_idx = 0
        super(MoleculeDataset, self).__init__(self.root)
        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
            self.smiles_list = pd.read_csv(os.path.join(self.processed_dir, 'smiles.csv'), header=None).to_numpy()[:, 0]
            self.seqences = pd.read_csv(os.path.join(self.processed_dir, 'sequences.csv'), header=None).to_numpy()

    def get(self, idx):
        data = Data()
        data['smiles'] = self.smiles_list[idx]
        data['sequence'] = self.seqences[idx]
        for key in ['x', 'edge_index', 'fp', 'y']:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        if self.dataset == 'DILIconcern_support' or self.dataset == 'DILIconcern_query' or self.dataset == 'DILIconcern_amb':
            data['severity'] = torch.tensor(get_severity(self.diliseverity, data['smiles']))
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return self.dataset + '_geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. ' 'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []
        smiles_list, rdkit_mol_objs, labels = _load_dataset(self.raw_paths[0], self.dataset)
        sequences = self.tokenizer.process_smiles(smiles_list)
        for i in range(len(smiles_list)):
            rdkit_mol = rdkit_mol_objs[i]
            data, self.symbol2idx, self._current_idx = smiles_to_graph(rdkit_mol, symbol2idx=self.symbol2idx, current_idx=self._current_idx)
            data.y = torch.tensor(labels[i, :])
            data.smiles = smiles_list[i]
            data.sequence = torch.tensor(sequences[i])
            data_list.append(data)
            data_smiles_list.append(smiles_list[i])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir, 'smiles.csv'), index=False, header=False)
        sequences = pd.DataFrame(sequences)
        sequences.to_csv(os.path.join(self.processed_dir, 'sequences.csv'), index=False, header=False)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        save_symbol2idx(self.symbol2idx, self.symbol2idx_path)
        self.tokenizer.save_vocab()


def _load_dataset(input_path, dataset='tox21'):
    binary_list, idx = None, None
    with open(input_path) as json_file:
        metrics = json.load(json_file)
    binary_list = metrics
    smiles_list = []
    for l in binary_list:
        for i in l:
            smiles_list.append(i)
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = np.zeros((len(smiles_list), 1), dtype=int)
    if len(binary_list) == 2:
        labels[len(binary_list[0]):, 0] = 1
    else:
        start_ind = 0
        for i in range(len(binary_list)-1):
            start_ind += len(binary_list[i])
            end_ind = start_ind + len(binary_list[i+1])
            labels[start_ind:end_ind, 0] = i+1
    return smiles_list, rdkit_mol_objs_list, labels


def obtain_distr_list(dataset):
    if dataset == "tox21":
        return [[6956, 309], [6521, 237], [5781, 768], [5521, 300], [5400, 793], [6605, 350], [6264, 186], [4890, 942], [6808, 264], [6095, 372], [4892, 918], [6351, 423]]
    elif dataset == "DILIconcern_support":
        return [[290, 188], [289, 189], [377, 101]]
    elif dataset == "DILIconcern_query":
        return [[160, 80], [160, 80], [160, 80]]
    else:
        return None


def sample_inds(data, size):
    len_data = len(data)
    if len_data >= size:
        return random.sample(data, size)
    else:
        return random.sample(data, len_data) + sample_inds(data, size - len_data)


def mydataset(args, tasks, max_length, dataset_name):
    preload_data = {}
    if args.preload_train_data:
        print(f'preload {dataset_name} data')
        for task in range(tasks):
            dataset = MoleculeDataset(args.data_dir, task, max_length, dataset=dataset_name)
            preload_data[task] = dataset
    return preload_data


def get_train_data_sample(task_id, tasks, preload_data, dataset, n_shot, batch_size):
    task = tasks[task_id]
    data = preload_data[task]
    distri_list = obtain_distr_list(dataset)
    thresh = distri_list[task][0]
    neg_sample = sample_inds(range(0, thresh), n_shot)
    pos_sample = sample_inds(range(thresh, len(data)), n_shot)
    list_ = neg_sample[:n_shot] + pos_sample[:n_shot]
    train_data = data[torch.tensor(list_)]
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_data = {'data_loader': loader}
    return eval_data


def get_test_data_sample(task_id, tasks, preload_s_data, preload_q_data, n_shot, batch_size):
    task = tasks[task_id]
    support_data = preload_s_data[task]
    query_data = preload_q_data[task]
    embedding = torch.cat([support_data.desc], dim=1)
    support_ind = []
    for label in range(2):
        indices = np.where(support_data.y.cpu() == label)[0]
        class_embs = embedding[indices]
        if len(class_embs) > n_shot:
            rep_indices = select_support_set_by_kmeans(class_embs, k=n_shot)
        else:
            rep_indices = select_support_set_by_kmeans(class_embs, k=len(class_embs) - 1)
        if n_shot == 1:
            rep_indices = [rep_indices]
        support_ind.extend(indices[rep_indices])
    s_data = support_data[torch.tensor(support_ind)]

    s_loader = DataLoader(s_data, batch_size=batch_size, shuffle=True, num_workers=0)
    q_loader = DataLoader(query_data, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_data = {'s_data': s_loader, 'q_data': q_loader}
    return eval_data


def select_support_set_by_kmeans(embeddings, k=10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(embeddings.detach().cpu().numpy())
    centers = torch.tensor(kmeans.cluster_centers_, device=embeddings.device)
    distances = torch.cdist(embeddings, centers)
    indices = torch.argmin(distances, dim=0)
    return indices
