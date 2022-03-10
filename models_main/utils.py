import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem
import yaml
import networkx as nx
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from helpers import enable_dropout
from sklearn.preprocessing import StandardScaler
import pickle


def predict(model, device, loader, mc_dropout=False, verbose=True, y_scaler=None):
    """
    function performing predictions of a GNN

    model: torch.nn.model
        the model to be trained
    device: torch.device
        indicates whether model is trained on GPU or CPU
    loader:
        data loader for data which predictions are performed on
    mc_dropout: bool
        whether we perform prediction in the course of a MC dropout model
    verbose: bool
        whether to print how many data points prediction is performed on
    y_scaler: sklearn.preprocessing.StandardScaler
        standard scaler transforming the target variable
    """
    model.eval()
    if mc_dropout:
        model.dropout_layer.train()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_vars = torch.Tensor()
    if verbose:
        print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    return y_scaler.inverse_transform(total_labels.numpy().flatten()), y_scaler.inverse_transform(total_preds.numpy().flatten()), total_vars.numpy().flatten() * y_scaler.var_


def predict_MLP(model, device, loader, mc_dropout=False, verbose=True, y_scaler=None):
    """
    function performing predictions of an MLP

    model: torch.nn.model
        the model to be trained
    device: torch.device
        indicates whether model is trained on GPU or CPU
    loader:
        data loader for data which predictions are performed on
    mc_dropout: bool
        whether we perform prediction in the course of a MC dropout model
    verbose: bool
        whether to print how many data points prediction is performed on
    y_scaler: sklearn.preprocessing.StandardScaler
        standard scaler transforming the target variable
    """
    model.eval()
    if mc_dropout:
        model.dropout_layer.train()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_vars = torch.Tensor()
    if verbose:
        print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = [d.to(device) if i % 2 == 0 else d.type(torch.LongTensor).to(device) for (i, d) in enumerate(data)]
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data[2].cpu()), 0)

    return y_scaler.inverse_transform(total_labels.numpy().flatten()), y_scaler.inverse_transform(total_preds.numpy().flatten()), total_vars.numpy().flatten() * y_scaler.var_

def atom_features(atom, features):
    """
    computes all features given in config
    """

    feature_list = []
    if "atom_symbol" in features:
        feature_list.extend(one_of_k_encoding(atom.GetSymbol(),['F', 'N', 'Cl', 'O', 'Br', 'C', 'H', 'P', 'I', 'S']))
    if "num_heavy_atoms" in features:  #  Adjusted with rdkit fix
        feature_list.extend(one_of_k_encoding(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"]), [0,1,2,3,4,5,6]))
    if "total_num_Hs" in features:  # Adjusted with rdkit fix
        feature_list.extend(one_of_k_encoding(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"]), [0,1,2,3,4]))
    if "explicit_valence" in features:  # -NEW ADDITION  -> replaces implicit valence
        feature_list.extend(one_of_k_encoding(atom.GetExplicitValence(), [0,1,2,3,4,5,6,7,8]))
    if "is_aromatic" in features:
        feature_list.extend([atom.GetIsAromatic()])
    if "is_in_ring" in features:
        feature_list.extend([atom.IsInRing()])
    if "formal_charge" in features:
        feature_list.extend(one_of_k_encoding(atom.GetFormalCharge(), [-3,-2,-1,0,1,2,3])) # THIS IS PROBABLY TOO HIGH OF A THRESHOLD -> NOTHING SHOULD HAVE MORE THAN +-2 -> pdbbind makes azides have triple bond (which makes the N +3 -> done this for quick fix)
    if "hybridization_type" in features:
        feature_list.extend(one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]))

    return np.array(feature_list)


def one_of_k_encoding(x, allowable_set):
    """
    taken from https://github.com/thinng/GraphDTA

    function which one hot encodes x w.r.t. allowable_set and x has to be in allowable_set

    x:
        element from allowable_set
    allowable_set: list
        list of elements x is from
    """
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """
    taken from https://github.com/thinng/GraphDTA

    function which one hot encodes x w.r.t. allowable_set with one bit reserved for elements not in allowable_set

    x:
        element from allowable_set
    allowable_set: list
        list of all known elements
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile, config):
    """
    adapted from https://github.com/thinng/GraphDTA

    constructs the ligand-based graph from the smile string

    smile: str
        SMILE string of molecule
    config:
        configuration file
    """
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom, config["preprocessing"]["atom_features"])
        features.append(feature)

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    # returns the size of the graph, the features of each node and the edge indices
    return c_size, features, edge_index


def seq_cat(prot, config):
    """
    adapted from https://github.com/thinng/GraphDTA

    encodes the amino acid sequence

    smile: str
        SMILE string of molecule
    config:
        configuration file
    """
    seq_voc = config["preprocessing"]["protein"]["seq_voc"]
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    max_seq_len = config["preprocessing"]["protein"]["max_seq_len"]
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

def init_weights(layer):
    """
    function which initializes weights
    """
    if hasattr(layer, "weight") and "BatchNorm" not in str(layer):
        torch.nn.init.xavier_normal_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is True:
            torch.nn.init.zeros_(layer.bias)


class TestbedDataset(InMemoryDataset):
    """
    adapted from https://github.com/thinng/GraphDTA

    class handling the dataset for a GNN
    """

    def __init__(self, root='/tmp', dataset=None,
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None, override=False, y_scaler=None):

        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]) and not override:
            self.data, self.slices = torch.load(self.processed_paths[0])
            print("processed paths:")
            print(self.processed_paths, self.processed_paths[0])

        else:
            self.process(xd, xt, y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
            print("run through processing")
        if y_scaler is None:
            y_scaler = StandardScaler()
            y_scaler.fit(np.reshape(self.data.y, (self.__len__(),1)))
        self.y_scaler = y_scaler
        self.data.y = [torch.tensor(element[0]).float() for element in self.y_scaler.transform(np.reshape(self.data.y, (self.__len__(),1)))]

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_len_protein_encoding(self):
        return self.data.target.shape[1]

    def process(self, xd, xt, y,smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting pdbcodes to graph: {}/{}'.format(i+1, data_len))
            pdbcodes = xd[i]
            target = xt[i]
            labels = y[i]
            # convert pdbcodes to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[pdbcodes]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])


class FPDataset(Dataset):
    """
    class handling the dataset for MLPs (i.e. fingerprint methods)
    """
    def __init__(self, pd_dir, config, y_scaler=None, list_training_proteins=None):
        # note that even if the data was preprocessed in "one_hot_encoding" mode we need to do it here again because
        # we do not access the .pt data is was saved to but only the csv file from preprocessing
        self.df = pd.read_csv(pd_dir)
        if config["preprocessing"]["fingerprints"]["use_MAPR"]:

            self.input = "mapr"
            print("MAPR fingerprints are used.")
                
        #MARC: added condition for ECIF
        elif config["preprocessing"]["fingerprints"]["use_ECIF"]:
            self.input = "ECIF"
            print("ECIF fingerprints are used.")

        elif config["preprocessing"]["fingerprints"]["use_ECFP512"]:
            self.input = "ECFP512"
            print("ECFP512 fingerprints are used.")

        elif config["preprocessing"]["fingerprints"]["use_ECFP1024"]:
            self.input = "ECFP1024"
            print("ECFP1024 fingerprints are used.")

        elif config["preprocessing"]["fingerprints"]["use_FCFP512"]:
            self.input = "FCFP512"
            print("FCFP512 fingerprints are used.")

        elif config["preprocessing"]["fingerprints"]["use_FCFP1024"]:
            self.input = "FCIF1024"
            print("FCFP1024 fingerprints are used.")

        #load the pickled contacts in
        pickle_file = config["preprocessing"]["fingerprints"]["FP_path"]

        with open(pickle_file, "rb") as f:
            self.fp_dict = pickle.load(f)

        self.col_index_target_seq = self.df.columns.get_loc("target_sequence")
        self.col_index_affinity = self.df.columns.get_loc("affinity")
        self.config = config
        self.list_training_proteins = list_training_proteins
        self.bool_prots_one_hot_encoded = (self.config["preprocessing"]["protein"]["protein_processing_type"] == "one_hot_encoding")
        if self.bool_prots_one_hot_encoded:
            if list_training_proteins is None:
                self.list_training_proteins = list(set(self.df.iloc[:, self.col_index_target_seq]))
            self.one_hot_encoded_proteins = one_hot_encode_proteins(self.df.iloc[:, self.col_index_target_seq], self.list_training_proteins)
        if y_scaler is None:
            y_scaler = StandardScaler()
            y_scaler.fit(np.reshape(np.array(self.df["affinity"]), (self.__len__(),1)))
        self.y_scaler = y_scaler
        self.df["affinity"] = self.y_scaler.transform(np.reshape(np.array(self.df["affinity"]), (self.__len__(),1)))

    def __len__(self):
        return self.df.shape[0]

    def get_encoded_proteins(self):
        return self.list_training_proteins

    # restructuring features so they can be loaded in from pickle files
    def __getitem__(self, idx):

        col_idx = self.df.columns.get_loc("Identifier")
        code = self.df.iloc[idx, col_idx]
        features = np.array(self.fp_dict[code])

        if self.bool_prots_one_hot_encoded is False:
            protein_sequence_encoded = seq_cat(self.df.iloc[idx, self.col_index_target_seq], self.config)
        else:
            protein_sequence_encoded = self.one_hot_encoded_proteins[idx]
        return features, protein_sequence_encoded, self.df.iloc[idx, self.col_index_affinity]

    def get_len_protein_encoding(self):
        if self.bool_prots_one_hot_encoded is False:
            return len(seq_cat(self.df.iloc[0, self.col_index_target_seq], self.config))
        else:
            return len(self.one_hot_encoded_proteins[0])

    def get_len_ligand_encoding(self):
        col_idx = self.df.columns.get_loc("Identifier")
        code = self.df.iloc[0, col_idx]
        features = np.array(self.fp_dict[code])

        return len(features)


def one_hot_encode_proteins(proteins, list_training_proteins):
    """
    outdated.
    this was needed when protein were one-hot encoded
    """

    if "other" not in list_training_proteins:
        list_training_proteins.append("other")
    ls = []
    for i,p in enumerate(proteins):
        ls += [one_of_k_encoding_unk(p, list_training_proteins)]
    return ls
