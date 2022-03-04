import torch
import os
import numpy as np
from math import sqrt
from scipy import stats
import torch.nn as nn

import torch.nn.functional as F

#get all the models
from models.gat_PLIG_no_p import GATNet_PLIG_no_p
from models.gat_PLIG_with_p import GATNet_PLIG_with_p
from models.gat_no_p import GATNet_no_p
from models.gat_with_p import GATNet_with_p

from models.gat_gcn_PLIG_no_p import GAT_GCN_PLIG_no_p
from models.gat_gcn_PLIG_with_p import GAT_GCN_PLIG_with_p
from models.gat_gcn_no_p import GAT_GCN_no_p
from models.gat_gcn_with_p import GAT_GCN_with_p

from models.gcn_PLIG_no_p import GCNNet_PLIG_no_p
from models.gcn_PLIG_with_p import GCNNet_PLIG_with_p
from models.gcn_no_p import GCNNet_no_p
from models.gcn_with_p import GCNNet_with_p

from models.ginconv_PLIG_no_p import GINConvNet_PLIG_no_p
from models.ginconv_PLIG_with_p import GINConvNet_PLIG_with_p
from models.ginconv_no_p import GINConvNet_no_p
from models.ginconv_with_p import GINConvNet_with_p

from models.SGConv_PLIG_no_p import SGCNet_PLIG_no_p
from models.SGConv_PLIG_with_p import SGCNet_PLIG_with_p
from models.SGConv_no_p import SGCNet_no_p
from models.SGConv_with_p import SGCNet_with_p

from models.sage_conv_PLIG_no_p import SageNet_PLIG_no_p
from models.sage_conv_PLIG_with_p import SageNet_PLIG_with_p
from models.sage_conv_no_p import SageNet_no_p
from models.sage_conv_with_p import SageNet_with_p

from models.mlp_ecfp_512_no_p import MLPNet_ECFP512_no_p
from models.mlp_ecfp_512_with_p import MLPNet_ECFP512_with_p
from models.mlp_ecfp_1024_no_p import MLPNet_ECFP1024_no_p
from models.mlp_ecfp_1024_with_p import MLPNet_ECFP1024_with_p
from models.mlp_fcfp_512_no_p import MLPNet_FCFP512_no_p
from models.mlp_fcfp_512_with_p import MLPNet_FCFP512_with_p
from models.mlp_fcfp_1024_no_p import MLPNet_FCFP1024_no_p
from models.mlp_fcfp_1024_with_p import MLPNet_FCFP1024_with_p
from models.mlp_ecif_no_p import MLPNet_ECIF_no_p
from models.mlp_ecif_with_p import MLPNet_ECIF_with_p


model_dict = {"GINConvNet_PLIG_no_p": GINConvNet_PLIG_no_p, "GINConvNet_PLIG_with_p": GINConvNet_PLIG_with_p, "GINConvNet_no_p": GINConvNet_no_p, "GINConvNet_with_p": GINConvNet_with_p,
              "GATNet_PLIG_no_p": GATNet_PLIG_no_p, "GATNet_PLIG_with_p": GATNet_PLIG_with_p, "GATNet_no_p": GATNet_no_p, "GATNet_with_p": GATNet_with_p,
              "GAT_GCN_PLIG_no_p": GAT_GCN_PLIG_no_p, "GAT_GCN_PLIG_with_p": GAT_GCN_PLIG_with_p, "GAT_GCN_no_p": GAT_GCN_no_p, "GAT_GCN_with_p": GAT_GCN_with_p,
              "GCNNet_PLIG_no_p": GCNNet_PLIG_no_p, "GCNNet_PLIG_with_p": GCNNet_PLIG_with_p, "GCNNet_no_p": GCNNet_no_p, "GCNNet_with_p": GCNNet_with_p,
              "SGCNet_PLIG_no_p": SGCNet_PLIG_no_p, "SGCNet_PLIG_with_p": SGCNet_PLIG_with_p, "SGCNet_no_p": SGCNet_no_p, "SGCNet_with_p": SGCNet_with_p,
              "SageNet_EPLIG_no_p": SageNet_PLIG_no_p, "SageNet_PLIG_with_p": SageNet_PLIG_with_p, "SageNet_no_p": SageNet_no_p, "SageNet_with_p": SageNet_with_p,
              "MLPNet_ECFP512_no_p": MLPNet_ECFP512_no_p, "MLPNet_ECFP512_with_p": MLPNet_ECFP512_with_p, "MLPNet_ECFP1024_no_p": MLPNet_ECFP1024_no_p, "MLPNet_ECFP1024_with_p": MLPNet_ECFP1024_with_p,
              "MLPNet_FCFP512_no_p": MLPNet_FCFP512_no_p, "MLPNet_FCFP512_with_p": MLPNet_FCFP512_with_p,  "MLPNet_FCFP1024_no_p": MLPNet_FCFP1024_no_p, "MLPNet_FCFP1024_with_p": MLPNet_FCFP1024_with_p,
              "MLPNet_ECIF_no_p": MLPNet_ECIF_no_p, "MLPNet_ECIF_with_p": MLPNet_ECIF_with_p} #ECIF are tuned like ECFP


activation_function_dict = {"relu": F.relu, "leaky_relu": F.leaky_relu, "sigmoid": F.sigmoid}

def get_num_parameters(model):
    """
    counts the number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def enable_dropout(m):
    """
    Parameters
    ----------
    m : submodule of a Pytorch model

    Activates dropout layers independently

    """
    if type(m) == nn.Dropout:
        m.train()


def collate_fn(batch):
    """
    function needed for data loaders
    """
    feature_list, protein_seq_list, label_list = [], [], []
    for _features, _protein_seq, _label in batch:
        #print(type(_features), type(_protein_seq), type(_label))
        feature_list.append(_features)
        protein_seq_list.append(_protein_seq)
        label_list.append(_label)
    return torch.Tensor(feature_list), torch.Tensor(protein_seq_list), torch.Tensor(label_list)

def rmse(y,f):
    """
    taken from https://github.com/thinng/GraphDTA

    computes the RMSE
    """
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    """
    taken from https://github.com/thinng/GraphDTA

    computes the MSE
    """
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    """
    taken from https://github.com/thinng/GraphDTA

    computes the pearson correlation coefficient
    """
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    """
    taken from https://github.com/thinng/GraphDTA

    computes the spearman correlation coefficient
    """
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    """
    taken from https://github.com/thinng/GraphDTA

    computes the concordance index
    """
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
