import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from models import activation_function_dict
# GCN-CNN based model

class GAT_GCN_no_p(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, protein_processing_type="convolution", featurizer=False):

        super(GAT_GCN_no_p, self).__init__()
        self.protein_processing_type = protein_processing_type
        self.featurizer = featurizer
        self.embed_dim = embed_dim
        #self.n_filters_power = 4
        self.embed_dim = embed_dim
        #self.kernel_size_power = 5
        self.heads_1 = 2
        self.dropout_GNN_layers = 0.000
        self.output_dim_power_0 = 1
        self.dropout_connection_layers = 0.090
        self.act = "leaky_relu"
        self.activation = activation_function_dict[self.act]

        # graph layers
        self.gnn1_1 = GCNConv(num_features_xd, num_features_xd * 2 ** self.output_dim_power_0)
        self.gnn1_2 = GATConv(num_features_xd * 2 ** self.output_dim_power_0, num_features_xd * 2 ** self.output_dim_power_0, heads=self.heads_1)

        self.fc_g1 = nn.Linear(num_features_xd * 2 ** self.output_dim_power_0 * self.heads_1, output_dim)
        self.dropout_gnn = nn.Dropout(self.dropout_GNN_layers)
        self.dropout_layer = nn.Dropout(self.dropout_connection_layers)

        # protein sequence branch
        if self.protein_processing_type == "convolution":
            self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
            self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=2 ** self.n_filters_power,
                                       kernel_size=2 ** self.kernel_size_power)
            self.fc1_xt = nn.Linear((2 ** self.n_filters_power) * (self.embed_dim - 2 ** self.kernel_size_power + 1),
                                    output_dim)
            self.fc1 = nn.Linear(2 * output_dim, 1024)
        elif self.protein_processing_type == "one_hot_encoding":
            self.fc1 = nn.Linear(output_dim + num_features_xt, 1024)
        elif self.protein_processing_type == "no_protein":
            self.fc1 = nn.Linear(output_dim, 1024)

        # combined layers
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)

        self.extractor_dim = 512

        # activation
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        x = self.gnn1_1(x, edge_index)
        x = self.activation(x)
        x = self.dropout_gnn(x)

        x = self.gnn1_2(x, edge_index)
        x = self.activation(x)
        x = self.dropout_gnn(x)

        x = gmp(x, batch)  # global max pooling

        x = self.fc_g1(x)
        x = self.activation(x)
        x = self.dropout_layer(x)

        if self.protein_processing_type == "convolution":
            # 1d conv layers
            embedded_xt = self.embedding_xt(target)
            conv_xt = self.conv_xt_1(embedded_xt)
            # flatten
            xt = conv_xt.view(-1, 2 ** self.n_filters_power * (self.embed_dim - 2 ** self.kernel_size_power + 1))
            xt = self.fc1_xt(xt)
        elif self.protein_processing_type == "one_hot_encoding":
            xt = target

        if self.protein_processing_type != "no_protein":
            # concat
            xc = torch.cat((x, xt), 1)
        else:
            xc = x
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout_layer(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout_layer(xc)
        if self.featurizer:
            return xc
        out = self.out(xc)
        return out
