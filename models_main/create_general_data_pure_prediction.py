import pandas as pd
import numpy as np
import json, pickle
from collections import OrderedDict
from utils import *
import yaml
from tqdm import tqdm
import sys

# get the config which we will need throughout the whole script
with open(sys.argv[1], "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

#this is just for the nameing convention as defined in config
dataset = config["preprocessing"]["dataset"]
print('\ndataset:', dataset)
opts = ["test"]

#READ IN MAIN CSV 
dataset_csv_file = config["preprocessing"]["dataset_csv"]
dataset_csv = pd.read_csv(dataset_csv_file, index_col=0)

test_df = dataset_csv
#test_df.to_csv('data/' + dataset + '_' + "test" + ".csv")

print("Number of data points: \nTest: {}".format(len(test_df)))

#  create the features and the graph, i.e. the input to GNNs -> REPLACED BY LOADING IN THE PICKLED GRAPH FROM PREPROCESSING FOR PLIG (MARC)
if config["preprocessing"]["use_graph"] == False:
    print("skipping graph gen and graph load, not a GNN run")
    exit()

elif config["preprocessing"]["external_graph"]["use"]:
    print("loading graph from pickle PLIG file")
    with open(str(config["preprocessing"]["external_graph"]["path"]), 'rb') as handle: #-> ath was set in config
        smile_graph_all = pickle.load(handle)

    #only load the subset of the data that you need.
    #the graph.pickle file should have all PDBBind_combined entries in it.

    smile_graph = {}
    for i in dataset_csv["Identifier"]:
        smile_graph[i] = smile_graph_all[i]
    #print(len(smile_graph.keys()))
    assert len(smile_graph.keys()) == len(dataset_csv["Identifier"])

else:
    print("creating new graphs")
    # collect all smiles
    compound_iso_smiles = []
    Identifier_list = []
    for opt in opts:
        df = pd.read_csv('data/' + dataset + '_' + opt + '.csv')
        compound_iso_smiles += list(df['compound_iso_smiles'])
        Identifier_list += list(df["Identifier"])


    # create the features and the graph, i.e. the input to GNNs
    smile_graph = {}
    for smile, Identifier in zip(compound_iso_smiles, Identifier_list):
        g = smile_to_graph(smile, config)
        smile_graph[Identifier] = g

    #Put this into the "else" statement since standardization not really applicable like this for PLIG
    # standardize the data: Therefore, first collect all the data into one array and fit a standard scaler.
    ls_compound_iso_smile = list(compound_iso_smiles)
    arr = np.array((smile_graph[Identifier_list[0]])[1])
    for id in tqdm(Identifier_list[1:]):
        atom_features = (smile_graph[id])[1]
        arr = np.concatenate((arr, np.stack(atom_features, axis=0)), axis=0)

    scaler = StandardScaler()
    scaler.fit(arr)

    # now apply the standard scaler to all data points
    for id in tqdm(Identifier_list):
        c = smile_graph[id][0]
        for j in range(c):
            if len(np.expand_dims(smile_graph[id][1][j], axis=0).shape) == 2:
                (smile_graph[id])[1][j] = scaler.transform(np.expand_dims(smile_graph[id][1][j],axis=0))[0]
            else:
                (smile_graph[id])[1][j] = scaler.transform(
                    smile_graph[id][1][j])[0]


# convert to PyTorch data format

processed_data_file_test = 'data/processed/' + dataset + '_test.pt'

df = test_df
test_idx, test_prots, test_Y = list(df['Identifier']), list(df['target_sequence']), list( #-> MARC (Identifier)
    df['affinity'])
if config["preprocessing"]["protein"]["protein_processing_type"] == "one_hot_encoding":
    test_prots_one_hot = one_hot_encode_proteins(test_prots, list_training_proteins=train_prots_ls)
    test_idx, test_prots, test_Y = np.asarray(test_idx), np.asarray(test_prots_one_hot), np.asarray(test_Y)
else:
    XT = [seq_cat(t, config) for t in test_prots]
    test_idx, test_prots, test_Y = np.asarray(test_idx), np.asarray(XT), np.asarray(test_Y)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = TestbedDataset(root='data', dataset=dataset + '_test', xd=test_idx, xt=test_prots, y=test_Y,
                           smile_graph=smile_graph, override=True)

print(processed_data_file_test, ' have been created')
