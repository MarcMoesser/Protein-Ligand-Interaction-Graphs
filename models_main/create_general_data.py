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

#this is just for the nameing convention as defined in config/create_general_data
dataset = config["preprocessing"]["dataset"]
print('\ndataset:', dataset)
opts = ["train", "valid", "test"]

#READ IN MAIN CSV FOR DATA SPLIT
dataset_csv_file = config["preprocessing"]["dataset_csv"]
dataset_csv = pd.read_csv(dataset_csv_file, index_col=0)

if not "split" in dataset_csv.columns:
    raise ValueError("PPlease create a .csv file with a column 'split' containing values 'train', 'valid', and 'test'.")
else:
    splits_given = np.unique(dataset_csv.split)
    print("Splits given in the data set are ", splits_given)
    if not all(["train" in splits_given, "valid" in splits_given, "test" in splits_given]):
        raise ValueError("Please create a .csv file with a column 'split' ONLY containing values 'train', 'valid', and 'test'.")
    else:
        train_df = dataset_csv.loc[dataset_csv.split == "train", :]
        #shuffle
        #train_df = train_df.sample(frac=1).reset_index(drop=True) #-> not necessary since we have shuffle=True in the DataLoader

        train_df.to_csv('data/' + dataset + '_' + "train" + ".csv")

        valid_df = dataset_csv.loc[dataset_csv.split == "valid", :]
        # shuffle
        #valid_df = valid_df.sample(frac=1).reset_index(drop=True)  #-> not necessary since we have shuffle=True in the DataLoader
        valid_df.to_csv('data/' + dataset + '_' + "valid" + ".csv")

        test_df = dataset_csv.loc[dataset_csv.split == "test", :]
        # shuffle
        #test_df = test_df.sample(frac=1).reset_index(drop=True)  #-> not necessary since we have shuffle=True in the DataLoader
        test_df.to_csv('data/' + dataset + '_' + "test" + ".csv")

        print("Number of data points: \nTrain: {} \nValidation: {} \nTest: {}".format(len(train_df), len(valid_df), len(test_df)))

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
    #If you want to train on a subset, this will select only the relevant PDB codes

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

processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
processed_data_file_valid = 'data/processed/' + dataset + '_valid.pt'
processed_data_file_test = 'data/processed/' + dataset + '_test.pt'

df = train_df
train_idx, train_prots, train_Y = list(df['Identifier']), list(df['target_sequence']), list(df['affinity'])
train_prots_ls = list(set(train_prots))
if config["preprocessing"]["protein"]["protein_processing_type"] == "one_hot_encoding":
    train_prots_one_hot = one_hot_encode_proteins(train_prots, list_training_proteins=train_prots_ls)
    train_idx, train_prots, train_Y = np.asarray(train_idx), np.asarray(train_prots_one_hot), np.asarray(train_Y)
else:
    XT = [seq_cat(t, config) for t in train_prots]
    train_idx, train_prots, train_Y = np.asarray(train_idx), np.asarray(XT), np.asarray(train_Y)


df = valid_df
valid_idx, valid_prots, valid_Y = list(df['Identifier']), list(df['target_sequence']), list( #-> MARC Identifier)
    df['affinity'])
if config["preprocessing"]["protein"]["protein_processing_type"] == "one_hot_encoding":
    valid_prots_one_hot = one_hot_encode_proteins(valid_prots, list_training_proteins=train_prots_ls)
    valid_idx, valid_prots, valid_Y = np.asarray(valid_idx), np.asarray(valid_prots_one_hot), np.asarray(valid_Y)
else:
    XT = [seq_cat(t, config) for t in valid_prots]
    valid_idx, valid_prots, valid_Y = np.asarray(valid_idx), np.asarray(XT), np.asarray(valid_Y)

df = test_df
test_idx, test_prots, test_Y = list(df['Identifier']), list(df['target_sequence']), list( #-> MARC (Identifier)
    df['affinity'])
if config["preprocessing"]["protein"]["protein_processing_type"] == "one_hot_encoding":
    test_prots_one_hot = one_hot_encode_proteins(test_prots, list_training_proteins=train_prots_ls)
    test_idx, test_prots, test_Y = np.asarray(test_idx), np.asarray(test_prots_one_hot), np.asarray(test_Y)
else:
    XT = [seq_cat(t, config) for t in test_prots]
    test_idx, test_prots, test_Y = np.asarray(test_idx), np.asarray(XT), np.asarray(test_Y)

# make data PyTorch Geometric ready
print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = TestbedDataset(root='data', dataset=dataset + '_train', xd=train_idx, xt=train_prots, y=train_Y,
                            smile_graph=smile_graph, override=True)
print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = TestbedDataset(root='data', dataset=dataset + '_valid', xd=valid_idx, xt=valid_prots,
                            y=valid_Y, smile_graph=smile_graph, override=True)
print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = TestbedDataset(root='data', dataset=dataset + '_test', xd=test_idx, xt=test_prots, y=test_Y,
                           smile_graph=smile_graph, override=True)

print(processed_data_file_train, processed_data_file_valid, ' and ', processed_data_file_test, ' have been created')
