# imports
import sys
import os
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric import data as DATA 
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

cwd_folder = os.getcwd().split('/')[-1]
assert cwd_folder == 'explanations', f"Run this program from the 'explanations' folder, currentyly at {os.getcwd()}"

from explanation_utils.gnn_explainer  import GNNExplainer
from example_model.x_gat_PLIG_no_p import GATNet_PLIG_no_p
from explanation_utils.aggr_edge_directions import aggregate_edge_directions


def create_graph(pdb_id, PLIG_data, PDBbind_combined_csv):
    # get general info from csv
    df = PDBbind_combined_csv.loc[PDBbind_combined_csv['Identifier'] == pdb_id] 
    y = df['affinity'].item()
    smiles = df['compound_iso_smiles'].item()
    split = df['split'].item()
    # get PLIG from pickled file
    features = PLIG_data[pdb_id][1]
    edge_index = PLIG_data[pdb_id][2]
    graph = DATA.Data(x=torch.Tensor(features),
                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                    y=torch.FloatTensor([y]),
                    smiles=smiles,
                    pdb_id=pdb_id,
                    split=split)
    return graph

def main():
    # load data
    path_to_model = 'example_model/GATNet.model'
    PLIG_data = pickle.load(open('../models_main/data/PDBBind/PDBbind_combined_crystal_PLIG_6A_std.pickle', 'rb'))
    PDBbind_combined_csv = pd.read_csv('../models_main/data/PDBBind/PDBbind_combined_casf_split_preprocessed.csv',
        index_col=0)
    model = GATNet_PLIG_no_p(num_features_xd = 27, protein_processing_type='no_protein')
    model.load_state_dict(torch.load(path_to_model))

    # process arguments / choose pdb_id
    pdb_id = sys.argv[1].lower() 
    assert pdb_id in PDBbind_combined_csv['Identifier'].values, f"{pdb_id} not in dataset" 
    if len(sys.argv) >= 3:
        num_explanations = int(sys.argv[2])
        assert num_explanations > 0, "Explain at least once."
    else:
        num_explanations = 1
    if len(sys.argv) >= 4:
        note = sys.argv[3] # appended to explanations file/folder name
    else:
        note = ""
    
    # get graph
    graph = create_graph(pdb_id, PLIG_data, PDBbind_combined_csv)

    # scale values
    assert graph.split in ['train', 'valid', 'test'], f"Unexpected datasplit: {graph.split}"
    y_data_test = PDBbind_combined_csv.loc[PDBbind_combined_csv['split']=='test']
    y_data_test = np.array(y_data_test['affinity']).reshape(-1, 1)
    y_scaler_test = StandardScaler()
    y_scaler_test.fit(y_data_test)
    y_scaler = y_scaler_test

    # calculate prediction
    model.eval()
    pred = model(graph.x, graph.edge_index,torch.zeros(graph.x.shape[0], dtype=int, device=graph.x.device))
    pred = y_scaler.inverse_transform(np.array([pred.item()]).reshape(-1,1)).item()
    label = graph.y.item()
    print(f"Model predicted a value of [{pred:.4f}] for label [{label:.4f}] on {pdb_id.upper()}.")

    # get atoms, bonds and features
    feature_type =['Heavy Neighb.', 'H Neighb.', 'Explicit Valence', 'Aromaticity', 'In Ring',
                'C;4;3;0;0;0', 'C;4;3;1;0;0', 'C;4;1;3;0;0', 'N;3;2;1;0;0', 'O;2;1;0;0;0',
                'C;4;2;2;0;0', 'C;6;3;0;0;0', 'N;4;2;1;0;0', 'N;4;1;2;0;0', 'N;3;1;2;0;0',
                'C;5;3;0;0;0', 'S;2;1;1;0;0', 'C;4;2;1;1;1', 'C;4;3;0;1;1', 'N;3;2;0;1;1',
                'N;3;2;1;1;1', 'N;4;1;3;0;0', 'S;2;2;0;0;0', 'C;4;3;1;0;1', 'C;4;2;2;0;1',
                'N;3;3;0;0;1', 'O;2;1;1;0;0'] 

    # create df_info
    df_info = pd.DataFrame({"PDB_ID": [pdb_id],
                            "SMILES": [graph.smiles],
                            "Label": [label],
                            "Prediction": [pred],
                            "Explanations": [num_explanations]}).transpose()


    # instance explainer
    xpl_feat = GNNExplainer(model, epochs=300, return_type='regression',
                            feat_mask_type='feature', log=0)

    xpl_nodes = GNNExplainer(model, epochs=300, return_type='regression',
                            feat_mask_type='scalar', log=0)

    xpl_indiv = GNNExplainer(model, epochs=300, return_type='regression',
                            feat_mask_type='individual_feature', log=0)

    # initialize tensors to save mask values to 
    feat_masks = torch.empty(num_explanations, graph.x.shape[1])
    feat_edge_masks = torch.empty(num_explanations, graph.edge_index.shape[1])
    node_masks = torch.empty(num_explanations, graph.x.shape[0])
    node_edge_masks = torch.empty(num_explanations, graph.edge_index.shape[1])
    indiv_masks = torch.empty(num_explanations, graph.x.shape[0], graph.x.shape[1])
    indiv_edge_masks = torch.empty(num_explanations, graph.edge_index.shape[1])

    # calculate explanations
    for i in tqdm(range(num_explanations), desc="Calculating explanations"):
        feat_mask, feat_edge_mask = xpl_feat.explain_graph(graph.x, graph.edge_index)
        feat_masks[i] = feat_mask 
        feat_edge_masks[i] = feat_edge_mask 
        node_mask, node_edge_mask = xpl_nodes.explain_graph(graph.x, graph.edge_index)
        node_masks[i] = node_mask 
        node_edge_masks[i] = node_edge_mask 
        indiv_mask, indiv_edge_mask = xpl_indiv.explain_graph(graph.x, graph.edge_index)
        indiv_masks[i] = indiv_mask 
        indiv_edge_masks[i] = indiv_edge_mask 

    
    if num_explanations > 1:
        # calc mean and std
        feat_masks_std, feat_masks_mean = torch.std_mean(feat_masks, dim=0)
        feat_edge_masks_std, feat_edge_masks_mean = torch.std_mean(feat_edge_masks, dim=0)
        node_masks_std, node_masks_mean = torch.std_mean(node_masks, dim=0)
        node_edge_masks_std, node_edge_masks_mean = torch.std_mean(node_edge_masks, dim=0)
        indiv_masks_std, indiv_masks_mean = torch.std_mean(indiv_masks, dim=0)
        indiv_edge_masks_std, indiv_edge_masks_mean = torch.std_mean(indiv_edge_masks, dim=0)
    elif num_explanations == 1:
        # reshape to same shape as mean values
        feat_masks_mean = feat_masks.squeeze()
        feat_edge_masks_mean = feat_edge_masks.squeeze()
        node_masks_mean = node_masks.squeeze()
        node_edge_masks_mean = node_edge_masks.squeeze()
        indiv_masks_mean = indiv_masks.squeeze()
        indiv_edge_masks_mean = indiv_edge_masks.squeeze()

    # aggregate mean edges
    feat_edge_masks_dict = aggregate_edge_directions(graph, feat_edge_masks_mean)
    node_edge_masks_dict = aggregate_edge_directions(graph, node_edge_masks_mean)
    indiv_edge_masks_dict = aggregate_edge_directions(graph, indiv_edge_masks_mean)
    bonds_list = list(node_edge_masks_dict.keys())

    # node explanations
    df_node_masks = pd.DataFrame({"node_masks": [x.item() for x in node_masks_mean]})
    df_node_edge_masks_aggr = pd.DataFrame({"node_edge_masks_aggr": [x.item() for x in node_edge_masks_dict.values()]},
                                            index=bonds_list)
    df_node_edge_masks = pd.DataFrame({"node_edge_masks": [x.item() for x in node_edge_masks_mean]})
    if num_explanations > 1:
        df_node_masks["node_masks_std"] = [x.item() for x in node_masks_std]
        df_node_edge_masks["node_edge_masks_std"] = [x.item() for x in node_edge_masks_std]

    # feature explanations
    df_feat_masks = pd.DataFrame({"feat_masks": [x.item() for x in feat_masks_mean]},
                                  index=feature_type)
    df_feat_edge_masks_aggr = pd.DataFrame({"feat_edge_masks_aggr": [x.item() for x in feat_edge_masks_dict.values()]},
                                            index=bonds_list)
    df_feat_edge_masks = pd.DataFrame({"feat_edge_masks": [x.item() for x in feat_edge_masks_mean]})
    if num_explanations > 1:
        df_feat_masks["feat_masks_std"] = [x.item() for x in feat_masks_std]
        df_feat_edge_masks["feat_edge_masks_std"] = [x.item() for x in feat_edge_masks_std]

    # individual explanations
    df_indiv_masks = pd.DataFrame(indiv_masks_mean.numpy(), columns=feature_type)
    df_indiv_edge_masks_aggr = pd.DataFrame({"indiv_edge_masks": [x.item() for x in indiv_edge_masks_dict.values()]},
                                            index=bonds_list)
    df_indiv_edge_masks = pd.DataFrame({"indiv_edge_masks": [x.item() for x in indiv_edge_masks_mean]})
    if num_explanations > 1:
        df_indiv_masks_std = pd.DataFrame(indiv_masks_std.numpy(), columns=feature_type)
        df_indiv_edge_masks["indiv_edge_masks_std"] = [x.item() for x in indiv_edge_masks_std]

    # save data to csv 
    save_dir = "explanation_outputs/"+pdb_id+"_"+str(num_explanations)+"_explanations"+note
    try:
        os.makedirs(save_dir)
    except: pass

    df_info.to_csv(save_dir+"/info.csv", header=None)

    df_node_masks.to_csv(save_dir+"/node_masks.csv")
    df_node_edge_masks.to_csv(save_dir+"/node_edge_masks.csv")
    df_node_edge_masks_aggr.to_csv(save_dir+"/node_edge_masks_aggr.csv")

    df_feat_masks.to_csv(save_dir+"/feat_masks.csv")
    df_feat_edge_masks.to_csv(save_dir+"/feat_edge_masks.csv")
    df_feat_edge_masks_aggr.to_csv(save_dir+"/feat_edge_masks_aggr.csv")

    df_indiv_masks.to_csv(save_dir+"/indiv_masks.csv")
    if num_explanations > 1:
        df_indiv_masks_std.to_csv(save_dir+"/indiv_masks_std.csv")
    df_indiv_edge_masks.to_csv(save_dir+"/indiv_edge_masks.csv")
    df_indiv_edge_masks_aggr.to_csv(save_dir+"/indiv_edge_masks_aggr.csv")

# entry point
if __name__ == "__main__":
    # arguments: [pdb_id, num_averaged_explanations(optional)]
    main()