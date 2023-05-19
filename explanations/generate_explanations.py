# imports
import sys
import os

import yaml
from utils import TestbedDataset 
from helpers import *
import pandas as pd

from explain_utils.gnn_explainer  import GNNExplainer
from explain_utils.explainable_model import GATNet_PLIG_no_p
from tqdm import tqdm

from rdkit import Chem
from explain_utils.aggr_edge_directions import aggregate_edge_directions

# # %% TESTING CELL ONLY
# with open("GATNet_config.yml", "r") as ymlfile:
#     config = yaml.load(ymlfile, Loader=yaml.FullLoader)

# test_df = pd.read_csv(os.path.join("data", "PDBbind_combined_test.csv"))
# pdb_id = "5dwr".lower()
# assert pdb_id in list(test_df["Identifier"])
# num_explanations = 10

def main():
    # process arguments
    with open(sys.argv[1], "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    test_df = pd.read_csv(os.path.join("data", "PDBbind_combined_test.csv"))
    pdb_id = sys.argv[2].lower() 
    assert pdb_id in list(test_df["Identifier"]), f"{pdb_id} not in test set" 
    if len(sys.argv) >= 4:
        num_explanations = int(sys.argv[3])
        assert num_explanations > 0, "Explain at least once."
    else:
        num_explanations = 1
    if len(sys.argv) >= 5:
        note = sys.argv[4]
    else:
        note = ""

    # load test_set
    dataset = config["pure_prediction"]["dataset"]

    test_data = TestbedDataset(root='data', dataset=dataset + '_test', y_scaler=None) # changed from train_data.y_scaler to none

    protein_processing_type = config["preprocessing"]["protein"]["protein_processing_type"]
    num_features_xt = test_data.get_len_protein_encoding() # changed train_data to test_data
    model = GATNet_PLIG_no_p(num_features_xd = test_data.num_node_features, num_features_xt=num_features_xt, protein_processing_type=protein_processing_type)

    model.load_state_dict(torch.load(config["pure_prediction"]["path_to_model"]))
    print("Model training data was loaded from ", config["pure_prediction"]["path_to_model"])


    # get graph and smiles from pdb_id
    graph = None
    for g in test_data:
        if g.pdb_id == pdb_id:
            graph = g
    smiles = test_df.loc[test_df["Identifier"] == pdb_id]["compound_iso_smiles"].iloc[0]

    # calculate prediction
    y_scaler = test_data.y_scaler
    model.eval()
    pred = model(graph.x, graph.edge_index,torch.zeros(graph.x.shape[0], dtype=int, device=graph.x.device))
    pred = y_scaler.inverse_transform(pred.detach().numpy()).item()
    label = y_scaler.inverse_transform(graph.y.unsqueeze(0).numpy()).item()
    print(f"Model predicted a value of [{pred:.4f}] for label [{label:.4f}] on {pdb_id.upper()}.")

    # get atoms, bonds and features
    feature_type =['Heavy Neighb.', 'H Neighb.', 'Explicit Valence', 'Aromaticity', 'In Ring',
                'C;4;3;0;0;0', 'C;4;3;1;0;0', 'C;4;1;3;0;0', 'N;3;2;1;0;0', 'O;2;1;0;0;0',
                'C;4;2;2;0;0', 'C;6;3;0;0;0', 'N;4;2;1;0;0', 'N;4;1;2;0;0', 'N;3;1;2;0;0',
                'C;5;3;0;0;0', 'S;2;1;1;0;0', 'C;4;2;1;1;1', 'C;4;3;0;1;1', 'N;3;2;0;1;1',
                'N;3;2;1;1;1', 'N;4;1;3;0;0', 'S;2;2;0;0;0', 'C;4;3;1;0;1', 'C;4;2;2;0;1',
                'N;3;3;0;0;1', 'O;2;1;1;0;0'] 
    mol = Chem.MolFromSmiles(smiles)
    atoms = [x for x in range(mol.GetNumAtoms())]

    # create df_info
    df_info = pd.DataFrame({"PDB_ID": [pdb_id],
                            "SMILES": [smiles],
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
            # timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = pdb_id+"_"+str(num_explanations)+"_explanations"+note
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
    # arguments: [config file, pdb_id, num_averaged_explanations(optional)]
    main()