
import sys

from torch_geometric.data import InMemoryDataset, DataLoader
from torch.utils.data import DataLoader as DL
import yaml
from utils import TestbedDataset, FPDataset
from helpers import *
import pandas as pd


with open(sys.argv[1], "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

pretrained_model_filename = os.path.basename(config["pure_prediction"]["path_to_model"])
all_models = list(model_dict.keys())
found_model = False
# automatically detect model architecture of pretrained model
for model_key in all_models:
    if model_key in pretrained_model_filename:
        model = model_key
        if found_model:
            raise ValueError("Pre-trained model couldn't be uniquely assigned to a model architecture.")
        found_model = True

modeling = model_dict[model]
model_st = modeling.__name__
test_batch_size = config["pure_prediction"]["test_batch_size"]
cuda_name = "cuda:0"

dataset = config["pure_prediction"]["dataset"]

if modeling.__name__.startswith("MLPNet"):
    is_GNN = False
    print("using MLPNet")
else:
    is_GNN = True
    print("using GNN")
if is_GNN:

    test_data = TestbedDataset(root='data', dataset=dataset + '_test', y_scaler=None)
    # make data PyTorch mini-batch processing ready

    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
else:


    test_data = FPDataset(pd_dir=os.path.join("data", dataset + "_test.csv"), config=config,
                            list_training_proteins=None, y_scaler=None)

    test_loader = DL(test_data, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)

protein_processing_type = config["preprocessing"]["protein"]["protein_processing_type"]
num_features_xt = test_data.get_len_protein_encoding() #changed train_data to test_data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if is_GNN:
    model = modeling(num_features_xd = test_data.num_node_features,  num_features_xt=num_features_xt, protein_processing_type=protein_processing_type) #changed train_data to test data
else:
    model = modeling(num_features_xd = test_data.get_len_ligand_encoding(), num_features_xt=num_features_xt, protein_processing_type=protein_processing_type) #changed train_data to test_data

model.load_state_dict(torch.load(config["pure_prediction"]["path_to_model"]))
pretrained_model_filename = os.path.basename(config["pure_prediction"]["path_to_model"])
print("Model was loaded from ", config["pure_prediction"]["path_to_model"])

# initialize tensors to save results into
total_preds_test = torch.Tensor()
total_labels_test = torch.Tensor()

with torch.no_grad():
    # compute performance on test data
    print('Make prediction for {} samples...'.format(len(test_loader.dataset)))
    y_scaler = None
    for data in test_loader:
        if is_GNN:
            data = data.to(device)
        else:
            data = [d.to(device) if i % 2 == 0 else d.type(torch.LongTensor).to(device) for (i, d) in enumerate(data)]

        output = model(data)

        if is_GNN:
            total_preds_test = torch.cat((total_preds_test, output.cpu()), 0)
            total_labels_test = torch.cat((total_labels_test, data.y.view(-1, 1).cpu()), 0)
        else:
            total_preds_test = torch.cat((total_preds_test, output.cpu()), 0)
            total_labels_test = torch.cat((total_labels_test, data[2].cpu()), 0)

    G_test = total_labels_test.numpy().flatten()
    P_test = total_preds_test.numpy().flatten()

    ret_test = {"RMSE": str(rmse(G_test, P_test)), "MSE": str(mse(G_test, P_test)),
                "pearson_correlation": str(pearson(G_test, P_test)),
                "spearman_correlation": str(spearman(G_test, P_test)), "CI": str(ci(G_test, P_test))}
    print("Results on the test data: Pearson correlation: {}, RMSE: {}".format(ret_test["pearson_correlation"],
                                                                              ret_test["RMSE"]))

df_test = pd.DataFrame(data=G_test, index=range(len(G_test)),
                          columns=["ground_truth"])
df_test["prediction"] = P_test
df_test.to_csv(os.path.join(os.path.join("output", "training_logs"),
                               "pure_predictions_test_set_" + pretrained_model_filename + ".csv"), index=False)

df_test = pd.DataFrame.from_dict(ret_test, orient='index', columns=["Test"])

df_to_save = df_test
df_to_save.to_csv(os.path.join(os.path.join("output", "training_logs"),
                               "performance_stats_" + pretrained_model_filename + ".csv"), index=True)
