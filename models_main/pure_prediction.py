
import sys

from torch_geometric.data import InMemoryDataset, DataLoader
from torch.utils.data import DataLoader as DL
import yaml
from utils import TestbedDataset, ECFPDataset
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
    train_data = TestbedDataset(root='data', dataset=dataset + '_train', y_scaler=None)
    valid_data = TestbedDataset(root='data', dataset=dataset + '_valid', y_scaler=train_data.y_scaler)
    test_data = TestbedDataset(root='data', dataset=dataset + '_test', y_scaler=train_data.y_scaler)

    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=test_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
else:

    train_data = ECFPDataset(pd_dir=os.path.join("data", dataset + "_train.csv"), config=config,
                             list_training_proteins=None, y_scaler=None)
    valid_data = ECFPDataset(pd_dir=os.path.join("data", dataset + "_valid.csv"), config=config,
                             list_training_proteins=train_data.get_encoded_proteins(), y_scaler=train_data.y_scaler)
    test_data = ECFPDataset(pd_dir=os.path.join("data", dataset + "_test.csv"), config=config,
                            list_training_proteins=train_data.get_encoded_proteins(), y_scaler=train_data.y_scaler)
    train_loader = DL(train_data, batch_size=test_batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DL(valid_data, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DL(test_data, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)

protein_processing_type = config["preprocessing"]["protein"]["protein_processing_type"]
num_features_xt = train_data.get_len_protein_encoding()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if is_GNN:
    model = modeling(num_features_xd = train_data.num_node_features, num_features_xt=num_features_xt, protein_processing_type=protein_processing_type)
else:
    model = modeling(num_features_xd = train_data.get_len_ligand_encoding(), num_features_xt=num_features_xt, protein_processing_type=protein_processing_type)

model.load_state_dict(torch.load(config["pure_prediction"]["path_to_model"]))
pretrained_model_filename = os.path.basename(config["pure_prediction"]["path_to_model"])
print("Model was loaded from ", config["pure_prediction"]["path_to_model"])

# initialize tensors to save results into
total_preds_train = torch.Tensor()
total_labels_train = torch.Tensor()
total_preds_valid = torch.Tensor()
total_labels_valid = torch.Tensor()
total_preds_test = torch.Tensor()
total_labels_test = torch.Tensor()

with torch.no_grad():
    # compute performance on training data
    print('Make prediction for {} samples...'.format(len(train_loader.dataset)))
    y_scaler = train_data.y_scaler
    for data in train_loader:
        #print(data)
        if is_GNN:
            data = data.to(device)
        else:
            data = [d.to(device) if i % 2 == 0 else d.type(torch.LongTensor).to(device) for (i, d) in enumerate(data)]

        output = model(data)

        if is_GNN:
            total_preds_train = torch.cat((total_preds_train, output.cpu()), 0)
            total_labels_train = torch.cat((total_labels_train, data.y.view(-1, 1).cpu()), 0)
        else:
            total_preds_train = torch.cat((total_preds_train, output.cpu()), 0)
            total_labels_train = torch.cat((total_labels_train, data[2].cpu()), 0)

    G_train = y_scaler.inverse_transform(total_labels_train.numpy().flatten())
    P_train = y_scaler.inverse_transform(total_preds_train.numpy().flatten())

    ret_train = {"RMSE": str(rmse(G_train, P_train)), "MSE": str(mse(G_train, P_train)),
                "pearson_correlation": str(pearson(G_train, P_train)),
                "spearman_correlation": str(spearman(G_train, P_train)), "CI": str(ci(G_train, P_train))}
    print("Results on the training data: Pearson correlation: {}, RMSE: {}".format(ret_train["pearson_correlation"],
                                                                               ret_train["RMSE"]))

    # compute performance on validation data
    print('Make prediction for {} samples...'.format(len(valid_loader.dataset)))
    for data in valid_loader:
        if is_GNN:
            data = data.to(device)
        else:
            data = [d.to(device) if i % 2 == 0 else d.type(torch.LongTensor).to(device) for (i, d) in enumerate(data)]

        output = model(data)

        if is_GNN:
            total_preds_valid = torch.cat((total_preds_valid, output.cpu()), 0)
            total_labels_valid = torch.cat((total_labels_valid, data.y.view(-1, 1).cpu()), 0)
        else:
            total_preds_valid = torch.cat((total_preds_valid, output.cpu()), 0)
            total_labels_valid = torch.cat((total_labels_valid, data[2].cpu()), 0)

    G_valid = y_scaler.inverse_transform(total_labels_valid.numpy().flatten())
    P_valid = y_scaler.inverse_transform(total_preds_valid.numpy().flatten())

    #print(len(G_valid), len(P_valid))
    ret_valid = {"RMSE": str(rmse(G_valid, P_valid)), "MSE": str(mse(G_valid, P_valid)),
                 "pearson_correlation": str(pearson(G_valid, P_valid)),
                 "spearman_correlation": str(spearman(G_valid, P_valid)), "CI": str(ci(G_valid, P_valid))}
    print("Results on the validation data: Pearson correlation: {}, RMSE: {}".format(ret_valid["pearson_correlation"],
                                                                               ret_valid["RMSE"]))

    # compute performance on test data
    print('Make prediction for {} samples...'.format(len(test_loader.dataset)))
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

    G_test = y_scaler.inverse_transform(total_labels_test.numpy().flatten())
    P_test = y_scaler.inverse_transform(total_preds_test.numpy().flatten())

    ret_test = {"RMSE": str(rmse(G_test, P_test)), "MSE": str(mse(G_test, P_test)),
                "pearson_correlation": str(pearson(G_test, P_test)),
                "spearman_correlation": str(spearman(G_test, P_test)), "CI": str(ci(G_test, P_test))}
    print("Results on the test data: Pearson correlation: {}, RMSE: {}".format(ret_test["pearson_correlation"],
                                                                              ret_test["RMSE"]))

df_train = pd.DataFrame(data=G_train, index=range(len(G_train)),
                          columns=["ground_truth"])
df_train["prediction"] = P_train
df_train.to_csv(os.path.join(os.path.join("output", "training_logs"),
                               "pure_predictions_train_set_" + pretrained_model_filename + ".csv"), index=False)

df_valid = pd.DataFrame(data=G_valid, index=range(len(G_valid)),
                          columns=["ground_truth"])
df_valid["prediction"] = P_valid
df_valid.to_csv(os.path.join(os.path.join("output", "training_logs"),
                               "pure_predictions_valid_set_" + pretrained_model_filename + ".csv"), index=False)

df_test = pd.DataFrame(data=G_test, index=range(len(G_test)),
                          columns=["ground_truth"])
df_test["prediction"] = P_test
df_test.to_csv(os.path.join(os.path.join("output", "training_logs"),
                               "pure_predictions_test_set_" + pretrained_model_filename + ".csv"), index=False)

df_train = pd.DataFrame.from_dict(ret_train, orient='index', columns=["Train"])
df_valid = pd.DataFrame.from_dict(ret_valid, orient='index', columns=["Valid"])
df_test = pd.DataFrame.from_dict(ret_test, orient='index', columns=["Test"])

df_to_save = pd.concat([df_train, df_valid, df_test], axis=1)
df_to_save.to_csv(os.path.join(os.path.join("output", "training_logs"),
                               "performance_stats_" + pretrained_model_filename + ".csv"), index=True)