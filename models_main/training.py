import random
import time
import torch.nn as nn
from torch.utils.data import DataLoader as DL
from helpers import collate_fn, rmse, pearson, spearman, ci, mse, get_num_parameters
from utils import *
import yaml
from shutil import copyfile
from helpers import model_dict
import sys

with open(sys.argv[1], "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)
seed = sys.argv[2]
random.seed(seed)
torch.manual_seed(int(seed))

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, loss_fn):
    """
    function handling the training for one epoch of an MLP

    model: torch.nn.model
        the model to be trained
    device: torch.device
        indicates whether model is trained on GPU or CPU
    train_loader:
        data loader for training data
    optimizer: torch.optim
        optimizer to train the model with
    epoch: int
        the current epoch
    loss_fn:
        loss function the model is trained w.r.t.
    """
    log_interval = config["training"]["params"]["log_interval"]
    model.train()

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.y),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return loss.item()

def train_MLP(model, device, train_loader, optimizer, epoch, loss_fn):
    """
    function handling the training for one epoch of an MLP

    model: torch.nn.model
        the model to be trained
    device: torch.device
        indicates whether model is trained on GPU or CPU
    train_loader:
        data loader for training data
    optimizer: torch.optim
        optimizer to train the model with
    epoch: int
        the current epoch
    loss_fn:
        loss function the model is trained w.r.t.
    """
    log_interval = config["training"]["params"]["log_interval"]
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()

    for batch_idx, data in enumerate(train_loader):
        data = [d.to(device) if i%2 == 0 else d.type(torch.LongTensor).to(device) for (i, d) in enumerate(data)]
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data[2].view(-1, 1).to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data[2]),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return loss.item()


def _train(model, device, loss_fn, train_loader, valid_loader, test_loader, optimizer, model_file_name, model_output_dir, n_epochs, is_GNN, y_scaler):
    """
    function handling the training process

    model: torch.nn.model
        the model to be trained
    device: torch.device
        indicates whether model is trained on GPU or CPU
    loss_fn:
        loss function according to which model is trained
    train_loader:
        data loader for training data
    valid_loader:
        data loader for validation data
    test_loader:
        data loader for test data
    optimizer: torch.optim
        optimizer to train the model with
    model_file_name: str
        name of the model file
    model_output_dir: str
        directory the output model is saved to
    n_epochs: int
        number of training epochs
    is_GNN: bool
        indicates whether we work with a GNN
    y_scaler: sklearn.preprocessing.StandardScaler
        standard scaler transforming the target variable
    """
    metric = config["training"]["params"]["metric"]

    #commented out early stopping (marc)
    #n_early_stopping = config["training"]["params"]["n_early_stopping"]
    #best_epoch = -1

    #if metric in ["pearson_correlation", "spearman_correlation"]:
    #    best_metric = -1
    #else:
    #    best_metric = np.inf


    train_rmse = list()
    train_pearson = list()
    loss = list()
    valid_rmse = list()
    valid_pearson = list()

    for epoch in range(n_epochs):
        if is_GNN:
            loss_epoch = train(model, device, train_loader, optimizer, epoch + 1, loss_fn)
            G_train, P_train, var_train = predict(model, device, train_loader, y_scaler=y_scaler)
            G, P, var = predict(model, device, valid_loader, y_scaler=y_scaler)
        else:
            loss_epoch = train_MLP(model, device, train_loader, optimizer, epoch + 1, loss_fn)
            G_train, P_train, var_train = predict_MLP(model, device, train_loader, y_scaler=y_scaler)
            G, P, var = predict_MLP(model, device, valid_loader, y_scaler=y_scaler)

        train_rmse.append(rmse(G_train, P_train))
        train_pearson.append(pearson(G_train, P_train))
        loss.append(loss_epoch)
        ret = {"RMSE": rmse(G, P), "MSE": mse(G, P), "pearson_correlation": pearson(G, P),
               "spearman_correlation": spearman(G, P), "CI": ci(G, P)}
        valid_rmse.append(ret["RMSE"])
        valid_pearson.append(ret["pearson_correlation"])

    #save model after epochs are done (new after early stopping was deleted)
    torch.save(model.state_dict(), os.path.join(model_output_dir, model_file_name))


    if is_GNN:
        G_test, P_test, var_test = predict(model, device, test_loader, y_scaler=y_scaler)
    else:
        G_test, P_test, var_test = predict_MLP(model, device, test_loader, y_scaler=y_scaler)
    ret_test = {"RMSE": str(rmse(G_test, P_test)), "MSE": str(mse(G_test, P_test)), "pearson_correlation": str(pearson(G_test, P_test)),
           "spearman_correlation": str(spearman(G_test, P_test)), "CI": str(ci(G_test, P_test))}

    return loss, train_rmse, train_pearson, valid_rmse, valid_pearson, ret_test, G_test, P_test, var_test


def train_NN():
    """
    this function is the main training function.

    path_pretrained_model: str
        path of pretrained model which is to be loaded.
    """

    modeling = model_dict[config["training"]["model"]]
    model_st = modeling.__name__

    if config["preprocessing"]["external_graph"]["use"]:
        model_st = model_st + "_" + config["preprocessing"]["external_graph"]["path"].split("PLIG_")[1].split("_std")[0]

    # if a keyword is given in the sys arguments add is to the model string (hence names of output files are changed)
    if len(sys.argv) > 3:
        keyword = sys.argv[3]
        model_st = model_st + "_" + keyword
    cuda_name = "cuda:0"
    # if len(sys.argv)>3:
    #    cuda_name = "cuda:" + str(int(sys.argv[3]))
    # print('cuda_name:', cuda_name)

    train_batch_size = config["training"]["params"]["train_batch_size"]
    test_batch_size = config["training"]["params"]["test_batch_size"]
    LR = config["training"]["params"]["LR"]
    n_epochs = config["training"]["params"]["n_epochs"]

    print('Train for {} epochs: '.format(n_epochs))

    #this is just for the nameing convention as defined in config/create_general_data
    dataset = config["training"]["dataset"]

    print('Running dataset {} on model {}.'.format(dataset, model_st))

    processed_data_file_train = os.path.join(os.path.join("data", "processed"), dataset + '_train.pt')
    processed_data_file_valid = os.path.join(os.path.join("data", "processed"), dataset + '_valid.pt')
    processed_data_file_test = os.path.join(os.path.join("data", "processed"),  dataset + '_test.pt')

    # the .pt files, i.e. the files containing the ligand-based graphs, must exist in order to run the training
    if (not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test)) or (not os.path.isfile(processed_data_file_valid)):
        if modeling.__name__.startswith("MLPNet")==False:
            print('please run create_general_data.py to prepare Graph data in pytorch format!')
            exit()

    # get timestamp to name the files
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_file_name = timestr + '_model_' + model_st + '_' + dataset + '.model'
    model_output_dir = os.path.join("output", "trained_models")

    # depending on whether we have a GNN or MLP we need different data loaders
    if modeling.__name__.startswith("MLPNet"):
        is_GNN = False
        print("running MLPNet")
    else:
        is_GNN = True
        print("running GNN")
    if is_GNN:
        # TestbedDataset and DataLoader correspond to graphs (as opposed to vectors) as input
        train_data = TestbedDataset(root='data', dataset=dataset+'_train', y_scaler=None)
        valid_data = TestbedDataset(root='data', dataset=dataset+'_valid', y_scaler=train_data.y_scaler)
        test_data = TestbedDataset(root='data', dataset=dataset+'_test', y_scaler=train_data.y_scaler)

        train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    else:
        # FPDataset and DL correspond to vectors (as opposed to graphs) as input (fingerprints)
        train_data = FPDataset(pd_dir=os.path.join("data", dataset + "_train.csv"), config=config, list_training_proteins=None, y_scaler=None)
        valid_data = FPDataset(pd_dir=os.path.join("data", dataset + "_valid.csv"), config=config, list_training_proteins=train_data.get_encoded_proteins(), y_scaler=train_data.y_scaler)
        test_data = FPDataset(pd_dir=os.path.join("data", dataset + "_test.csv"), config=config, list_training_proteins=train_data.get_encoded_proteins(), y_scaler=train_data.y_scaler)

        train_loader = DL(train_data, batch_size = train_batch_size, shuffle = True, collate_fn=collate_fn)
        valid_loader = DL(valid_data, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DL(test_data, batch_size = test_batch_size, shuffle = False, collate_fn=collate_fn)

    protein_processing_type = config["preprocessing"]["protein"]["protein_processing_type"]
    # get the length of the encoing of the protein
    num_features_xt = train_data.get_len_protein_encoding()

    if protein_processing_type == "no_protein":
        print("No protein encoding utilized")
    elif protein_processing_type == "convolution":
        print("The protein encoding is a vector of length ", num_features_xt)
    else:
        print("chosen incorrect protein encoding, please specify either 'no_protein' or 'convolution'. Exiting script now")
        exit()

    # training the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if is_GNN:
        model = modeling(num_features_xd=train_data.num_node_features, num_features_xt=num_features_xt, protein_processing_type=protein_processing_type, featurizer=False)
        model.apply(init_weights)
        if config["training"]["path_model_to_load"] != "None":
            model.load_state_dict(torch.load(config["training"]["path_model_to_load"]))
            print("Model was loaded from ", config["training"]["path_model_to_load"])
        print("The number of node features is ", train_data.num_node_features)
    else:
        model = modeling(num_features_xd = train_data.get_len_ligand_encoding(), num_features_xt=num_features_xt, protein_processing_type=protein_processing_type, featurizer=False)
        model.apply(init_weights)
        print("The dimensionality of the fingerprint is ", train_data.get_len_ligand_encoding())
        # if we have pretrained weights we load them here
        if config["training"]["path_model_to_load"] != "None":
            model.load_state_dict(torch.load(config["training"]["path_model_to_load"]))
            print("Model was loaded from ", config["training"]["path_model_to_load"])

    weight_decay = 0
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)

    model.to(device)
    loss, train_rmse, train_pearson, valid_rmse, valid_pearson, ret_test, G_test, P_test, var_test = \
        _train(model, device, loss_fn, train_loader, valid_loader, test_loader, optimizer, model_file_name, model_output_dir, n_epochs,
               is_GNN, train_data.y_scaler)

    print("Results on the test data: Pearson correlation: {}, RMSE: {}".format(ret_test["pearson_correlation"],
                                                                              ret_test["RMSE"]))
    #print("Number of parameters is ", get_num_parameters(model))

    # only save the results if we haven't done yet
    if not "recursive_call" in locals():
        df_training_history = pd.DataFrame({"loss": loss, "train_rmse": train_rmse, "train_pearson": train_pearson, "valid_rmse": valid_rmse, "valid_pearson": valid_pearson, "test_results": [ret_test]+[None]*(len(valid_pearson)-1)})
        df_training_history.to_csv(os.path.join(os.path.join("output","training_logs"), timestr + '_training_history_model_' + model_st + '_' + dataset + ".csv"))
        # copy the config file to the output directory to save the settings
        copyfile(sys.argv[1], os.path.join("output/training_logs", "config_template.yml"))
        os.rename(os.path.join(os.path.join("output", "training_logs"), "config_template.yml"),
                  os.path.join(os.path.join("output", "training_logs"), timestr+"_config.yml"))

        df = pd.DataFrame(data=G_test, index=range(len(G_test)),
                          columns=["ground_truth"])

        df["prediction"] = P_test
        df.to_csv(os.path.join(os.path.join("output", "training_logs"),
                               timestr + '_test_predictions_' + model_st + '_' + dataset + ".csv"), index=False)


# entry point
if __name__ == "__main__":
    train_NN()

