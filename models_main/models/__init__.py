import torch.nn.functional as F
import torch.nn as nn

activation_function_dict = {"relu": F.relu, "leaky_relu": F.leaky_relu, "sigmoid": F.sigmoid}
activation_mlp_dict = {"relu": nn.ReLU(), "leaky_relu": nn.LeakyReLU(), "sigmoid": nn.Sigmoid()}
