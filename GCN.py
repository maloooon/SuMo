import numpy as np
import torch
import os
import DataInputNew
import ReadInData
import torchtuples as tt
import pandas as pd
import time
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, SAGPooling
from torch_geometric.nn import global_max_pool as gmp
import math
import matplotlib.pyplot as plt





class GCN(nn.Module):
    """https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv"""
    def __init__(self, num_nodes, edge_index, in_features, n_hidden_layer_dims, activ_funcs, dropout_prob = 0.1, ratio = 0.1):
        super(GCN, self).__init__()
        self.num_nodes = num_nodes # number of proteins
        self.edge_index = edge_index
        self.ratio = ratio # SAGPooling ratio
        self.in_features = in_features # how many features per node
        self.n_hidden_layer_dims = n_hidden_layer_dims
        self.activ_funcs = activ_funcs
        self.dropout_prob = dropout_prob
        self.hidden_layers = nn.ParameterList([])
        self.params_for_print = nn.ParameterList([])


        for c,afunc in enumerate(activ_funcs):
                if afunc.lower() == 'relu':
                    activ_funcs[c] = nn.ReLU()
                elif afunc.lower() == 'sigmoid':
                    activ_funcs[c] = nn.Sigmoid()



        self.conv1 = GraphConv(in_features, in_features)
        self.params_for_print.append(self.conv1)
        self.params_for_print.append(nn.ReLU) # TODO : not tested
        self.pool1 = SAGPooling(in_features, ratio=ratio, GNN=GraphConv) # in channel same as out of conv1
        self.params_for_print.append(self.pool1)

        first_in = math.ceil(ratio * (in_features * num_nodes))
        for c in range(len(n_hidden_layer_dims) +1):
            if c == 0: # first layer
                self.hidden_layers.append(nn.Sequential(nn.Linear(first_in, n_hidden_layer_dims[0]),
                                                        nn.BatchNorm1d(n_hidden_layer_dims[0]),
                                                        activ_funcs[0]))
                self.params_for_print.append(self.hidden_layers[-1])

            elif c == len(n_hidden_layer_dims): # last layer (no activation function)
                self.hidden_layers.append(nn.Sequential(nn.Linear(n_hidden_layer_dims[-1], 1),
                                                        nn.BatchNorm1d(1),
                                                        ))
                self.params_for_print.append(self.hidden_layers[-1])
            else: # other layers
                self.hidden_layers.append(nn.Sequential(nn.Linear(n_hidden_layer_dims[c-1], n_hidden_layer_dims[c]),
                                                        nn.BatchNorm1d(n_hidden_layer_dims[c]),
                                                        activ_funcs[c]))
                self.params_for_print.append(self.hidden_layers[-1])




        print("Model: ", self.params_for_print)


        self.batches = {}
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, data):
        batch_size = data.shape[0]
        x = data[:, :self.num_nodes * self.in_features]
        x = x.reshape(batch_size, self.num_nodes, self.in_features)
        if batch_size not in self.batches:
            l = []
            for i in range(batch_size):
                l.append(Data(x=x[i], edge_index=self.edge_index))
            batch = Batch.from_data_list(l)
            self.batches[batch_size] = batch

        batch = self.batches[batch_size]
        x = x.reshape(-1, self.in_features)
        x = F.relu(self.conv1(x=x, edge_index=batch.edge_index))
        x, edge_index, _, batch, perm, score = self.pool1(
            x, batch.edge_index, None, batch.batch)
        x = x.view(batch_size, -1)

        for layer in self.hidden_layers:
            x = layer(x)

        return x


# https://github.com/bio-ontology-research-group/DeepMOCCA/blob/master/step-by-step/deepmocca_training.ipynb
def normalize(data, minx=None, maxx=None):
    if minx is None:
        minx = torch.min(data)
        maxx = torch.max(data)
    if minx == maxx:
        return data
    return (data - minx) / (maxx - minx)

def normalize_by_row(data):
    for i in range(data.shape[0]):
        data[i, :] = normalize(data[i, :])
    return data

def normalize_by_column(data):
    for i in range(data.shape[1]):
        data[:, i] = normalize(data[:, i])
    return data




def train(module,views, batch_size =128, n_epochs = 512, lr_scheduler_type = 'onecyclecos', l2_regularization = False, feature_names = None, batch_size_validation=20):
    """

    :param module: basically the dataset to be used
    :param batch_size: batch size for training
    :param n_epochs: number of epochs
    :param lr_scheduler_type: type of lr_scheduler : 'lambda','mulitplicative','step','multistep','exponential',...
    :param l2_regularization : bool whether should be applied or not
    """




    # Setup all the data
    n_train_samples, n_test_samples, n_val_samples = module.setup()



    #Select method for feature selection
    edge_index, proteins_used = module.feature_selection('ppi', feature_names)

    # Load Dataloaders
    trainloader = module.train_dataloader(batch_size=n_train_samples) # all training examples
    testloader =module.test_dataloader(batch_size=n_test_samples)
    valloader = module.validation_dataloader(batch_size=n_val_samples)

    # Load data and set device to cuda if possible

    #Train
    for train_data, train_duration, train_event in trainloader:

        train_data.to(device=device)
        train_duration.to(device=device)
        train_event.to(device=device)

    print("Train data shape after feature selection {}".format(train_data.shape))

    #Validation
    for val_data, val_duration, val_event in valloader:

        val_data.to(device=device)
        val_duration.to(device=device)
        val_event.to(device=device)


    print("Validation data shape after feature selection {}".format(val_data.shape))

    #Test
    for test_data, test_duration, test_event in testloader:

        test_data.to(device=device)
        test_duration.to(device=device)
        test_event.to(device=device)


    print("Test data shape after feature selection {}".format(test_data.shape))

    views_with_proteins = train_data.size(2)

    num_features = views_with_proteins
    num_nodes = len(proteins_used)


    # normalizing data by rows (genes)

    # reshape not necessary as data already implemented in this manner
    # features_used = features_used.reshape(-1, num_nodes, num_features)
    # first dimensions (samples) is derived, so that 2/3 dimension we have [...], [...]
    # where each list has num_features elements and we have num_nodes in total

    for i in range(num_features):
        train_data[:,:,i] = normalize(train_data[:,:,i])
        val_data[:,:,i] = normalize(val_data[:,:,i])
        test_data[:,:,i] = normalize(test_data[:,:,i])


    # reshape structure for use of GCN
    train_data = train_data.reshape(-1, num_nodes * num_features)
    val_data = val_data.reshape(-1, num_nodes * num_features)
    test_data = test_data.reshape(-1, num_nodes * num_features)

    callbacks = [tt.callbacks.EarlyStopping(patience=10)]

    val_full = (val_data, (val_duration,val_event))
    train_de_pycox = (train_duration, train_event)


    torch.manual_seed(0)
    edge_index = torch.LongTensor(edge_index).to(device)
    net = GCN(len(proteins_used), edge_index, in_features=num_features,n_hidden_layer_dims=[1024,512,256],
              activ_funcs=['sigmoid','sigmoid','sigmoid']).to(device)

    model = CoxPH(net, tt.optim.Adam(0.001))

    log = model.fit(train_data,train_de_pycox, batch_size, n_epochs, callbacks, verbose=True,
                    val_data=val_full, val_batch_size= batch_size_validation)

    train = train_data, train_de_pycox

    _ = model.compute_baseline_hazards(*train)

    surv = model.predict_surv_df(test_data)


    # Plot it
    surv.iloc[:, :5].plot()
    plt.ylabel('S(t | x)')
    _ = plt.xlabel('Time')

    # test_dur and test_ev need to be numpy arrays for EvalSurv()
    test_duration = test_duration.numpy()
    test_event = test_event.numpy()

    ev = EvalSurv(surv, test_duration, test_event, censor_surv='km')


    # concordance
    concordance_index = ev.concordance_td()

    #brier score
    time_grid = np.linspace(test_duration.min(), test_duration.max(), 100)
    _ = ev.brier_score(time_grid).plot
    brier_score = ev.integrated_brier_score(time_grid)

    #binomial log-likelihood
    binomial_score = ev.integrated_nbll(time_grid)

    print("Concordance index : {} , Integrated Brier Score : {} , Binomial Log-Likelihood : {}".format(concordance_index,
                                                                                                       brier_score,
                                                                                                       binomial_score))





if __name__ == '__main__':
    cancer_data = ReadInData.readcancerdata()
    multimodule = DataInputNew.SurvMultiOmicsDataModule(cancer_data[0][0],cancer_data[0][1],cancer_data[0][2])
    views = cancer_data[0][2]
    feature_names = cancer_data[0][3]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(module= multimodule,views= views, l2_regularization=True, feature_names=feature_names,batch_size=16,n_epochs=100,
          batch_size_validation=5)




