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





class GCN(nn.Module):
    """https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv"""
    def __init__(self, num_nodes, edge_index, in_features):
        super(GCN, self).__init__()
        self.num_nodes = num_nodes # number of proteins
        self.edge_index = edge_index
        self.in_features = in_features # how many features per node (?) ; in_channel = -1 derive from first input (?)
        self.conv1 = GraphConv(in_features, 2) #in_channels / out_channels
        self.pool1 = SAGPooling(2, ratio=0.1, GNN=GraphConv) # in channel same as out of conv1
        self.fc1 = nn.Linear(1230, 1024, bias=False) #in_channel --> feature size after SAGPooling (not clear yet how do get to this number)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, 512, bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(512, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(1)
        self.dropout3 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.batches = {}

    def forward(self, data):
        batch_size = data.shape[0]
        x = data[:, :self.num_nodes * 2] # * 6 bc 6 features per protein, so in my case 2 (?)
        x = x.reshape(batch_size, self.num_nodes, 2)
        if batch_size not in self.batches:
            l = []
            for i in range(batch_size):
                l.append(Data(x=x[i], edge_index=self.edge_index))
            batch = Batch.from_data_list(l)
            self.batches[batch_size] = batch
            print('ok')
        batch = self.batches[batch_size]
        x = x.reshape(-1, 2)
        x = F.relu(self.conv1(x=x, edge_index=batch.edge_index))
        x, edge_index, _, batch, perm, score = self.pool1(
            x, batch.edge_index, None, batch.batch)
        x = x.view(batch_size, -1)
        # print(x.shape)
        x = self.dropout1(self.bn1(torch.relu(self.fc1(x))))
        x = self.dropout2(self.bn2(torch.relu(self.fc2(x))))
        x = self.dropout3(self.bn3(self.fc3(x)))
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




def train(module,views, batch_size =25, n_epochs = 512, lr_scheduler_type = 'onecyclecos', l2_regularization = False, feature_names = None):
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
    features_used, proteins_used, edge_index = module.feature_selection('ppi', feature_names)

    # Load Dataloaders
    trainloader = module.train_dataloader(batch_size=n_train_samples) # all training examples
    testloader =module.test_dataloader(batch_size=n_test_samples)
    valloader = module.validation_dataloader(batch_size=n_val_samples)

    # Load data and set device to cuda if possible

    #Train
    for train_data, train_duration, train_event in trainloader:
        for view in range(len(train_data)):
            train_data[view] = train_data[view].to(device=device)

        train_duration.to(device=device)
        train_event.to(device=device)

    for c,_ in enumerate(train_data):
        print("Train data shape after feature selection {}".format(train_data[c].shape))

    #Validation
    for val_data, val_duration, val_event in valloader:
        for view in range(len(val_data)):
            val_data[view] = val_data[view].to(device=device)

        val_duration.to(device=device)
        val_event.to(device=device)

    for c,_ in enumerate(val_data):
        print("Validation data shape after feature selection {}".format(val_data[c].shape))

    #Test
    for test_data, test_duration, test_event in testloader:
        for view in range(len(test_data)):
            test_data[view] = test_data[view].to(device=device)


        test_duration.to(device=device)
        test_event.to(device=device)

    for c,_ in enumerate(test_data):
        print("Test data shape after feature selection {}".format(test_data[c].shape))



    # Input dimensions (features for each view) for NN based on different data (train/validation/test)
    # Need to be the same for NN to work
    dimensions_train = [x.size(1) for x in train_data]
    dimensions_val = [x.size(1) for x in val_data]
    dimensions_test = [x.size(1) for x in test_data]

    assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/val/test'

    dimensions = dimensions_train

    # Get feature offsets for train/validation/test
    # Need to be the same for NN to work
    feature_offsets_train = [0] + np.cumsum(dimensions_train).tolist()
    feature_offsets_val = [0] + np.cumsum(dimensions_val).tolist()
    feature_offsets_test = [0] + np.cumsum(dimensions_test).tolist()

    feature_offsets = feature_offsets_train

    # Number of all features (summed up) for train/validation/test
    # These need to be the same, otherwise NN won't work
    feature_sum_train = feature_offsets_train[-1]
    feature_sum_val = feature_offsets_val[-1]
    feature_sum_test = feature_offsets_test[-1]

    feature_sum = feature_sum_train

    # Initialize empty tensors to store the data for train/validation/test
    train_data_pycox = torch.empty(n_train_samples, feature_sum_train).to(torch.float32)
    val_data_pycox = torch.empty(n_val_samples, feature_sum_val).to(torch.float32)
    test_data_pycox = torch.empty(n_test_samples, feature_sum_test).to(torch.float32)

    # Train
    for idx_view,view in enumerate(train_data):
        for idx_sample, sample in enumerate(view):
            train_data_pycox[idx_sample][feature_offsets_train[idx_view]:
                                         feature_offsets_train[idx_view+1]] = sample

    # Validation
    for idx_view,view in enumerate(val_data):
        for idx_sample, sample in enumerate(view):
            val_data_pycox[idx_sample][feature_offsets_val[idx_view]:
                                       feature_offsets_val[idx_view+1]] = sample

    # Test
    for idx_view,view in enumerate(test_data):
        for idx_sample, sample in enumerate(view):
            test_data_pycox[idx_sample][feature_offsets_test[idx_view]:
                                        feature_offsets_test[idx_view+1]] = sample


    # Turn validation (duration,event) in correct structure for pycox .fit() call
    # dde : data duration event ; de: duration event ; d : data
    train_duration_numpy = train_duration.detach().cpu().numpy()
    train_event_numpy = train_event.detach().cpu().numpy()
    val_duration_numpy = val_duration.detach().cpu().numpy()
    val_event_numpy = val_event.detach().cpu().numpy()



    train_de_pycox = (train_duration, train_event)
    val_dde_pycox = val_data_pycox, (val_duration, val_event)
    val_de_pycox = (val_duration, val_event)
    test_d_pycox = test_data_pycox

    train_data_pycox_numpy = train_data_pycox.detach().cpu().numpy()
    val_data_pycox_numpy = val_data_pycox.detach().cpu().numpy()
    train_de_pycox_numpy = (train_duration_numpy, train_event_numpy)#event_temporary_placeholder_train_numpy)
    val_de_pycox_numpy = (val_duration_numpy, val_event_numpy)
    train_ded_pycox = tt.tuplefy(train_de_pycox, train_data_pycox) # TODO : Problem hier wegen (221,) und (221,20) als shapes

    full_train = tt.tuplefy(train_data_pycox, (train_de_pycox, train_data_pycox))
    full_validation = tt.tuplefy(val_data_pycox, (val_de_pycox, val_data_pycox))




    num_features = 2  # only mRNA & DNA data
    num_nodes = len(proteins_used)



    # normalizing data by rows (genes)

    # reshape not necessary as data already implemented in this manner
    # features_used = features_used.reshape(-1, num_nodes, num_features) # first dimensions (samples) is derived, so that 2/3 dimension we have [...], [...] where each list has num_features elements and we have num_nodes in total
    for i in range(num_features):
        features_used[:,:,i] = normalize(features_used[:,:,i])

    features_used = features_used.reshape(-1, num_nodes * num_features)

    callbacks = [tt.callbacks.EarlyStopping(patience=10)]


    torch.manual_seed(0)
    edge_index = torch.LongTensor(edge_index).to(device)
    net = GCN(len(proteins_used), edge_index, in_features=num_features).to(device)

    model = CoxPH(net, tt.optim.Adam(0.001))

    log = model.fit(features_used,train_de_pycox, batch_size, n_epochs, callbacks, verbose=True)

    train = features_used, train_de_pycox

    _ = model.compute_baseline_hazards(*train)












if __name__ == '__main__':
    cancer_data = ReadInData.readcancerdata()
    multimodule = DataInputNew.SurvMultiOmicsDataModule(cancer_data[0][0],cancer_data[0][1],cancer_data[0][2])
    views = cancer_data[0][2]
    feature_names = cancer_data[0][3]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(module= multimodule,views= views, l2_regularization=True, feature_names=feature_names,batch_size=32,n_epochs=100)




