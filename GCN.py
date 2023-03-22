import numpy as np
import torch
import torchtuples as tt
import torch.nn as nn
import torch.nn.functional as F
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from torch_geometric.data import Data,Batch
from torch_geometric.nn import GraphConv, SAGPooling
import math
from torch.optim import Adam
import pandas as pd
import os
import optuna
from torch_geometric.nn import global_max_pool as gmp
import copy



class GCN(nn.Module):
    """
    GCN with changeable hyperparameters. The input is passed through Graph Convolutional Layer(s) and Self Attention
    Pooling. Finally, it is connected to a FCNN which compresses values to a single dimensional
    value, used for the Proportional Hazards Model.
    """
    def __init__(self, num_nodes, edge_index, in_features, n_hidden_layer_dims, activ_funcs,
                 dropout_prob = 0.1, ratio = 0.4, dropout_bool = False, batchnorm_bool = False,
                 dropout_layers = None, batchnorm_layers = None,
                 activ_funcs_graphconv = None, graphconvs = None, prelu_init = 0.25, print_bool = False):
        """

        :param num_nodes: Number of nodes (proteins) ; dtype : Int
        :param edge_index: Protein Pairs with Interactions ; dtype : List of Lists of Ints [Indices]
        :param in_features: Number of features per proteins ; dtype : Int
        :param n_hidden_layer_dims: Hidden layers dimensions of the FCNN ; dtype : List of Ints
        :param activ_funcs: Activation Functions aswell as for the last layer, the last layer can have no activation
         function ['none‘]  ; dtype : List of Lists of Strings ['relu', 'sigmoid', 'prelu']
        :param dropout_prob: Probability of Neuron Dropouts ; dtype : Int
        :param ratio: Graph pooling ratio for self attention pooling layer ; dtype : Float
        :param dropout_bool: Decide whether Dropout is to be applied or not ; dtype : Boolean
        :param batchnorm_bool: Decide whether Batch Normalization is to be applied or not ; dtype : Boolean
        :param dropout_layers: Layers in which to apply Dropout ; dtype : List of Lists of Strings ['yes','no']
        :param batchnorm_layers: Layers in which to apply Batch Normalization ; dtype : List of Lists of Strings ['yes','no']
        :param activ_funcs_graphconv: Activation Functions for the Graph Convolutional Layers ; dtype : List of Strings ['relu', 'sigmoid']
        :param graphconvs: Dimensions of Layers for the Graph Convolutional Layers ; dtype : List of Ints
        :param prelu_init: Initial Value for PreLU activation ; dtype : Float [between 0 and 1]
        :param print_bool: Decide whether to print the models structure ; dtype : Boolean
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super(GCN, self).__init__()
        self.num_nodes = num_nodes # number of proteins
        self.edge_index = edge_index.to(device)
        self.ratio = ratio # SAGPooling ratio
        self.in_features = in_features # how many features per node
        self.n_hidden_layer_dims = n_hidden_layer_dims
        self.activ_funcs = activ_funcs
        self.dropout_prob = dropout_prob
        self.dropout_bool = dropout_bool
        self.batchnorm_bool = batchnorm_bool
        self.dropout_layers = dropout_layers
        self.batchnorm_layers = batchnorm_layers
        self.hidden_layers = nn.ParameterList([])
        self.params_for_print = nn.ParameterList([])
        self.activ_funcs_graphconv = activ_funcs_graphconv
        self.graphconvs = graphconvs # 1 or 2 graph conv layers ; inputs are the wanted out_features
        self.prelu_init = prelu_init
        self.print_bool = print_bool

        # Replace strings with actual activation functions
        for c,afunclst in enumerate(activ_funcs):
            for c2, afunc in enumerate(afunclst):
                if afunc.lower() == 'relu':
                    activ_funcs[c][c2] = nn.ReLU()
                elif afunc.lower() == 'sigmoid':
                    activ_funcs[c][c2] = nn.Sigmoid()
                elif afunc.lower() == 'prelu':
                    activ_funcs[c][c2] = nn.PReLU(init= prelu_init)










        self.conv1 = GraphConv(in_features, graphconvs[0])

        self.pool1 = SAGPooling(graphconvs[0], ratio=ratio, GNN=GraphConv) # in channel same as out of conv1

        if len(graphconvs) == 2: # Two GraphConv layers, only 1 pooling layer
            self.conv2 = GraphConv(graphconvs[0], graphconvs[1])
            self.first_in = graphconvs[1]


        else: # In this case pooling layer last layer before we apply FCNN ; -1 added after testing  as rounding up
            # mostly differs by one for some reason
            self.first_in = math.ceil(ratio * (graphconvs[0] * num_nodes))



        self.params_for_print.append(self.pool1)

        # Assign Layers
        for c in range(len(n_hidden_layer_dims) +1):
            # First layer
            if c == 0:
                # Batch normalization
                if batchnorm_bool == True and batchnorm_layers[0][c] == 'yes':
                    self.hidden_layers.append(nn.Sequential(nn.Linear(self.first_in, n_hidden_layer_dims[0]),
                                                            nn.BatchNorm1d(n_hidden_layer_dims[0]),
                                                            activ_funcs[0][0]))
                    self.params_for_print.append(self.hidden_layers[-1])
                # No Batch normalization
                else:
                    self.hidden_layers.append(nn.Sequential(nn.Linear(self.first_in, n_hidden_layer_dims[0]),
                                                            activ_funcs[0][0]))
                    self.params_for_print.append(self.hidden_layers[-1])

            # Last layer
            elif c == len(n_hidden_layer_dims):
                # Batch normalization
                if batchnorm_bool == True and batchnorm_layers[-1][0] == 'yes':
                    # Activation function
                    if activ_funcs[-1][0] != 'none':
                        self.hidden_layers.append(nn.Sequential(nn.Linear(n_hidden_layer_dims[-1], 1),
                                                                nn.BatchNorm1d(1), activ_funcs[-1][0]
                                                                ))
                        self.params_for_print.append(self.hidden_layers[-1])
                    # No activation function
                    else:
                        self.hidden_layers.append(nn.Sequential(nn.Linear(n_hidden_layer_dims[-1], 1),
                                                                nn.BatchNorm1d(1)
                                                                ))
                        self.params_for_print.append(self.hidden_layers[-1])
                # No batch normalization
                else:
                    # Activation function
                    if activ_funcs[-1][0] != 'none':
                        self.hidden_layers.append(nn.Sequential(nn.Linear(n_hidden_layer_dims[-1], 1)
                                                                ,activ_funcs[-1][0]
                                                                ))
                        self.params_for_print.append(self.hidden_layers[-1])
                    # No activation function
                    else:
                        self.hidden_layers.append(nn.Sequential(nn.Linear(n_hidden_layer_dims[-1], 1)

                                                                ))
                        self.params_for_print.append(self.hidden_layers[-1])
            # Other Layers
            else:
                # Batch normalization
                if batchnorm_bool == True and batchnorm_layers[0][c] == 'yes':
                    self.hidden_layers.append(nn.Sequential(nn.Linear(n_hidden_layer_dims[c-1], n_hidden_layer_dims[c]),
                                                            nn.BatchNorm1d(n_hidden_layer_dims[c]),
                                                            activ_funcs[0][c]))
                    self.params_for_print.append(self.hidden_layers[-1])

                # No batch normalization
                else:
                    self.hidden_layers.append(nn.Sequential(nn.Linear(n_hidden_layer_dims[c-1], n_hidden_layer_dims[c]),
                                                            activ_funcs[0][c]))
                    self.params_for_print.append(self.hidden_layers[-1])



        if print_bool == True:
            print("Model: ", self.params_for_print)


        self.batches = {}
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, data):
        """

        :param data: Data Input ; dtype : Tuple/List of Tensor(n_samples_in_batch,n_proteins * n_features)
        :return: "Risk ratio" ; dtype : Tensor(n_samples_in_batch,1)
        """

        batch_size = data.shape[0]
        x = data[:, :self.num_nodes * self.in_features]
        input_size = self.num_nodes

        # One Graph Convolutional Layer
        if len(self.graphconvs) == 1:
            x = x.reshape(batch_size, self.num_nodes, self.in_features)
            if batch_size not in self.batches:
                l = []
                # For each sample in the batch assign the edge indices to it
                for i in range(batch_size):
                    l.append(Data(x=x[i], edge_index=self.edge_index))
                # batch of form : DataBatch(x = [n_samples_in_batch * n_nodes , n_features],
                # edge_index=[n_features, n_edge_indices * n_samples],
                # batch = [n_samples_in_batch * n_nodes], ptr=[n_samples_in_batch + 1]
                batch = Batch.from_data_list(l)
                self.batches[batch_size] = batch

            batch = self.batches[batch_size]
            # reshape to n_samples_in_batch * n_nodes, n_features
            x = x.reshape(-1, self.in_features)

            if self.activ_funcs_graphconv[0].lower() == 'relu':
                x = F.relu(self.conv1(x=x, edge_index=batch.edge_index))
            elif self.activ_funcs_graphconv[0].lower() == 'sigmoid':
                x = torch.sigmoid(self.conv1(x=x, edge_index=batch.edge_index))
                # PreLU doesn't work here for some reason
            #    elif self.activ_funcs_graphconv[0].lower() == 'prelu':
            #        x = nn.PReLU(self.conv1(x=x, edge_index=batch.edge_index))

            x, edge_index, _, batch, perm, score = self.pool1(
                x, batch.edge_index, None, batch.batch)

            x = x.view(batch_size, -1)

            for layer_c, layer in enumerate(self.hidden_layers):
                # Last Layer
                if layer_c == len(self.hidden_layers) - 1:
                    if self.dropout_bool == True and self.dropout_layers[-1][0] == 'yes':
                        x = self.dropout(x)
                # Other Layers
                else:
                    if self.dropout_bool == True and self.dropout_layers[0][layer_c] == 'yes':
                        x = self.dropout(x)
                x = layer(x)



        # 2 Graph Convolutional Layers
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x = x.reshape(-1, self.in_features)
            batches = []
            for i in range(batch_size):
                tr = torch.ones(input_size, dtype=torch.int64) * i
                batches.append(tr)
                batch = torch.cat(batches, 0)

            if self.activ_funcs_graphconv[0].lower() == 'relu':
                x = F.relu(self.conv1(x=x, edge_index=self.edge_index))
            elif self.activ_funcs_graphconv[0].lower() == 'sigmoid':
                x = torch.sigmoid(self.conv1(x=x, edge_index=self.edge_index))
            x = x.to(device)
            batch = batch.to(device)
            self.edge_index = self.edge_index.to(device)
            x, edge_index, _, batch, perm, score = self.pool1(x, self.edge_index, None, batch)

            if self.activ_funcs_graphconv[1].lower() == 'relu':
                x = F.relu(self.conv2(x=x, edge_index=edge_index))
            elif self.activ_funcs_graphconv[1].lower() == 'sigmoid':
                x = torch.sigmoid(self.conv2(x=x, edge_index=edge_index))

            x = gmp(x, batch)
            x = x.view(batch_size, -1)
            for layer_c, layer in enumerate(self.hidden_layers):
                # Last Layer
                if layer_c == len(self.hidden_layers) - 1:
                    if self.dropout_bool == True and self.dropout_layers[-1][0] == 'yes':
                        x = self.dropout(x)
                # Other Layers
                else:
                    if self.dropout_bool == True and self.dropout_layers[0][layer_c] == 'yes':
                        x = self.dropout(x)
                x = layer(x)



        return x


# https://github.com/bio-ontology-research-group/DeepMOCCA/blob/master/step-by-step/deepmocca_training.ipynb
def normalize(data, minx=None, maxx=None):
    """
    Normalizing Function from :
    https://github.com/bio-ontology-research-group/DeepMOCCA/blob/master/step-by-step/deepmocca_training.ipynb
    """
    if minx is None:
        minx = np.min(data)
        maxx = np.max(data)
    if minx == maxx:
        return data
    return (data - minx) / (maxx - minx)

def normalize_by_row(data):
    """
    Normalizing Function from :
    https://github.com/bio-ontology-research-group/DeepMOCCA/blob/master/step-by-step/deepmocca_training.ipynb
    """
    for i in range(data.shape[0]):
        data[i, :] = normalize(data[i, :])
    return data

def normalize_by_column(data):
    """
    Normalizing Function from :
    https://github.com/bio-ontology-research-group/DeepMOCCA/blob/master/step-by-step/deepmocca_training.ipynb
    """
    for i in range(data.shape[1]):
        data[:, i] = normalize(data[:, i])
    return data





def objective(trial,n_fold,cancer,t_preprocess,layer_amount):
    """
    Optuna Optimization for Hyperparameters.
    :param trial: Settings of the current trial of Hyperparameters
    :param n_fold : Number of fold to be optimized ; dtype : Int
    :param cancer : Name of cancer (folder) ; dtype : String
    :param t_preprocess : Type of preprocessing ; dtype : String
    :return: Concordance Index ; dtype : Float
    """

    direc_set = 'Desktop'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load in data
    dir = os.path.expanduser('~/{}/Project/PreparedData/{}/PPI/median/{}/'.format(direc_set,cancer,t_preprocess))

    print("Running for cancer {} with preprocessing type {} on fold {}".format(cancer,t_preprocess,n_fold))
    trainset_0,trainset_1,trainset_2,trainset_3,trainset_4,valset_0,valset_1,valset_2,valset_3,valset_4,testset_0,testset_1,testset_2,testset_3,testset_4,trainset_feat_0, \
    trainset_feat_1,trainset_feat_2,trainset_feat_3,trainset_feat_4, num_nodes, num_features, edge_index = load_data(data_dir = dir)


    # Load in data
    num_nodes = int(num_nodes)
    num_features = int(num_features)

    # Feature offsets need to be the same in train/val/test for each fold, otherwise NN wouldn't work (diff dimension inputs)
    feat_offs = [trainset_feat_0,trainset_feat_1,trainset_feat_2,trainset_feat_3,trainset_feat_4]


    for c2,_ in enumerate(feat_offs):
        feat_offs[c2] = list(feat_offs[c2].values)
        for idx,_ in enumerate(feat_offs[c2]):
            feat_offs[c2][idx] = feat_offs[c2][idx].item()

    trainset = [trainset_0 ,trainset_1,trainset_2,trainset_3,trainset_4]
    valset = [valset_0 ,valset_1,valset_2,valset_3,valset_4]
    testset = [testset_0,testset_1,testset_2,testset_3,testset_4]
    train_data_folds = []
    train_duration_folds = []
    train_event_folds = []
    val_data_folds = []
    val_duration_folds = []
    val_event_folds = []
    test_data_folds = []


    # LOAD IN DATA
    for c2,_ in enumerate(trainset):
        for c,feat in enumerate(feat_offs[c2]):
            if c < len(feat_offs[c2]) - 3: # train data
                data_np = (np.array((trainset[c2].iloc[:, feat_offs[c2][c] : feat_offs[c2][c+1]]).values).astype('float32'))
                data_tensor = torch.from_numpy(data_np).to(torch.float32)
                data_tensor = data_tensor.to(device)
                train_data = data_tensor
            elif c == len(feat_offs[c2]) - 3: # duration
                duration_np = (np.array((trainset[c2].iloc[:, feat_offs[c2][c] : feat_offs[c2][c+1]]).values).astype('float32')).squeeze(axis=1)
                duration_tensor = torch.from_numpy(duration_np).to(torch.float32)
                duration_tensor = duration_tensor.to(device)
                train_duration = duration_tensor
            elif c == len(feat_offs[c2]) -2: # event
                event_np = (np.array((trainset[c2].iloc[:, feat_offs[c2][c] : feat_offs[c2][c+1]]).values).astype('float32')).squeeze(axis=1)
                event_tensor = torch.from_numpy(event_np).to(torch.float32)
                event_tensor = event_tensor.to(device)
                train_event = event_tensor




        for c,feat in enumerate(feat_offs[c2]):
            if c < len(feat_offs[c2]) - 3: # train data views
                data_np = np.array((valset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')
                data_tensor = torch.from_numpy(data_np).to(torch.float32)
                data_tensor = data_tensor.to(device)
                val_data = data_tensor
            elif c == len(feat_offs[c2]) - 3: # duration
                duration_np = (np.array((valset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')).squeeze(axis=1)
                duration_tensor = torch.from_numpy(duration_np).to(torch.float32)
                duration_tensor = duration_tensor.to(device)
                val_duration = duration_tensor
            elif c == len(feat_offs[c2]) -2: # event
                event_np = (np.array((valset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')).squeeze(axis=1)
                event_tensor = torch.from_numpy(event_np).to(torch.float32)
                event_tensor = event_tensor.to(device)
                val_event = event_tensor



        for c,feat in enumerate(feat_offs[c2]):
            if c < len(feat_offs[c2]) - 3: # train data views
                data_np = np.array((testset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')
                data_tensor = torch.from_numpy(data_np).to(torch.float32)
                data_tensor = data_tensor.to(device)
                test_data = data_tensor
            elif c == len(feat_offs[c2]) - 3: # duration
                duration_np = (np.array((testset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')).squeeze(axis=1)
                duration_tensor = torch.from_numpy(duration_np).to(torch.float32)
                duration_tensor = duration_tensor.to(device)
                test_duration = duration_tensor
            elif c == len(feat_offs[c2]) -2: # event
                event_np = (np.array((testset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')).squeeze(axis=1)
                event_tensor = torch.from_numpy(event_np).to(torch.float32)
                event_tensor = event_tensor.to(device)
                test_event = event_tensor

        train_data_folds.append(train_data)
        val_data_folds.append(val_data)
        train_duration_folds.append(train_duration)
        val_duration_folds.append(val_duration)
        train_event_folds.append(train_event)
        val_event_folds.append(val_event)
        test_data_folds.append(test_data)

    # Rename so we have same structure as in train function
    train_data = train_data_folds
    train_duration = train_duration_folds
    train_event = train_event_folds
    val_data = val_data_folds
    val_duration = val_duration_folds
    val_event = val_event_folds
    test_data = test_data_folds


    # Current fold to be optimized
    c_fold = n_fold

    #reshape so we have the same structure as in train function
    if num_features == 1:
        train_data[c_fold] = torch.unsqueeze(train_data[c_fold],dim=2)
        val_data[c_fold] = torch.unsqueeze(val_data[c_fold],dim = 2)
        test_data[c_fold] = torch.unsqueeze(test_data[c_fold],dim = 2)
    else:
        train_data[c_fold] = train_data[c_fold].reshape(-1, num_nodes, num_features)
        val_data[c_fold] = val_data[c_fold].reshape(-1, num_nodes, num_features)
        test_data[c_fold] = test_data[c_fold].reshape(-1, num_nodes, num_features)


    dimensions_train = train_data[c_fold].shape[1]
    dimensions_val = val_data[c_fold].shape[1]
    dimensions_test = test_data[c_fold].shape[1]

    assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/test'

    dimensions = dimensions_train

    #Needed for GCN
    edge_index = torch.LongTensor(edge_index).to(device)


    #   processing_type = trial.suggest_categorical('processing_type', ['normalize','normalizebyrow','normalizebycolumn','none'])

    # Best results : normalize/standardize in preprocessing, 'none' in postprocessing
    processing_type = 'none'


    if processing_type.lower() == 'normalize':
        for i in range(num_features):
            train_data[c_fold][:,:,i] = normalize(train_data[c_fold][:,:,i])
            val_data[c_fold][:,:,i] = normalize(val_data[c_fold][:,:,i])
            test_data[c_fold][:,:,i] = normalize(test_data[c_fold][:,:,i])
    elif processing_type.lower() == 'normalizebyrow':
        for i in range(num_features):
            train_data[c_fold][:,:,i] = normalize_by_row(train_data[c_fold][:,:,i])
            val_data[c_fold][:,:,i] = normalize_by_row(val_data[c_fold][:,:,i])
            test_data[c_fold][:,:,i] = normalize_by_row(test_data[c_fold][:,:,i])
    elif processing_type.lower() == 'normalizebycolumn':
        for i in range(num_features):
            train_data[c_fold][:,:,i] = normalize_by_column(train_data[c_fold][:,:,i])
            val_data[c_fold][:,:,i] = normalize_by_column(val_data[c_fold][:,:,i])
            test_data[c_fold][:,:,i] = normalize_by_column(test_data[c_fold][:,:,i])






    train_samples = len(train_duration[c_fold])
    val_samples = len(val_duration[c_fold])
    test_samples = len(test_duration)


    # reshape structure for use of GCN # replace -1 with sample sizes
    train_data[c_fold] = train_data[c_fold].reshape(train_samples, num_nodes * num_features)
    val_data[c_fold] = val_data[c_fold].reshape(val_samples, num_nodes * num_features)
    test_data[c_fold] = test_data[c_fold].reshape(test_samples, num_nodes * num_features)

    # Transforms for PyCox
    train_surv = (train_duration[c_fold], train_event[c_fold])
    val_data_full = (val_data[c_fold], (val_duration[c_fold], val_event[c_fold]))



    l2_regularization_bool = trial.suggest_categorical('l2_regularization_bool', [True,False])
    learning_rate = trial.suggest_float("learning_rate", 1e-5,1e-1,log=True)
    l2_regularization_rate = trial.suggest_float("l2_regularization_rate", 1e-6,1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [5,17,32,64,128])
    n_epochs = 100
    dropout_prob = trial.suggest_float("dropout_prob", 0,0.5,step=0.1)
    dropout_bool = trial.suggest_categorical('dropout_bool', [True,False])
    batchnorm_bool = trial.suggest_categorical('batchnorm_bool',[True,False])
    prelu_rate = trial.suggest_float('prelu_rate',0,1,step=0.05)
    ratio = trial.suggest_float('ratio', 0.1,0.9,step=0.1)
    # ratio = 0.7
    # ratio = 0.2
    #  layers_1_FCNN = 4
    # layers_2_FCNN =2
    layers_1_FCNN = trial.suggest_categorical('layers_1_FCNN',[4,8])
    layers_2_FCNN = trial.suggest_categorical('layers_2_FCNN',[1,2])


    layers_FCNN = [layers_1_FCNN,layers_2_FCNN]

    layers_1_FCNN_activfunc = trial.suggest_categorical('layers_1_FCNN_activfunc', ['relu','prelu','sigmoid'])
    layers_2_FCNN_activfunc = trial.suggest_categorical('layers_2_FCNN_activfunc', ['relu','prelu','sigmoid'])


    FCNN_activation_functions = [[layers_1_FCNN_activfunc, layers_2_FCNN_activfunc]]


    layers_1_FCNN_dropout = trial.suggest_categorical('layers_1_FCNN_dropout', ['yes','no'])
    layers_2_FCNN_dropout = trial.suggest_categorical('layers_2_FCNN_dropout', ['yes','no'])

    FCNN_dropouts = [[layers_1_FCNN_dropout, layers_2_FCNN_dropout]]


    layers_1_FCNN_batchnorm = trial.suggest_categorical('layers_1_FCNN_batchnorm', ['yes', 'no'])
    layers_2_FCNN_batchnorm = trial.suggest_categorical('layers_2_FCNN_batchnorm', ['yes', 'no'])
    #   layers_3_FCNN_batchnorm = trial.suggest_categorical('layers_3_FCNN_batchnorm', ['yes', 'no'])

    FCNN_batchnorms = [[layers_1_FCNN_batchnorm, layers_2_FCNN_batchnorm]]

    # Last Layer
    layer_final_activfunc = trial.suggest_categorical('layers_final_activfunc', ['relu','sigmoid','prelu','none'])
    layer_final_dropout = trial.suggest_categorical('layer_final_dropout', ['yes','no'])
    layer_final_batchnorm = trial.suggest_categorical('layer_final_batchnorm', ['yes','no'])
    FCNN_activation_functions.append([layer_final_activfunc])
    FCNN_dropouts.append([layer_final_dropout])
    FCNN_batchnorms.append([layer_final_batchnorm])

    # out_1_graphconv = 16
    out_1_graphconv = trial.suggest_categorical('out_1_graphconv', [1,2,4,8])
    #  out_1_graphconv = num_features # constant bc of some float error ; need to set to the same amount as num_features

    graphconv_1_activation_function = trial.suggest_categorical('graphconv_1_activation_function', ['relu','sigmoid'])
    #  graphconv_1_activation_function = 'relu'
    #  out_2_graphconv = 4
    # decide whether second graphconv layer
    out_2_graphconv = trial.suggest_categorical('out_2_graphconv', [1,2,4,8])
    graphconv_2_activation_function = trial.suggest_categorical('graphconv_2_activation_function', ['relu','sigmoid'])

    # if no second graphconv layer, take it out here
    graphconvs = [out_1_graphconv, out_2_graphconv]
    # graphconvs = [out_1_graphconv]
    graphconvs_activation_functions = [graphconv_1_activation_function, graphconv_2_activation_function]
    # graphconvs_activation_functions = [graphconv_1_activation_function]



    callbacks = [tt.callbacks.EarlyStopping(patience=10)]



    net = GCN(num_nodes = num_nodes,
              edge_index = edge_index,
              in_features=num_features,
              dropout_prob= dropout_prob,
              n_hidden_layer_dims=layers_FCNN,
              activ_funcs=FCNN_activation_functions,
              dropout_bool= dropout_bool,
              dropout_layers=FCNN_dropouts,
              batchnorm_bool=batchnorm_bool,
              batchnorm_layers=FCNN_batchnorms,
              activ_funcs_graphconv= graphconvs_activation_functions,
              graphconvs=graphconvs,
              ratio=ratio,
              prelu_init= prelu_rate,
              print_bool= False).to(device)



    if l2_regularization_bool == True:
        optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=l2_regularization_rate)
    else:
        optimizer = Adam(net.parameters(), lr=learning_rate)

    model = CoxPH(net, optimizer)
    model.set_device(torch.device(device))

    train_print = True

    log = model.fit(train_data[c_fold],
                    train_surv,
                    batch_size,
                    n_epochs,
                    callbacks,
                    verbose=train_print,
                    val_data=val_data_full,
                    val_batch_size= batch_size)

    # Plot it
    #    _ = log.plot()

    # Change for EvalSurv-Function
    try:
        test_duration = test_duration.cpu().detach().numpy()
        test_event = test_event.cpu().detach().numpy()
    except AttributeError:
        pass


    for c,fold in enumerate(train_data):
        try:
            train_duration[c_fold] = train_duration[c_fold].cpu().detach().numpy()
            train_event[c_fold] = train_event[c_fold].cpu().detach().numpy()
            val_duration[c_fold] = val_duration[c_fold].cpu().detach().numpy()
            val_event[c_fold] = val_event[c_fold].cpu().detach().numpy()
        except AttributeError: # in this case already numpy arrays
            pass

    train = train_data[c_fold] ,train_surv

    _ = model.compute_baseline_hazards(*train)

    surv = model.predict_surv_df(val_data[c_fold])


    # Plot it
    #   surv.iloc[:, :5].plot()
    #   plt.ylabel('S(t | x)')
    #   _ = plt.xlabel('Time')


    ev = EvalSurv(surv, val_duration[c_fold], val_event[c_fold], censor_surv='km')


    # concordance
    concordance_index = ev.concordance_td()

    if concordance_index < 0.5:
        concordance_index = 1 - concordance_index

    #brier score
    #  time_grid = np.linspace(test_duration.min(), test_duration.max(), 100)
    #  _ = ev.brier_score(time_grid).plot
    # brier_score = ev.integrated_brier_score(time_grid)

    #binomial log-likelihood
    # binomial_score = ev.integrated_nbll(time_grid)

    # SAVING MODEL POSSIBILITY
    # dir = os.path.expanduser(r'~/SUMO/Project/Trial/Models/Fold_{}_Trial_{}'.format(c_fold,trial.number))

    # torch.save(net,dir)


    return concordance_index

def test_model(n_fold,cancer,t_preprocess):
    """Function to test the model on optimized hyperparameter settings.
    :param n_fold : Number of the fold to test ; dtype : Int
    :param t_preprocess : Type of preprocessing ; dtype : String
    :param cancer : Name of the cancer folder ; dtype : String"""


    direc_set = 'Desktop'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load in data
    dir = os.path.expanduser('~/{}/Project/PreparedData/{}/PPI/median/{}/'.format(direc_set,cancer,t_preprocess))

    print("Running for cancer {} with preprocessing type {} on fold {}".format(cancer,t_preprocess,n_fold))
    trainset_0,trainset_1,trainset_2,trainset_3,trainset_4,valset_0,valset_1,valset_2,valset_3,valset_4,testset_0,testset_1,testset_2,testset_3,testset_4,trainset_feat_0, \
    trainset_feat_1,trainset_feat_2,trainset_feat_3,trainset_feat_4, num_nodes, num_features, edge_index = load_data(data_dir = dir)


    # Load in data
    num_nodes = int(num_nodes)
    num_features = int(num_features)

    # Feature offsets need to be the same in train/val/test for each fold, otherwise NN wouldn't work (diff dimension inputs)
    feat_offs = [trainset_feat_0,trainset_feat_1,trainset_feat_2,trainset_feat_3,trainset_feat_4]


    for c2,_ in enumerate(feat_offs):
        feat_offs[c2] = list(feat_offs[c2].values)
        for idx,_ in enumerate(feat_offs[c2]):
            feat_offs[c2][idx] = feat_offs[c2][idx].item()

    trainset = [trainset_0 ,trainset_1,trainset_2,trainset_3,trainset_4]
    valset = [valset_0 ,valset_1,valset_2,valset_3,valset_4]
    testset = [testset_0,testset_1,testset_2,testset_3,testset_4]
    train_data_folds = []
    train_duration_folds = []
    train_event_folds = []
    val_data_folds = []
    val_duration_folds = []
    val_event_folds = []
    test_data_folds = []


    # LOAD IN DATA
    for c2,_ in enumerate(trainset):
        for c,feat in enumerate(feat_offs[c2]):
            if c < len(feat_offs[c2]) - 3: # train data
                data_np = (np.array((trainset[c2].iloc[:, feat_offs[c2][c] : feat_offs[c2][c+1]]).values).astype('float32'))
                data_tensor = torch.from_numpy(data_np).to(torch.float32)
                data_tensor = data_tensor.to(device)
                train_data = data_tensor
            elif c == len(feat_offs[c2]) - 3: # duration
                duration_np = (np.array((trainset[c2].iloc[:, feat_offs[c2][c] : feat_offs[c2][c+1]]).values).astype('float32')).squeeze(axis=1)
                duration_tensor = torch.from_numpy(duration_np).to(torch.float32)
                duration_tensor = duration_tensor.to(device)
                train_duration = duration_tensor
            elif c == len(feat_offs[c2]) -2: # event
                event_np = (np.array((trainset[c2].iloc[:, feat_offs[c2][c] : feat_offs[c2][c+1]]).values).astype('float32')).squeeze(axis=1)
                event_tensor = torch.from_numpy(event_np).to(torch.float32)
                event_tensor = event_tensor.to(device)
                train_event = event_tensor




        for c,feat in enumerate(feat_offs[c2]):
            if c < len(feat_offs[c2]) - 3: # train data views
                data_np = np.array((valset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')
                data_tensor = torch.from_numpy(data_np).to(torch.float32)
                data_tensor = data_tensor.to(device)
                val_data = data_tensor
            elif c == len(feat_offs[c2]) - 3: # duration
                duration_np = (np.array((valset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')).squeeze(axis=1)
                duration_tensor = torch.from_numpy(duration_np).to(torch.float32)
                duration_tensor = duration_tensor.to(device)
                val_duration = duration_tensor
            elif c == len(feat_offs[c2]) -2: # event
                event_np = (np.array((valset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')).squeeze(axis=1)
                event_tensor = torch.from_numpy(event_np).to(torch.float32)
                event_tensor = event_tensor.to(device)
                val_event = event_tensor



        for c,feat in enumerate(feat_offs[c2]):
            if c < len(feat_offs[c2]) - 3: # train data views
                data_np = np.array((testset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')
                data_tensor = torch.from_numpy(data_np).to(torch.float32)
                data_tensor = data_tensor.to(device)
                test_data = data_tensor
            elif c == len(feat_offs[c2]) - 3: # duration
                duration_np = (np.array((testset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')).squeeze(axis=1)
                duration_tensor = torch.from_numpy(duration_np).to(torch.float32)
                duration_tensor = duration_tensor.to(device)
                test_duration = duration_tensor
            elif c == len(feat_offs[c2]) -2: # event
                event_np = (np.array((testset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')).squeeze(axis=1)
                event_tensor = torch.from_numpy(event_np).to(torch.float32)
                event_tensor = event_tensor.to(device)
                test_event = event_tensor

        train_data_folds.append(train_data)
        val_data_folds.append(val_data)
        train_duration_folds.append(train_duration)
        val_duration_folds.append(val_duration)
        train_event_folds.append(train_event)
        val_event_folds.append(val_event)
        test_data_folds.append(test_data)

    # Rename so we have same structure as in train function
    train_data = train_data_folds
    train_duration = train_duration_folds
    train_event = train_event_folds
    val_data = val_data_folds
    val_duration = val_duration_folds
    val_event = val_event_folds
    test_data = test_data_folds


    # Current fold to be optimized
    c_fold = n_fold

    #reshape so we have the same structure as in train function
    if num_features == 1:
        train_data[c_fold] = torch.unsqueeze(train_data[c_fold],dim=2)
        val_data[c_fold] = torch.unsqueeze(val_data[c_fold],dim = 2)
        test_data[c_fold] = torch.unsqueeze(test_data[c_fold],dim = 2)
    else:
        train_data[c_fold] = train_data[c_fold].reshape(-1, num_nodes, num_features)
        val_data[c_fold] = val_data[c_fold].reshape(-1, num_nodes, num_features)
        test_data[c_fold] = test_data[c_fold].reshape(-1, num_nodes, num_features)


    dimensions_train = train_data[c_fold].shape[1]
    dimensions_val = val_data[c_fold].shape[1]
    dimensions_test = test_data[c_fold].shape[1]

    assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/test'

    dimensions = dimensions_train

    #Needed for GCN
    edge_index = torch.LongTensor(edge_index).to(device)


    #   processing_type = trial.suggest_categorical('processing_type', ['normalize','normalizebyrow','normalizebycolumn','none'])

    # Best results : normalize/standardize in preprocessing, 'none' in postprocessing
    processing_type = 'none'


    if processing_type.lower() == 'normalize':
        for i in range(num_features):
            train_data[c_fold][:,:,i] = normalize(train_data[c_fold][:,:,i])
            val_data[c_fold][:,:,i] = normalize(val_data[c_fold][:,:,i])
            test_data[c_fold][:,:,i] = normalize(test_data[c_fold][:,:,i])
    elif processing_type.lower() == 'normalizebyrow':
        for i in range(num_features):
            train_data[c_fold][:,:,i] = normalize_by_row(train_data[c_fold][:,:,i])
            val_data[c_fold][:,:,i] = normalize_by_row(val_data[c_fold][:,:,i])
            test_data[c_fold][:,:,i] = normalize_by_row(test_data[c_fold][:,:,i])
    elif processing_type.lower() == 'normalizebycolumn':
        for i in range(num_features):
            train_data[c_fold][:,:,i] = normalize_by_column(train_data[c_fold][:,:,i])
            val_data[c_fold][:,:,i] = normalize_by_column(val_data[c_fold][:,:,i])
            test_data[c_fold][:,:,i] = normalize_by_column(test_data[c_fold][:,:,i])






    train_samples = len(train_duration[c_fold])
    val_samples = len(val_duration[c_fold])
    test_samples = len(test_duration)


    # reshape structure for use of GCN # replace -1 with sample sizes
    train_data[c_fold] = train_data[c_fold].reshape(train_samples, num_nodes * num_features)
    val_data[c_fold] = val_data[c_fold].reshape(val_samples, num_nodes * num_features)
    test_data[c_fold] = test_data[c_fold].reshape(test_samples, num_nodes * num_features)

    # Transforms for PyCox
    train_surv = (train_duration[c_fold], train_event[c_fold])
    val_data_full = (val_data[c_fold], (val_duration[c_fold], val_event[c_fold]))


    callbacks = [tt.callbacks.EarlyStopping(patience=10)]

    params={'l2_regularization_bool': True, 'learning_rate': 0.003463396777939911,
            'l2_regularization_rate': 8.522954373859003e-06, 'batch_size': 128, 'dropout_prob': 0.5,
            'dropout_bool': False, 'batchnorm_bool': True, 'prelu_rate': 0.5, 'ratio': 0.5, 'layers_1_FCNN': 4,
            'layers_2_FCNN': 2, 'layers_1_FCNN_activfunc': 'prelu', 'layers_2_FCNN_activfunc': 'relu',
            'layers_1_FCNN_dropout': 'no', 'layers_2_FCNN_dropout': 'yes', 'layers_1_FCNN_batchnorm': 'yes',
            'layers_2_FCNN_batchnorm': 'no', 'layers_final_activfunc': 'none', 'layer_final_dropout': 'no',
            'layer_final_batchnorm': 'no', 'out_1_graphconv': 2, 'graphconv_1_activation_function': 'relu',
            'out_2_graphconv': 1, 'graphconv_2_activation_function': 'sigmoid'}



    net = GCN(num_nodes = num_nodes,
              edge_index = edge_index,
              in_features=num_features,
              dropout_prob= params['dropout_prob'],
              n_hidden_layer_dims=[params['layers_1_FCNN'],params['layers_2_FCNN']],
              activ_funcs=[[params['layers_1_FCNN_activfunc'],params['layers_2_FCNN_activfunc']]],
              dropout_bool= params['dropout_bool'],
              dropout_layers=[[params['layers_1_FCNN_dropout'],params['layers_2_FCNN_dropout']]],
              batchnorm_bool=params['batchnorm_bool'],
              batchnorm_layers=[[params['layers_1_FCNN_batchnorm'],params['layers_2_FCNN_batchnorm']]],
              activ_funcs_graphconv= [params['graphconv_1_activation_function'],params['graphconv_2_activation_function']],
              graphconvs=[params['out_1_graphconv'],params['out_2_graphconv']],
              ratio=params['ratio'],
              prelu_init= params['prelu_rate'],
              print_bool= False).to(device)



    if params['l2_regularization_bool'] == True:
        optimizer = Adam(net.parameters(), lr=params['learning_rate'], weight_decay=params['l2_regularization_rate'])
    else:
        optimizer = Adam(net.parameters(), lr=params['learning_rate'])

    model = CoxPH(net, optimizer)
    model.set_device(torch.device(device))

    train_print = True

    log = model.fit(train_data[c_fold],
                    train_surv,
                    params['batch_size'],
                    100,
                    callbacks,
                    verbose=train_print,
                    val_data=val_data_full,
                    val_batch_size= params['batch_size'])

    # Plot it
    #    _ = log.plot()

    # Change for EvalSurv-Function
    try:
        test_duration = test_duration.cpu().detach().numpy()
        test_event = test_event.cpu().detach().numpy()
    except AttributeError:
        pass


    for c,fold in enumerate(train_data):
        try:
            train_duration[c_fold] = train_duration[c_fold].cpu().detach().numpy()
            train_event[c_fold] = train_event[c_fold].cpu().detach().numpy()
            val_duration[c_fold] = val_duration[c_fold].cpu().detach().numpy()
            val_event[c_fold] = val_event[c_fold].cpu().detach().numpy()
        except AttributeError: # in this case already numpy arrays
            pass

    train = train_data[c_fold] ,train_surv

    _ = model.compute_baseline_hazards(*train)

    surv = model.predict_surv_df(val_data[c_fold])


    # Plot it
    #   surv.iloc[:, :5].plot()
    #   plt.ylabel('S(t | x)')
    #   _ = plt.xlabel('Time')


    ev = EvalSurv(surv, val_duration[c_fold], val_event[c_fold], censor_surv='km')


    # concordance
    concordance_index = ev.concordance_td()

    if concordance_index < 0.5:
        concordance_index = 1 - concordance_index

    #brier score
    #  time_grid = np.linspace(test_duration.min(), test_duration.max(), 100)
    #  _ = ev.brier_score(time_grid).plot
    # brier_score = ev.integrated_brier_score(time_grid)

    #binomial log-likelihood
    # binomial_score = ev.integrated_nbll(time_grid)


    print(concordance_index)


def optuna_optimization(n_fold,cancer,t_preprocess,layer_amount):
    """
    Optuna Optimization for Hyperparameters.
    """


    # Set amount of different trials
    EPOCHS = 30
    func = lambda trial: objective(trial, n_fold,cancer,t_preprocess,layer_amount)

    study = optuna.create_study(directions=['maximize'],sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.MedianPruner())
    study.optimize(func, n_trials = EPOCHS)
    trial = study.best_trials
    direc_set = 'Desktop'
    dir = os.path.expanduser(r'~/{}/Project/Trial/GCN/{}/{}/GCN_BEST_{}.txt'.format(direc_set,layer_amount,t_preprocess,n_fold))
    with open(dir, 'w') as fp:
        for item in trial:
            # write each item on a new line
            fp.write("%s\n" % item)
    # Show change of c-Index across folds
    fig = optuna.visualization.plot_optimization_history(study)
    dir = os.path.expanduser(r'~/{}/Project/Trial/GCN/{}/{}/GCN_{}_C-INDICES.png'.format(direc_set,layer_amount,t_preprocess,n_fold))
    fig.write_image(dir)
    # Show hyperparameter importance
    fig = optuna.visualization.plot_param_importances(study)
    dir = os.path.expanduser(r'~/{}/Project/Trial/GCN/{}/{}/GCN_{}_HPARAMIMPORTANCE.png'.format(direc_set,layer_amount,t_preprocess,n_fold))
    fig.write_image(dir)




def train(train_data,val_data,test_data,
          train_duration,val_duration,test_duration,
          train_event,val_event,test_event,
          n_epochs,
          batch_size,
          l2_regularization,
          l2_regularization_rate,
          learning_rate,
          prelu_rate,
          layers,
          activation_layers,
          dropout,
          dropout_rate,
          dropout_layers,
          batchnorm,
          batchnorm_layers,
          processing_type,
          edge_index,
          proteins_used,
          activation_layers_graphconv,
          layers_graphconv,
          ratio):
    """

    :param train_data: Training data for each fold ; dtype : List of Tensors(n_samples,n_proteins,n_feature_values)
    :param val_data: Validation data for each fold ; dtype : List of Tensors(n_samples,n_proteins,n_feature_values)
    :param test_data: Test data for each fold ; dtype : List of Tensors(n_samples,n_proteins,n_feature_values)
    :param train_duration: Training Duration for each fold  ; dtype : List of Tensors(n_samples,)
    :param val_duration: Validation Duration for each fold  ; dtype : List of Tensors(n_samples,)
    :param test_duration: Test Duration for each fold  ; dtype : List of Tensors(n_samples,)
    :param train_event: Training Event for each fold  ; dtype : List of Tensors(n_samples,)
    :param val_event: Validation Event for each fold  ; dtype : List of Tensors(n_samples,)
    :param test_event: Test Event for each fold  ; dtype : List of Tensors(n_samples,)
    :param n_epochs: Number of Epochs ; dtype : Int
    :param batch_size: Batch Size ; dtype : Int
    :param l2_regularization: Decide whether to apply L2 regularization ; dtype : Boolean
    :param l2_regularization_rate: L2 regularization rate ; dtype : Float
    :param learning_rate: Learning rate ; dtype : Float
    :param prelu_rate: Initial Value for PreLU activation ; dtype : Float [between 0 and 1]
    :param layers: Dimension of Layers ; dtype : List of Ints
    :param activation_layers: Activation Functions aswell as for the last layer, the last layer can have no activation
    function ['none‘]         ; dtype : List of Lists of Strings ['relu', 'sigmoid', 'prelu']
    :param dropout: Decide whether Dropout is to be applied or not ; dtype : Boolean
    :param dropout_rate: Probability of Neuron Dropouts ; dtype : Int
    :param dropout_layers:  Layers in which to apply Dropout ; dtype : List of Lists of Strings ['yes','no']
    :param batchnorm: Decide whether Batch Normalization is to be applied or not ; dtype : Boolean
    :param batchnorm_layers: Layers in which to apply Batch Normalization ; dtype : List of Lists of Strings ['yes','no']
    :param processing_type: Type of processing on feature values mapped to proteins
                            ; dtype : String ['normalize', 'normalizebyrow', 'normalizebycolumn']
    :param edge_index: Protein Pairs with Interactions ; dtype : List of Lists of Ints [Indices]
    :param proteins_used: Protein to Indices mapping ; dtype : Dictionary(Proteins,Indices)
    :param activation_layers_graphconv: Activation Functions for the Graph Convolutional Layers ; dtype : List of Strings ['relu', 'sigmoid']
    :param layers_graphconv: Dimensions of Layers for the Graph Convolutional Layers ; dtype : List of Ints
    """




    ############################# FOLD X ###################################
    for c_fold,fold in enumerate(train_data):


        # For GPU acceleration, we need to have everything as tensors for the training loop, but pycox EvalSurv
        # Needs duration & event to be numpy arrays, thus at the start we set duration/event to tensors
        # and before EvalSurv to numpy
        try:
            test_duration = torch.from_numpy(test_duration).to(torch.float32)
            test_event = torch.from_numpy(test_event).to(torch.float32)
        except TypeError:
            pass



        try:
            train_duration[c_fold] = torch.from_numpy(train_duration[c_fold]).to(torch.float32)
            train_event[c_fold] = torch.from_numpy(train_event[c_fold]).to(torch.float32)
            val_duration[c_fold] = torch.from_numpy(val_duration[c_fold]).to(torch.float32)
            val_event[c_fold] = torch.from_numpy(val_event[c_fold]).to(torch.float32)
        except TypeError:
            pass

        print("Split {} : ".format(c_fold + 1))
        print("Train data has shape : {} ".format(train_data[c_fold].shape))
        print("Validation data has shape : {} ".format(val_data[c_fold].shape))
        print("Test data has shape : {} ".format(test_data[c_fold].shape))


        dimensions_train = [x.size(1) for x in train_data]
        dimensions_val = [x.size(1) for x in val_data]
        dimensions_test = [x.size(1) for x in test_data]

        assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/test'

        dimensions = dimensions_train


        views_with_proteins = train_data[c_fold].size(2)

        # Needed for GCN
        num_features = views_with_proteins
        num_nodes = len(proteins_used)
        edge_index = torch.LongTensor(edge_index)



        if processing_type.lower() == 'normalize':
            for i in range(num_features):
                train_data[c_fold][:,:,i] = normalize(train_data[c_fold][:,:,i])
                val_data[c_fold][:,:,i] = normalize(val_data[c_fold][:,:,i])
                test_data[c_fold][:,:,i] = normalize(test_data[c_fold][:,:,i])
        elif processing_type.lower() == 'normalizebyrow':
            for i in range(num_features):
                train_data[c_fold][:,:,i] = normalize_by_row(train_data[c_fold][:,:,i])
                val_data[c_fold][:,:,i] = normalize_by_row(val_data[c_fold][:,:,i])
                test_data[c_fold][:,:,i] = normalize_by_row(test_data[c_fold][:,:,i])
        elif processing_type.lower() == 'normalizebycolumn':
            for i in range(num_features):
                train_data[c_fold][:,:,i] = normalize_by_column(train_data[c_fold][:,:,i])
                val_data[c_fold][:,:,i] = normalize_by_column(val_data[c_fold][:,:,i])
                test_data[c_fold][:,:,i] = normalize_by_column(test_data[c_fold][:,:,i])




        # Transforms for PyCox
        train_surv = (train_duration[c_fold], train_event[c_fold])
        val_data_full = (val_data[c_fold], (val_duration[c_fold], val_event[c_fold]))


        # reshape structure for use of GCN # replace -1 with sample sizes
        train_data[c_fold] = train_data[c_fold].reshape(-1, num_nodes * num_features)
        val_data[c_fold] = val_data[c_fold].reshape(-1, num_nodes * num_features)
        test_data[c_fold] = test_data[c_fold].reshape(-1, num_nodes * num_features)


        callbacks = [tt.callbacks.EarlyStopping(patience=10)]


        torch.manual_seed(0)

        layers_u = copy.deepcopy(layers)
        activation_layers_u = copy.deepcopy(activation_layers_u)
        dropout_layers_u = copy.deepcopy(dropout_layers)
        batchnorm_layers_u = copy.deepcopy(batchnorm_layers)
        activation_layers_graphconv_u = copy.deepcopy(activation_layers_graphconv)
        layers_graphconv_u = copy.deepcopy(layers_graphconv)
        net = GCN(num_nodes = len(proteins_used),
                  edge_index = edge_index,
                  in_features=num_features,
                  n_hidden_layer_dims=layers_u,
                  activ_funcs=activation_layers_u,
                  dropout_bool= dropout,
                  dropout_prob= dropout_rate,
                  dropout_layers=dropout_layers,
                  batchnorm_bool=batchnorm,
                  batchnorm_layers=batchnorm_layers_u,
                  activ_funcs_graphconv= activation_layers_graphconv_u,
                  ratio=ratio,
                  graphconvs=layers_graphconv_u,
                  prelu_init= prelu_rate,
                  print_bool= False)



        if l2_regularization == True:
            optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=l2_regularization_rate)
        else:
            optimizer = Adam(net.parameters(), lr=learning_rate)

        model = CoxPH(net, optimizer)

        print_train = False

        log = model.fit(train_data[c_fold],
                        train_surv,
                        batch_size,
                        n_epochs,
                        callbacks,
                        verbose=print_train,
                        val_data=val_data_full,
                        val_batch_size= batch_size)

        # Plot it
        #    _ = log.plot()

        # Change for EvalSurv-Function
        try:
            test_duration = test_duration.cpu().detach().numpy()
            test_event = test_event.cpu().detach().numpy()
        except AttributeError:
            pass


        for c,fold in enumerate(train_data):
            try:
                train_duration[c_fold] = train_duration[c_fold].cpu().detach().numpy()
                train_event[c_fold] = train_event[c_fold].cpu().detach().numpy()
                val_duration[c_fold] = val_duration[c_fold].cpu().detach().numpy()
                val_event[c_fold] = val_event[c_fold].cpu().detach().numpy()
            except AttributeError: # in this case already numpy arrays
                pass

        train = train_data[c_fold] , train_surv

        _ = model.compute_baseline_hazards(*train)

        surv = model.predict_surv_df(test_data[c_fold])

        # Needed for PyCox (if already in numpy, no need to transform --> try/except for error handling)

        try:
            test_duration = test_duration.numpy()
            test_event = test_event.numpy()
        except AttributeError:
            pass


        # Plot it
        #    surv.iloc[:, :5].plot()
        #    plt.ylabel('S(t | x)')
        #    _ = plt.xlabel('Time')


        ev = EvalSurv(surv, test_duration, test_event, censor_surv='km')


        # concordance
        concordance_index = ev.concordance_td()

        if concordance_index < 0.5:
            concordance_index = 1 - concordance_index

        #brier score
        time_grid = np.linspace(test_duration.min(), test_duration.max(), 100)
        _ = ev.brier_score(time_grid).plot
        brier_score = ev.integrated_brier_score(time_grid)

        #binomial log-likelihood
        binomial_score = ev.integrated_nbll(time_grid)

        print("Concordance index : {} , Integrated Brier Score : {} , Binomial Log-Likelihood : {}".format(concordance_index,
                                                                                                           brier_score,
                                                                                                           binomial_score))






def load_data(data_dir):

    """
    Function to load data. Needed for Optuna Optimization.
    :param data_dir: Directory in which data is stored.
    :return: data and feature offsets (for feature values, duration and event), number of nodes, features and edge_index
    """

    trainset_0 = pd.read_csv(
        os.path.join(data_dir + "TrainData_0.csv"), index_col=0)
    trainset_1 = pd.read_csv(
        os.path.join(data_dir + "TrainData_1.csv"), index_col=0)
    trainset_2 = pd.read_csv(
        os.path.join(data_dir + "TrainData_2.csv"), index_col=0)
    trainset_3 = pd.read_csv(
        os.path.join(data_dir + "TrainData_3.csv"), index_col=0)
    trainset_4 = pd.read_csv(
        os.path.join(data_dir + "TrainData_4.csv"), index_col=0)

    trainset_feat_0 = pd.read_csv(
        os.path.join(data_dir +"TrainDataFeatOffs_0.csv"), index_col=0)

    trainset_feat_1 = pd.read_csv(
        os.path.join(data_dir +"TrainDataFeatOffs_1.csv"), index_col=0)

    trainset_feat_2 = pd.read_csv(
        os.path.join(data_dir +"TrainDataFeatOffs_2.csv"), index_col=0)

    trainset_feat_3 = pd.read_csv(
        os.path.join(data_dir +"TrainDataFeatOffs_3.csv"), index_col=0)

    trainset_feat_4 = pd.read_csv(
        os.path.join(data_dir +"TrainDataFeatOffs_4.csv"), index_col=0)


    valset_0 = pd.read_csv(
        os.path.join(data_dir + "ValData_0.csv"), index_col=0)
    valset_1 = pd.read_csv(
        os.path.join(data_dir + "ValData_1.csv"), index_col=0)
    valset_2 = pd.read_csv(
        os.path.join(data_dir + "ValData_2.csv"), index_col=0)
    valset_3 = pd.read_csv(
        os.path.join(data_dir + "ValData_3.csv"), index_col=0)
    valset_4 = pd.read_csv(
        os.path.join(data_dir + "ValData_4.csv"), index_col=0)


    testset_0 = pd.read_csv(
        os.path.join(data_dir +  "TestData_0.csv"), index_col=0)
    testset_1 = pd.read_csv(
        os.path.join(data_dir +  "TestData_1.csv"), index_col=0)
    testset_2 = pd.read_csv(
        os.path.join(data_dir +  "TestData_2.csv"), index_col=0)
    testset_3 = pd.read_csv(
        os.path.join(data_dir +  "TestData_3.csv"), index_col=0)
    testset_4 = pd.read_csv(
        os.path.join(data_dir +  "TestData_4.csv"), index_col=0)

    num_nodes = np.loadtxt(data_dir + "num_nodes.txt", unpack=False)
    num_features = np.loadtxt(data_dir + "num_features.txt", unpack=False)



    edge_index_1 = np.loadtxt(data_dir +"edge_index_1.txt" , dtype=int, comments="#", delimiter=",", unpack=False)
    edge_index_2 = np.loadtxt(data_dir +"edge_index_2.txt" , dtype=int, comments="#", delimiter=",", unpack=False)

    edge_index = [list(edge_index_1), list(edge_index_2)]


    return trainset_0,trainset_1,trainset_2,trainset_3,trainset_4,valset_0,valset_1,valset_2,valset_3,valset_4, \
           testset_0,testset_1,testset_2,testset_3,testset_4, \
           trainset_feat_0,trainset_feat_1,trainset_feat_2,trainset_feat_3,trainset_feat_4, num_nodes, num_features, edge_index

