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
        super(GCN, self).__init__()
        self.num_nodes = num_nodes # number of proteins
        self.edge_index = edge_index
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


        else: # In this case pooling layer last layer before we apply FCNN
            self.first_in = math.ceil(ratio * (graphconvs[0] * num_nodes))



        self.params_for_print.append(self.pool1)


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
        :return: "Risk ratio" ; dtype : Tensor(n_samples_in_batch,1) TODO : Namen finden, den man auch in der BA dann benutzt
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

            x, edge_index, _, batch, perm, score = self.pool1(x, self.edge_index, None, batch)

            if self.activ_funcs_graphconv[1].lower() == 'relu':
                x = F.relu(self.conv2(x=x, edge_index=edge_index))
            elif self.activ_funcs_graphconv[1].lower() == 'sigmoid':
                x = torch.sigmoid(self.conv2(x=x, edge_index=edge_index))

            x = gmp(x, batch)
            x = x.view(batch_size, -1)
            for layer_c, layer in enumerate(self.hidden_layers):
                x = layer(x)
                if self.dropout_bool == True and self.dropout_layers[layer_c] == 'yes':
                    x = self.dropout(x)



        return x # all return values have the same value --> error in net


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





def objective(trial):
    """
    Optuna Optimization for Hyperparameters.
    :param trial: Settings of the current trial of Hyperparameters
    :return: Concordance Index ; dtype : Float
    """

    trainset, trainset_feat, valset,valset_feat, testset,testset_feat, num_nodes, num_features, edge_index = load_data()


    # Load in data (##### For testing for first fold, later on
    num_nodes = int(num_nodes)
    num_features = int(num_features)

    trainset_feat = list(trainset_feat.values)
    for idx,_ in enumerate(trainset_feat):
        trainset_feat[idx] = trainset_feat[idx].item()

    valset_feat = list(valset_feat.values)
    for idx,_ in enumerate(valset_feat):
        valset_feat[idx] = valset_feat[idx].item()

    testset_feat = list(testset_feat.values)
    for idx,_ in enumerate(testset_feat):
        testset_feat[idx] = testset_feat[idx].item()



    for c,feat in enumerate(trainset_feat):
        if c < len(trainset_feat) - 3: # train data
            train_data = (np.array((trainset.iloc[:, trainset_feat[c] : trainset_feat[c+1]]).values).astype('float32'))
        elif c == len(trainset_feat) - 3: # duration
            train_duration = (np.array((trainset.iloc[:, trainset_feat[c] : trainset_feat[c+1]]).values).astype('float32')).squeeze(axis=1)
        elif c == len(trainset_feat) -2: # event
            train_event = (np.array((trainset.iloc[:, trainset_feat[c] : trainset_feat[c+1]]).values).astype('float32')).squeeze(axis=1)




    for c,feat in enumerate(valset_feat):
        if c < len(valset_feat) - 3: # train data views
            val_data = (np.array((valset.iloc[:, valset_feat[c]: valset_feat[c + 1]]).values).astype('float32'))
        elif c == len(valset_feat) - 3: # duration
            val_duration = (np.array((valset.iloc[:, valset_feat[c]: valset_feat[c + 1]]).values).astype('float32')).squeeze(axis=1)
        elif c == len(valset_feat) -2: # event
            val_event = (np.array((valset.iloc[:, valset_feat[c]: valset_feat[c + 1]]).values).astype('float32')).squeeze(axis=1)



    for c,feat in enumerate(testset_feat):
        if c < len(testset_feat) - 3: # train data views
            test_data = (np.array((testset.iloc[:, testset_feat[c]: testset_feat[c + 1]]).values).astype('float32'))
        elif c == len(testset_feat) - 3: # duration
            test_duration = (np.array((testset.iloc[:, testset_feat[c]: testset_feat[c + 1]]).values).astype('float32')).squeeze(axis=1)
        elif c == len(testset_feat) -2: # event
            test_event = (np.array((testset.iloc[:, testset_feat[c]: testset_feat[c + 1]]).values).astype('float32')).squeeze(axis=1)




    #reshape so we have the same structure as in train function
    if num_features == 1:
        train_data = np.expand_dims(train_data,axis = 2)
        val_data = np.expand_dims(val_data,axis = 2)
        test_data = np.expand_dims(test_data,axis = 2)
    else:
        train_data = train_data.reshape(-1, num_nodes, num_features)
        val_data = val_data.reshape(-1, num_nodes, num_features)
        test_data = test_data.reshape(-1, num_nodes, num_features)


    dimensions_train = train_data.shape[1]
    dimensions_val = val_data.shape[1]
    dimensions_test = test_data.shape[1]

    assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/test'

    dimensions = dimensions_train

    #Needed for GCN
    edge_index = torch.LongTensor(edge_index) #.to(device)


    processing_type = trial.suggest_categorical('processing_type', ['normalize','normalizebyrow','normalizebycolumn','none'])


    if processing_type.lower() == 'normalize':
        for i in range(num_features):
            train_data[:,:,i] = normalize(train_data[:,:,i])
            val_data[:,:,i] = normalize(val_data[:,:,i])
            test_data[:,:,i] = normalize(test_data[:,:,i])
    elif processing_type.lower() == 'normalizebyrow':
        for i in range(num_features):
            train_data[:,:,i] = normalize_by_row(train_data[:,:,i])
            val_data[:,:,i] = normalize_by_row(val_data[:,:,i])
            test_data[:,:,i] = normalize_by_row(test_data[:,:,i])
    elif processing_type.lower() == 'normalizebycolumn':
        for i in range(num_features):
            train_data[:,:,i] = normalize_by_column(train_data[:,:,i])
            val_data[:,:,i] = normalize_by_column(val_data[:,:,i])
            test_data[:,:,i] = normalize_by_column(test_data[:,:,i])



    # Transforms for PyCox
    train_surv = (train_duration, train_event)
    val_data_full = (val_data, (val_duration, val_event))


    train_samples = len(train_duration)
    val_samples = len(val_duration)
    test_samples = len(test_duration)


    # reshape structure for use of GCN # replace -1 with sample sizes
    train_data = train_data.reshape(train_samples, num_nodes * num_features)
    val_data = val_data.reshape(val_samples, num_nodes * num_features)
    test_data = test_data.reshape(test_samples, num_nodes * num_features)



    l2_regularization_bool = trial.suggest_categorical('l2_regularization_bool', [True,False])
    learning_rate = trial.suggest_float("learning_rate", 1e-5,1e-1,log=True)
    l2_regularization_rate = trial.suggest_float("l2_regularization_rate", 1e-6,1e-3, log=True)
  #  batch_size = trial.suggest_int("batch_size", 5, 200) # TODO : batch size so wählen, dass train samples/ batch_size und val samples/batch_size nie 1 ergeben können, da sonst Error : noch besser error abfangen und einfach skippen, da selten passiert !
    batch_size = trial.suggest_categorical("batch_size", [8,16,32,64,128,256])
  #  n_epochs = trial.suggest_int("n_epochs", 10,20) # setting num of epochs to 10-20 instead of 10-100 bc. it takes too much time
    n_epochs = 15
    dropout_prob = trial.suggest_float("dropout_prob", 0,0.5,step=0.1)
    dropout_bool = trial.suggest_categorical('dropout_bool', [True,False])
    batchnorm_bool = trial.suggest_categorical('batchnorm_bool',[True,False])
    prelu_rate = trial.suggest_float('prelu_rate',0,1,step=0.05)
    ratio = trial.suggest_float('ratio', 0,1,step=0.1)




    layers_1_FCNN = trial.suggest_int('layers_1_FCNN', 5, 300)
    layers_2_FCNN = trial.suggest_int('layers_2_FCNN', 5, 300)

    layers_FCNN = [layers_1_FCNN,layers_2_FCNN]


    layers_1_FCNN_activfunc = trial.suggest_categorical('layers_1_FCNN_activfunc', ['relu','prelu','sigmoid'])
  #  layers_1_FCNN_activfunc = 'relu'
    layers_2_FCNN_activfunc = trial.suggest_categorical('layers_2_FCNN_activfunc', ['relu','prelu', 'sigmoid'])
  #  layers_2_FCNN_activfunc = 'relu'

    FCNN_activation_functions = [[layers_1_FCNN_activfunc, layers_2_FCNN_activfunc]]


    layers_1_FCNN_dropout = trial.suggest_categorical('layers_1_FCNN_dropout', ['yes','no'])
    layers_2_FCNN_dropout = trial.suggest_categorical('layers_2_FCNN_dropout', ['yes','no'])
 #   layers_3_FCNN_dropout = trial.suggest_categorical('layers_3_FCNN_dropout', ['yes','no'])

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

 #   out_1_graphconv = trial.suggest_int('out_1_graphconv', 5, 300)
    out_1_graphconv = 2 # constant bc of some float error
    # TODO : prelu doesnt work for graph conv activation for some reason
 #   graphconv_1_activation_function = trial.suggest_categorical('graphconv_1_activation_function', ['relu','prelu'])
    graphconv_1_activation_function = 'relu'

    # decide whether second graphconv layer
  #  out_2_graphconv = trial.suggest_int('out_2_graphconv', 5, 300)
 #   graphconv_2_activation_function = trial.suggest_categorical('graphconv_2_activation_function', ['relu','sigmoid'])

    # if no second graphconv layer, take it out here
 #   graphconvs = [out_1_graphconv, out_2_graphconv] # TODO : with only one rundungsfehler im pooling layer
    graphconvs = [out_1_graphconv]
  #  graphconvs_activation_functions = [graphconv_1_activation_function, graphconv_2_activation_function]
    graphconvs_activation_functions = [graphconv_1_activation_function]



    callbacks = [tt.callbacks.EarlyStopping(patience=10)]


    torch.manual_seed(0)

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
              print_bool= False)



    if l2_regularization_bool == True:
        optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=l2_regularization_rate)
    else:
        optimizer = Adam(net.parameters(), lr=learning_rate)

    model = CoxPH(net, optimizer)

    train_print = True

    log = model.fit(train_data,train_surv, batch_size, n_epochs, callbacks, verbose=train_print,
                    val_data=val_data_full, val_batch_size= batch_size)

    # Plot it
#    _ = log.plot()

    train = train_data , train_surv

    _ = model.compute_baseline_hazards(*train)

    surv = model.predict_surv_df(test_data)

    # Needed for PyCox (if already in numpy, no need to transform --> try/except for error handling)

    try:
        test_duration = test_duration.numpy()
        test_event = test_event.numpy()
    except AttributeError:
        pass


    # Plot it
 #   surv.iloc[:, :5].plot()
 #   plt.ylabel('S(t | x)')
 #   _ = plt.xlabel('Time')


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


    return concordance_index



def optuna_optimization(fold = 1):
    """
    Optuna Optimization for Hyperparameters.
    """

    EPOCHS = 150
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials = EPOCHS)

    trial = study.best_trial

    print("Best Concordance", trial.value)
    print("Best Hyperparamters : {}".format(trial.params))




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


    # As we use PPI feature selection in GCN, we don't have multiple views structure : we don't need numpy transforms

    ############################# FOLD X ###################################
    for c_fold,fold in enumerate(train_data):

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

        net = GCN(num_nodes = len(proteins_used),
                  edge_index = edge_index,
                  in_features=num_features,
                  n_hidden_layer_dims=layers,
                  activ_funcs=activation_layers,
                  dropout_bool= dropout,
                  dropout_prob= dropout_rate,
                  dropout_layers=dropout_layers,
                  batchnorm_bool=batchnorm,
                  batchnorm_layers=batchnorm_layers,
                  activ_funcs_graphconv= activation_layers_graphconv,
                  ratio=ratio,
                  graphconvs=layers_graphconv,
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
        _ = log.plot()

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






def load_data(data_dir="/Users/marlon/Desktop/Project/PreparedData/"):

    """
    Function to load data. Needed for Optuna Optimization.
    :param data_dir: Directory in which data is stored.
    :return: data and feature offsets (for feature values, duration and event), number of nodes, features and edge_index
    """

    trainset = pd.read_csv(
        os.path.join(data_dir + "TrainData.csv"), index_col=0)

    trainset_feat = pd.read_csv(
        os.path.join(data_dir +"TrainDataFeatOffs.csv"), index_col=0)


    valset = pd.read_csv(
        os.path.join(data_dir + "ValData.csv"), index_col=0)


    valset_feat = pd.read_csv(
        os.path.join(data_dir +"ValDataFeatOffs.csv"), index_col=0)


    testset = pd.read_csv(
        os.path.join(data_dir +  "TestData.csv"), index_col=0)

    testset_feat = pd.read_csv(
        os.path.join(data_dir + "TestDataFeatOffs.csv"), index_col=0)

#    num_nodes = pd.read_csv(data_dir + "num_nodes.txt", sep=" ", header=None)
#    num_features = pd.read_csv(data_dir + "num_features.txt", sep=" ", header=None)

    num_nodes = np.loadtxt(data_dir + "num_nodes.txt", unpack=False)
    num_features = np.loadtxt(data_dir + "num_features.txt", unpack=False)



    edge_index_1 = np.loadtxt(data_dir +"edge_index_1.txt" , dtype=int, comments="#", delimiter=",", unpack=False)
    edge_index_2 = np.loadtxt(data_dir +"edge_index_2.txt" , dtype=int, comments="#", delimiter=",", unpack=False)

    edge_index = [list(edge_index_1), list(edge_index_2)]






    return trainset, trainset_feat, valset,valset_feat, testset,testset_feat, num_nodes, num_features, edge_index

