import numpy as np
import torch
import torchtuples as tt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, SAGPooling
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.optim import Adam
import pandas as pd
import os
import optuna
from torch_geometric.nn import global_max_pool as gmp






class GCN(nn.Module):
    """https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv"""
    def __init__(self, num_nodes, edge_index, in_features, n_hidden_layer_dims, activ_funcs,
                 dropout_prob = 0.1, ratio = 0.4, dropout_bool = False, batchnorm_bool = False,
                 dropout_layers = None, batchnorm_layers = None,
                 activ_funcs_graphconv = None, graphconvs = None):
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

        for c,afunc in enumerate(activ_funcs):
            if afunc.lower() == 'relu':
                activ_funcs[c] = nn.ReLU()
            elif afunc.lower() == 'sigmoid':
                activ_funcs[c] = nn.Sigmoid()










        self.conv1 = GraphConv(in_features, graphconvs[0])

        self.pool1 = SAGPooling(graphconvs[0], ratio=ratio, GNN=GraphConv) # in channel same as out of conv1

        if len(graphconvs) == 2: # Two GraphConv layers, only 1 pooling layer
            self.conv2 = GraphConv(graphconvs[0], graphconvs[1])
            self.first_in = graphconvs[1]


        else: # In this case pooling layer last layer before we apply FCNN
            self.first_in = math.ceil(ratio * (graphconvs[0] * num_nodes)) # TODO : führt zu rundungsfehlern



        self.params_for_print.append(self.pool1)


        for c in range(len(n_hidden_layer_dims) +1):
            if c == 0: # first layer
                if batchnorm_bool == True and batchnorm_layers[c] == 'yes':
                    self.hidden_layers.append(nn.Sequential(nn.Linear(self.first_in, n_hidden_layer_dims[0]),
                                                            nn.BatchNorm1d(n_hidden_layer_dims[0]),
                                                            activ_funcs[0]))
                    self.params_for_print.append(self.hidden_layers[-1])
                else:
                    self.hidden_layers.append(nn.Sequential(nn.Linear(self.first_in, n_hidden_layer_dims[0]),
                                                            activ_funcs[0]))
                    self.params_for_print.append(self.hidden_layers[-1])

            elif c == len(n_hidden_layer_dims): # last layer (no activation function)
                if batchnorm_bool == True and batchnorm_layers[c] == 'yes':
                    self.hidden_layers.append(nn.Sequential(nn.Linear(n_hidden_layer_dims[-1], 1),
                                                            nn.BatchNorm1d(1),
                                                            ))
                    self.params_for_print.append(self.hidden_layers[-1])
                else:
                    self.hidden_layers.append(nn.Sequential(nn.Linear(n_hidden_layer_dims[-1], 1),
                                                            nn.BatchNorm1d(1),
                                                            ))
                    self.params_for_print.append(self.hidden_layers[-1])
            else: # other layers
                if batchnorm_bool == True and batchnorm_layers == 'yes':
                     self.hidden_layers.append(nn.Sequential(nn.Linear(n_hidden_layer_dims[c-1], n_hidden_layer_dims[c]),
                                                            nn.BatchNorm1d(n_hidden_layer_dims[c]),
                                                            activ_funcs[c]))
                     self.params_for_print.append(self.hidden_layers[-1])

                else:
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
        input_size = self.num_nodes

        if len(self.graphconvs) == 1:
            x = x.reshape(batch_size, self.num_nodes, self.in_features)
            if batch_size not in self.batches:
                l = []
                for i in range(batch_size):
                    l.append(Data(x=x[i], edge_index=self.edge_index))
                batch = Batch.from_data_list(l)
                self.batches[batch_size] = batch

            batch = self.batches[batch_size]
            x = x.reshape(-1, self.in_features)

            if self.activ_funcs_graphconv[0].lower() == 'relu':
                x = F.relu(self.conv1(x=x, edge_index=batch.edge_index))
            elif self.activ_funcs_graphconv[0].lower() == 'sigmoid':
                x = F.sigmoid(self.conv1(x=x, edge_index=batch.edge_index))
            x, edge_index, _, batch, perm, score = self.pool1(
                x, batch.edge_index, None, batch.batch)

            x = x.view(batch_size, -1)

            for layer_c, layer in enumerate(self.hidden_layers):
                x = layer(x)
                if self.dropout_bool == True and self.dropout_layers[layer_c] == 'yes':
                    x = self.dropout(x)



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
    if minx is None:
        minx = np.min(data)
        maxx = np.max(data)
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





def objective(trial):
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


    for i in range(num_features):
        train_data[:,:,i] = normalize_by_row(train_data[:,:,i])
        val_data[:,:,i] = normalize_by_row(val_data[:,:,i])
        test_data[:,:,i] = normalize_by_row(test_data[:,:,i])

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
    batch_size = trial.suggest_int("batch_size", 5, 200) # TODO : batch size so wählen, dass train samples/ batch_size und val samples/batch_size nie 1 ergeben können, da sonst Error : noch besser error abfangen und einfach skippen, da selten passiert !
    n_epochs = trial.suggest_int("n_epochs", 10,20) # setting num of epochs to 10-20 instead of 10-100 bc. it takes too much time
    dropout_prob = trial.suggest_float("dropout_prob", 0,0.5,step=0.1)
    dropout_bool = trial.suggest_categorical('dropout_bool', [True,False])
    batchnorm_bool = trial.suggest_categorical('batchnorm_bool',[True,False])


    layers_1_FCNN = trial.suggest_int('layers_1_FCNN', 5, 300)
    layers_2_FCNN = trial.suggest_int('layers_2_FCNN', 5, 300)

    layers_FCNN = [layers_1_FCNN,layers_2_FCNN]

    layers_1_FCNN_activfunc = trial.suggest_categorical('layers_1_FCNN_activfunc', ['relu','sigmoid'])
    layers_2_FCNN_activfunc = trial.suggest_categorical('layers_2_FCNN_activfunc', ['relu','sigmoid'])

    FCNN_activation_functions = [layers_1_FCNN_activfunc, layers_2_FCNN_activfunc]


    layers_1_FCNN_dropout = trial.suggest_categorical('layers_1_FCNN_dropout', ['yes','no'])
    layers_2_FCNN_dropout = trial.suggest_categorical('layers_2_FCNN_dropout', ['yes','no'])
    layers_3_FCNN_dropout = trial.suggest_categorical('layers_3_FCNN_dropout', ['yes','no'])

    FCNN_dropouts = [layers_1_FCNN_dropout, layers_2_FCNN_dropout, layers_3_FCNN_dropout]


    layers_1_FCNN_batchnorm = trial.suggest_categorical('layers_1_FCNN_batchnorm', ['yes', 'no'])
    layers_2_FCNN_batchnorm = trial.suggest_categorical('layers_2_FCNN_batchnorm', ['yes', 'no'])
    layers_3_FCNN_batchnorm = trial.suggest_categorical('layers_3_FCNN_batchnorm', ['yes', 'no'])

    FCNN_batchnorms = [layers_1_FCNN_batchnorm, layers_2_FCNN_batchnorm, layers_3_FCNN_batchnorm]

    out_1_graphconv = trial.suggest_int('out_1_graphconv', 5, 300)
    graphconv_1_activation_function = trial.suggest_categorical('graphconv_1_activation_function', ['relu','sigmoid'])

    # decide whether second graphconv layer
    out_2_graphconv = trial.suggest_int('out_2_graphconv', 5, 300)
    graphconv_2_activation_function = trial.suggest_categorical('graphconv_2_activation_function', ['relu','sigmoid'])

    # if no second graphconv layer, take it out here
    graphconvs = [out_1_graphconv, out_2_graphconv] # TODO : with only one rundungsfehler im pooling layer
    graphconvs_activation_functions = [graphconv_1_activation_function, graphconv_2_activation_function]



    callbacks = [tt.callbacks.EarlyStopping(patience=10)]


    torch.manual_seed(0)

    net = GCN(num_nodes = num_nodes,
              edge_index = edge_index,
              in_features=num_features,
              dropout_prob= dropout_prob,
              n_hidden_layer_dims=layers_FCNN,
              activ_funcs=FCNN_activation_functions,
              dropout_bool= dropout_bool,
              dropout_layers=FCNN_dropouts, # Testing if dropout & batchnorm in last layer is effective ; TODO : implement possibility in NN
              batchnorm_bool=batchnorm_bool,
              batchnorm_layers=FCNN_batchnorms,
              activ_funcs_graphconv= graphconvs_activation_functions,
              graphconvs=graphconvs) #.to(device)



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


    EPOCHS = 150
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials = EPOCHS)

    trial = study.best_trial

    print("Best Concordance", trial.value)
    print("Best Hyperparamters : {}".format(trial.params))




def train(module,
          device,
          batch_size =128,
          n_epochs = 512,
          l2_regularization = False,
          val_batch_size=20,
          number_folds = 5,
          feature_names = None,
          n_train_samples = 0,
          n_test_samples = 0,
          n_val_samples = 0,
          view_names = None,
          processing_bool = False):
    """

    :param module: basically the dataset to be used
    :param batch_size: batch size for training
    :param n_epochs: number of epochs
    :param lr_scheduler_type: type of lr_scheduler : 'lambda','mulitplicative','step','multistep','exponential',...
    :param l2_regularization : bool whether should be applied or not
    :param cross_validation_bool : bool whether cross validation is to be applied ; False means that we just have cross
                                   validation with one split (one validation set)
    """



    #Select method for feature selection
    edge_index, proteins_used, train_data, val_data, test_data, \
    train_duration, train_event, \
    val_duration, val_event, \
    test_duration, test_event = module.feature_selection('ppi', feature_names)



    # As we use PPI feature selection in GCN, we don't have multiple views structure : we don't need numpy transforms



    ############################# FOLD X ###################################
    for c_fold,fold in enumerate(train_data):

        print("Split {} : ".format(c_fold))
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
        edge_index = torch.LongTensor(edge_index).to(device)








        if processing_bool == True: # only normalize when we dont do preprocessing on data before feature selection
            for i in range(num_features):
                train_data[c_fold][:,:,i] = normalize(train_data[c_fold][:,:,i])
                val_data[c_fold][:,:,i] = normalize(val_data[c_fold][:,:,i])
                test_data[c_fold][:,:,i] = normalize(test_data[c_fold][:,:,i])


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
                  n_hidden_layer_dims=[512,256],
                  activ_funcs=['relu','relu'],
                  dropout_bool= True,
                  dropout_layers=['yes','yes','yes'], # Testing if dropout & batchnorm in last layer is effective ; TODO : implement possibility in NN
                  batchnorm_bool=True,
                  batchnorm_layers=['yes','yes','yes'],
                  activ_funcs_graphconv= ['relu'],
                  graphconvs=[5]).to(device)



        if l2_regularization == True:
            optimizer = Adam(net.parameters(), lr=0.0005, weight_decay=0.000001)
        else:
            optimizer = Adam(net.parameters(), lr=0.0005)

        model = CoxPH(net, optimizer)

        log = model.fit(train_data[c_fold],train_surv, batch_size, n_epochs, callbacks, verbose=True,
                        val_data=val_data_full, val_batch_size= val_batch_size)

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
        surv.iloc[:, :5].plot()
        plt.ylabel('S(t | x)')
        _ = plt.xlabel('Time')


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

