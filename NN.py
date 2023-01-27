import copy
import pandas as pd
import torch
from torch import nn
import numpy as np
from pycox import models
import torchtuples as tt
import matplotlib.pyplot as plt
from pycox.evaluation import EvalSurv
from torch.optim import Adam
from functools import partial
import os
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
import random
from ray.tune.schedulers import ASHAScheduler
import optuna


class NN_changeable(nn.Module):
    def __init__(self,views,in_features, trial, n_hidden_layers_dims =None,
                 activ_funcs = None,dropout_prob = None, dropout_layers = None,
                 batch_norm = None, dropout_bool = None, batch_norm_bool = None,print_bool = False):
        """
        :param views: list of views (strings)
        :param in_features: list of input features
        :param feature_offsets: List of feature offsets over all views
        :param n_hidden_layers_dims: List of lists containing output_dim of each hidden layer for each view,
                                     where the length of each list is the amount of hidden layers for this view
        :param activ_funcs: List of lists containing activation functions for each hidden layer. This list has
                            one more list than n_hidden_layers_dim to determine the activation function for the final
                            layer. If the sublist only contains one activation function , this is to be used
                            for each hidden layer for this view. If the list only contains one value (no sublists), this
                            activation function is to be used for each hidden layer and the output layer. Activation func
                            are to be put in as strings. 'relu', 'sigmoid' , 'softmax' , ..
        :param dropout : probability of neuron dropout ; int
        :param dropout_layers : layers in n_hidden_layers_dims where dropout is to be applied ; str ('yes'/'no')
        :param batch_norm : layers in n_hidden_layers_dims where batch normalization is to be applied ; str ('yes'/'no')
        :param dropout_bool : Decide wether dropout is applied or not ; bool (True/False)
        :param batch_norm_bool : Decide wether batch normalization is applied or not ; bool (True/False)
        :param ae_bool : Check whether we pass data from AE (concatenated or element wise avg)
        :param print_bool : Decide whether the model is to be printed (needed so we don't get a print for each validation set in cross validation

        """
        super().__init__()
        self.views =views
        self.in_features = in_features
        #self.feature_offsets = feature_offsets
        self.n_hidden_layers_dims = n_hidden_layers_dims

        self.activ_funcs = activ_funcs
        self.dropout_prob = dropout_prob
        self.dropout_layers = dropout_layers
        self.batch_norm = batch_norm
        self.dropout_bool = dropout_bool
        self.batch_norm_bool = batch_norm_bool
        # Create list of lists which will store each hidden layer call for each view
        self.hidden_layers = nn.ParameterList([nn.ParameterList([]) for x in range(len(in_features))])
        self.print_bool = print_bool






        # Produce activation functions list of lists


        if len(activ_funcs) == 1 and type(activ_funcs[0]) is not list:

            func = activ_funcs[0]
            activ_funcs = [[func] for x in range(len(views) + 1)]



        if len(activ_funcs) == len(views) + 1:

            for c,view in enumerate(activ_funcs):
                # if only one activ function given in sublist, use this for each layer
                # if we look at the output layer activ function, we only have the last layer (otherwise index error)
                if len(activ_funcs[c]) == 1 and c != len(views):
                    # -1 because we already have one activ func in our activ funcs list
                    for x in range(len(n_hidden_layers_dims[c]) -1):
                        activ_funcs[c].append(activ_funcs[c][0])

                for c2,activfunc in enumerate(view):
                    if activfunc.lower() == 'relu':
                        activ_funcs[c][c2] = nn.ReLU()
                    elif activfunc.lower() == 'sigmoid':
                        activ_funcs[c][c2] = nn.Sigmoid()
                    elif activfunc.lower() == 'softmax':
                        activ_funcs[c][c2] = nn.Softmax()


        else:
            raise ValueError("Your activation function input seems to be wrong. Check if it is a list of lists with a"
                             " sublist for each view and one list for the output layer or just a single activation function"
                             " value in a list")

        # Produce hidden layer list of lists
        for c,view in enumerate(n_hidden_layers_dims):
            for c2 in range(len(view)):
                if c2 == 0: # first layer
                    if batch_norm_bool == True and batch_norm[c][c2] == 'yes':
                        if activ_funcs[c][c2] != 'none':
                            self.hidden_layers[c].append(nn.Sequential(nn.Linear(in_features[c],
                                                                                 n_hidden_layers_dims[c][c2]),
                                                                                 nn.BatchNorm1d(n_hidden_layers_dims[c][c2]),
                                                                                 activ_funcs[c][c2]))
                        else:
                            self.hidden_layers[c].append(nn.Sequential(nn.Linear(in_features[c],
                                                                                 n_hidden_layers_dims[c][c2]),
                                                                       nn.BatchNorm1d(n_hidden_layers_dims[c][c2])
                                                                       ))

                    else:
                        if activ_funcs[c][c2] != 'none':
                            self.hidden_layers[c].append(nn.Sequential(nn.Linear(in_features[c],
                                                                                 n_hidden_layers_dims[c][c2]),
                                                                                 activ_funcs[c][c2]))
                        else:
                            self.hidden_layers[c].append(nn.Sequential(nn.Linear(in_features[c],
                                                                                 n_hidden_layers_dims[c][c2])
                                                                       ))


                else: # other layers
                    if batch_norm_bool == True and batch_norm[c][c2] == 'yes':
                        if activ_funcs[c][c2] != 'none':
                            self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][c2-1],
                                                                                 n_hidden_layers_dims[c][c2]),
                                                                                 nn.BatchNorm1d(n_hidden_layers_dims[c][c2]),
                                                                                 activ_funcs[c][c2]))
                        else:
                            self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][c2-1],
                                                                                 n_hidden_layers_dims[c][c2]),
                                                                       nn.BatchNorm1d(n_hidden_layers_dims[c][c2])
                                                                       ))
                    else:
                        if activ_funcs[c][c2] != 'none':
                            self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][c2-1],
                                                                                 n_hidden_layers_dims[c][c2]),
                                                                                 activ_funcs[c][c2]))
                        else:
                            self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][c2-1],
                                                                                 n_hidden_layers_dims[c][c2])
                                                                       ))








        # The final layer takes each output from each hidden layer sequence of each view, concatenates them together
        # and reduces them to output dim of 1
        sum_dim_last_layers = sum([dim[-1] for dim in n_hidden_layers_dims])

        if activ_funcs[-1][0] != 'none':
            self.final_out = nn.Sequential(nn.Linear(sum_dim_last_layers,1), activ_funcs[-1][0])
        else:
            self.final_out = nn.Sequential(nn.Linear(sum_dim_last_layers,1))



        # Dropout
        self.dropout = nn.Dropout(self.dropout_prob) # TODO:  hier self.dropout_prob oder der trial. Ausdruck ? möglicher Fehler!


        if print_bool == True:
            # Print the model
            print("Data input has the following views : {}, each containing {} features.".format(self.views,
                                                                                                    self.in_features))

            print("Dropout : {}, Batch Normalization : {}".format(dropout_bool, batch_norm_bool))
            for c,_ in enumerate(self.views):
                print("The view {} has the following pipeline : {}".format(_, self.hidden_layers[c],
                                                                                            ))
                if dropout_bool == True:
                    print("dropout in layers : {}".format(dropout_layers[c]))



            print("Finally, the last output of each layer is summed up ({} features) and casted to a single element, "
                  "the hazard".format(sum_dim_last_layers))







    def forward(self,*x):
        """

        :param x: data input
        :return: hazard
        """
        # Needed for concat first, none second AE implementation bc. somehow a tuple of a lsit of our data is given
        # back instead of just a tuple of the data
        if type(x[0]) is list: # CHECK
            x = tuple(x[0])
        # list of lists to store encoded features for each view
        encoded_features = [[] for x in range(len(self.views))]

        #order data by views for diff. hidden layers
    #    data_ordered = []


    #    if self.ae_bool == False:
            #Get batch size
    #        batch_size = x.size(0)

    #        for view in range(len(self.views)):

                # Calculate the amount of features for current view via offset
    #            feature_size = self.feature_offsets[view+1] - self.feature_offsets[view]

                # Intialize an empty tensor to store data for current view
    #            temp = torch.empty(batch_size,feature_size)

                # fill empty tensor with according features for view for each sample in batch
    #            for i in range(batch_size):
    #                temp[i, :] = x[i][self.feature_offsets[view] : self.feature_offsets[view+1]]

    #            data_ordered.append(temp)


   #     if type(x) is list or type(x) is tuple:
        data_ordered = x
    #    else:
    #        data_ordered.append(x)

        batch_size = x[0].size(0) # take arbitrary view for batch size, since for each view same batch size

        for c,view in enumerate(self.hidden_layers):
            for c2,encoder in enumerate(view):
                if c2 == 0: #first layer
                    encoded_features[c].append(self.hidden_layers[c][c2](data_ordered[c]))
                    # Apply dropout layer
                    if self.dropout_bool == True and self.dropout_layers[c][c2] == 'yes':
                        encoded_features[c][c2] = self.dropout(encoded_features[c][c2])

                else : # other layers
                    encoded_features[c].append(self.hidden_layers[c][c2](encoded_features[c][c2-1]))
                    if self.dropout_bool == True and self.dropout_layers[c][c2] == 'yes':
                        encoded_features[c][c2] = self.dropout(encoded_features[c][c2])




        final_in = torch.cat(tuple([dim[-1] for dim in encoded_features]), dim=-1)



        predict = self.final_out(final_in)



        return predict







def objective(trial):


    # Load in data (##### For testing for first fold, later on

    trainset, trainset_feat, valset, valset_feat, testset, testset_feat = load_data()

    trainset_feat = list(trainset_feat.values)
    for idx,_ in enumerate(trainset_feat):
        trainset_feat[idx] = trainset_feat[idx].item()

    valset_feat = list(valset_feat.values)
    for idx,_ in enumerate(valset_feat):
        valset_feat[idx] = valset_feat[idx].item()

    testset_feat = list(testset_feat.values)
    for idx,_ in enumerate(testset_feat):
        testset_feat[idx] = testset_feat[idx].item()


    train_data = []
    for c,feat in enumerate(trainset_feat):
        if c < len(trainset_feat) - 3: # train data views
            train_data.append(np.array((trainset.iloc[:, trainset_feat[c] : trainset_feat[c+1]]).values).astype('float32'))
        elif c == len(trainset_feat) - 3: # duration
            train_duration = (np.array((trainset.iloc[:, trainset_feat[c] : trainset_feat[c+1]]).values).astype('float32')).squeeze(axis=1)
        elif c == len(trainset_feat) -2: # event
            train_event = (np.array((trainset.iloc[:, trainset_feat[c] : trainset_feat[c+1]]).values).astype('float32')).squeeze(axis=1)

    train_data = tuple(train_data)

    val_data = []
    for c,feat in enumerate(valset_feat):
        if c < len(valset_feat) - 3: # train data views
            val_data.append(np.array((valset.iloc[:, valset_feat[c]: valset_feat[c + 1]]).values).astype('float32'))
        elif c == len(valset_feat) - 3: # duration
            val_duration = (np.array((valset.iloc[:, valset_feat[c]: valset_feat[c + 1]]).values).astype('float32')).squeeze(axis=1)
        elif c == len(valset_feat) -2: # event
            val_event = (np.array((valset.iloc[:, valset_feat[c]: valset_feat[c + 1]]).values).astype('float32')).squeeze(axis=1)

    test_data = []

    for c,feat in enumerate(testset_feat):
        if c < len(testset_feat) - 3: # train data views
            test_data.append(np.array((testset.iloc[:, testset_feat[c]: testset_feat[c + 1]]).values).astype('float32'))
        elif c == len(testset_feat) - 3: # duration
            test_duration = (np.array((testset.iloc[:, testset_feat[c]: testset_feat[c + 1]]).values).astype('float32')).squeeze(axis=1)
        elif c == len(testset_feat) -2: # event
            test_event = (np.array((testset.iloc[:, testset_feat[c]: testset_feat[c + 1]]).values).astype('float32')).squeeze(axis=1)



    views = []
    read_in = open('/Users/marlon/Desktop/Project/TCGAData/cancerviews.txt', 'r')
    for view in read_in:
            views.append(view)

    view_names = [line[:-1] for line in views]


    dimensions_train = [x.shape[1] for x in train_data]
    dimensions_val = [x.shape[1] for x in val_data]
    dimensions_test = [x.shape[1] for x in test_data]

    assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/test'

    dimensions = dimensions_train

    # Transforms for PyCox
    train_surv = (train_duration, train_event)
    val_data_full = (val_data, (val_duration, val_event))


    ##################################### HYPERPARAMETER SEARCH SETTINGS ##############################################
    l2_regularization_bool = trial.suggest_categorical('l2_regularization_bool', [True,False])
    learning_rate = trial.suggest_float("learning_rate", 1e-5,1e-1,log=True)
    l2_regularization_rate = trial.suggest_float("l2_regularization_rate", 1e-6,1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 5, 200) # TODO : batch size so wählen, dass train samples/ batch_size und val samples/batch_size nie 1 ergeben können, da sonst Error : noch besser error abfangen und einfach skippen, da selten passiert !
    n_epochs = trial.suggest_int("n_epochs", 10,100)
    dropout_prob = trial.suggest_float("dropout_prob", 0,0.5,step=0.1)
    dropout_bool = trial.suggest_categorical('dropout_bool', [True,False])
    batchnorm_bool = trial.suggest_categorical('batchnorm_bool',[True,False])

    layers_1_mRNA = trial.suggest_int('layers_1_mRNA', 5, 1200)
    layers_2_mRNA = trial.suggest_int('layers_2_mRNA', 5, 1200)
    layers_1_DNA = trial.suggest_int('layers_1_DNA', 5, 1200)
    layers_2_DNA = trial.suggest_int('layers_2_DNA', 5, 1200)
    layers_1_microRNA = trial.suggest_int('layers_1_microRNA', 5, 1200)
    layers_2_microRNA = trial.suggest_int('layers_1_microRNA', 5, 1200)
    layers_1_RPPA = trial.suggest_int('layers_1_microRNA', 5, 1200)
    layers_2_RPPA = trial.suggest_int('layers_1_microRNA', 5, 1200)

    layers_1_mRNA_activfunc = trial.suggest_categorical('layers_1_mRNA_activfunc', ['relu','sigmoid'])
    layers_2_mRNA_activfunc = trial.suggest_categorical('layers_2_mRNA_activfunc', ['relu','sigmoid'])
    layers_1_DNA_activfunc = trial.suggest_categorical('layers_1_DNA_activfunc', ['relu','sigmoid'])
    layers_2_DNA_activfunc = trial.suggest_categorical('layers_2_DNA_activfunc', ['relu','sigmoid'])
    layers_1_microRNA_activfunc = trial.suggest_categorical('layers_1_microRNA_activfunc', ['relu','sigmoid'])
    layers_2_microRNA_activfunc = trial.suggest_categorical('layers_2_microRNA_activfunc', ['relu','sigmoid'])
    layers_1_RPPA_activfunc = trial.suggest_categorical('layers_1_RPPA_activfunc', ['relu','sigmoid'])
    layers_2_RPPA_activfunc = trial.suggest_categorical('layers_2_RPPA_activfunc', ['relu','sigmoid'])

    layers_1_mRNA_dropout = trial.suggest_categorical('layers_1_mRNA_dropout', ['yes','no'])
    layers_2_mRNA_dropout = trial.suggest_categorical('layers_2_mRNA_dropout', ['yes','no'])
    layers_1_DNA_dropout = trial.suggest_categorical('layers_1_DNA_dropout', ['yes','no'])
    layers_2_DNA_dropout = trial.suggest_categorical('layers_2_DNA_dropout', ['yes','no'])
    layers_1_microRNA_dropout = trial.suggest_categorical('layers_1_microRNA_dropout', ['yes','no'])
    layers_2_microRNA_dropout = trial.suggest_categorical('layers_2_microRNA_dropout', ['yes','no'])
    layers_1_RPPA_dropout = trial.suggest_categorical('layers_1_RPPA_dropout', ['yes','no'])
    layers_2_RPPA_dropout = trial.suggest_categorical('layers_2_RPPA_dropout', ['yes','no'])

    layers_1_mRNA_batchnorm = trial.suggest_categorical('layers_1_mRNA_batchnorm', ['yes', 'no'])
    layers_2_mRNA_batchnorm = trial.suggest_categorical('layers_2_mRNA_batchnorm', ['yes', 'no'])

    layers_1_DNA_batchnorm = trial.suggest_categorical('layers_1_DNA_batchnorm', ['yes', 'no'])
    layers_2_DNA_batchnorm = trial.suggest_categorical('layers_2_DNA_batchnorm', ['yes', 'no'])

    layers_1_microRNA_batchnorm = trial.suggest_categorical('layers_1_microRNA_batchnorm', ['yes', 'no'])
    layers_2_microRNA_batchnorm = trial.suggest_categorical('layers_2_microRNA_batchnorm', ['yes', 'no'])

    layers_1_RPPA_batchnorm = trial.suggest_categorical('layers_1_RPPA_batchnorm', ['yes', 'no'])
    layers_2_RPPA_batchnorm = trial.suggest_categorical('layers_2_RPPA_batchnorm', ['yes', 'no'])


    layers = []
    activation_functions = []
    dropouts = []
    batchnorms = []

    if 'MRNA' in view_names:
        layers.append([layers_1_mRNA,layers_2_mRNA])
        activation_functions.append([layers_1_mRNA_activfunc, layers_2_mRNA_activfunc])
        dropouts.append([layers_1_mRNA_dropout, layers_2_mRNA_dropout])
        batchnorms.append([layers_1_mRNA_batchnorm, layers_2_mRNA_batchnorm])

    if 'DNA' in view_names:
        layers.append([layers_1_DNA,layers_2_DNA])
        activation_functions.append([layers_1_DNA_activfunc, layers_2_DNA_activfunc])
        dropouts.append([layers_1_DNA_dropout, layers_2_DNA_dropout])
        batchnorms.append([layers_1_DNA_batchnorm, layers_2_DNA_batchnorm])

    if 'MICRORNA' in view_names:
        layers.append([layers_1_microRNA,layers_2_microRNA])
        activation_functions.append([layers_1_microRNA_activfunc, layers_2_microRNA_activfunc])
        dropouts.append([layers_1_microRNA_dropout, layers_2_microRNA_dropout])
        batchnorms.append([layers_1_microRNA_batchnorm, layers_2_microRNA_batchnorm])

    if 'RPPA' in view_names:
        layers.append([layers_1_RPPA,layers_2_RPPA])
        activation_functions.append([layers_1_RPPA_activfunc, layers_2_RPPA_activfunc])
        dropouts.append([layers_1_RPPA_dropout, layers_2_RPPA_dropout])
        batchnorms.append([layers_1_RPPA_batchnorm, layers_2_RPPA_batchnorm])

    activation_functions.append(['none'])





    net = NN_changeable(views=view_names,
                              in_features=dimensions,
                              trial=trial,
                              n_hidden_layers_dims=layers,
                              activ_funcs=activation_functions,
                              dropout_prob=dropout_prob,
                              dropout_layers=dropouts,
                              batch_norm=batchnorms,
                              dropout_bool=dropout_bool,
                              batch_norm_bool=batchnorm_bool,
                              print_bool=False
                              )



    if l2_regularization_bool == True:
        optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=l2_regularization_rate)
    else:
        optimizer = Adam(net.parameters(), lr=learning_rate)

    callbacks = [tt.callbacks.EarlyStopping(patience=10)]


    model = models.CoxPH(net,optimizer)
    print_loss = False

    # Fit model
    log = model.fit(train_data,
                    train_surv,
                    batch_size,
                    n_epochs,
                    callbacks = callbacks,
                    val_data=val_data_full,
                    val_batch_size= batch_size,
                    verbose=print_loss)


    # Since Cox semi parametric, we calculate a baseline hazard to introduce a time variable
    _ = model.compute_baseline_hazards()


    # Predict based on test data
    surv = model.predict_surv_df(test_data)

    # Plot it
    #     surv.iloc[:, :5].plot()
    #     plt.ylabel('S(t | x)')
    #     _ = plt.xlabel('Time')




    # Evaluate with concordance, brier score and binomial log-likelihood
    ev = EvalSurv(surv, test_duration, test_event, censor_surv='km') # censor_surv : Kaplan-Meier

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

#    print("Concordance index : {} , Integrated Brier Score : {} , Binomial Log-Likelihood : {}".format(concordance_index,
#                                                                                                       brier_score,
#                                                                                                       binomial_score))
    return concordance_index














def optuna_optimization(fold = 1):


    EPOCHS = 150
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials = EPOCHS)

    trial = study.best_trial

    print("Best Concordance", trial.value)
    print("Best Hyperparamters : {}".format(trial.params))






def train(module,
          feature_select_method = 'eigengenes',
          components = None,
          thresholds = None,
          feature_names = None,
          learning_rate = 0.0001,
          batch_size =128,
          n_epochs = 512,
          l2_regularization = False,
          l2_regularization_rate = 0.000001,
          batchnorm = False,
          batchnorm_layers = None,
          dropout_layers = None,
          val_batch_size = 16,
          dropout_rate = 0.1,
          dropout = False,
          activation_layers = None,
          view_names = None,
          config = None,
          n_grid_search_iterations = 100,
          testing_config = None):
    """

    :param module: basically the dataset to be used
    :param batch_size: batch size for training
    :param n_epochs: number of epochs
    :param lr_scheduler_type: type of lr_scheduler : 'lambda','mulitplicative','step','multistep','exponential',...
    """


  #  # Setup all the data
  #  n_train_samples, n_test_samples,n_val_samples, view_names = module.setup()



    #Select method for feature selection
    train_data, val_data, test_data, \
    train_duration, train_event, \
    val_duration, val_event, \
    test_duration, test_event = module.feature_selection(method=feature_select_method,
                                                         components= components,
                                                         thresholds= thresholds,
                                                         feature_names= feature_names)



    # Cast to numpy arrays if necessary(if we get an error, we already have numpy arrays --> no need to cast)
    try:
        test_duration = test_duration.numpy()
        test_event = test_event.numpy()
    except AttributeError:
        pass


    for c,fold in enumerate(train_data):
        try:
            train_duration[c] = train_duration[c].numpy()
            train_event[c] = train_event[c].numpy()
            val_duration[c] = val_duration[c].numpy()
            val_event[c] = val_event[c].numpy()
        except AttributeError: # in this case already numpy arrays
            pass



        for c2,view in enumerate(fold):
            try:
                train_data[c][c2] = (train_data[c][c2]).numpy()
                val_data[c][c2] = (val_data[c][c2]).numpy()
                test_data[c][c2] = (test_data[c][c2]).numpy()
            except AttributeError:
                pass


        # Need tuple structure for PyCox
        train_data[c] = tuple(train_data[c])
        val_data[c] = tuple(val_data[c])
        test_data[c] = tuple(test_data[c])

    best_concordance_folds = []
    best_config_folds = []
    concordances = []
    all_concordances = [[] for _ in range(len(train_data))]
    configs_for_good_concordances = [[] for _ in range(len(train_data))]
    ############################# FOLD X ###################################
    for c_fold,fold in enumerate(train_data):

        for c2,view in enumerate(fold):

            print("Train data has shape : {} for view {}".format(train_data[c_fold][c2].shape, view_names[c2]))
            print("Validation data has shape : {} for view {}".format(val_data[c_fold][c2].shape, view_names[c2]))
            print("Test data has shape : {} for view {}".format(test_data[c_fold][c2].shape, view_names[c2]))


        dimensions_train = [x.shape[1] for x in train_data[c_fold]]
        dimensions_val = [x.shape[1] for x in val_data[c_fold]]
        dimensions_test = [x.shape[1] for x in test_data[c_fold]]

        assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/test'

        dimensions = dimensions_train


        # Transforms for PyCox
        train_surv = (train_duration[c_fold], train_event[c_fold])
        val_data_full = (val_data[c_fold], (val_duration[c_fold], val_event[c_fold]))


        curr_concordance = 0

        """
        # Call NN
        # TODO : gleiche Durchläufe verhindern (check curr_config != new_config)
        for grid_count in range(n_grid_search_iterations):
            # Grid Search
            curr_config = []
            mRNA_layers = []
            DNA_layers = []
            microRNA_layers = []
            RPPA_layers = []
            for i in testing_config["Layers_mRNA"]:
                curr_layer_size = random.choice(i)
                mRNA_layers.append(curr_layer_size)
            for i in testing_config["Layers_DNA"]:
                curr_layer_size = random.choice(i)
                DNA_layers.append(curr_layer_size)
            for i in testing_config["Layers_microRNA"]:
                curr_layer_size = random.choice(i)
                microRNA_layers.append(curr_layer_size)
            for i in testing_config["Layers_RPPA"]:
                curr_layer_size = random.choice(i)
                RPPA_layers.append(curr_layer_size)

            # Set layers to views we currently look at
            layers = [mRNA_layers, DNA_layers, microRNA_layers]

            batch_size = random.choice(testing_config["BatchSize"])
            val_batch_size = random.choice(testing_config["BatchSizeVal"])
            learning_rate = random.choice(testing_config["LearningRate"])
            dropout = random.choice(testing_config["DropoutBool"])
            batchnorm = random.choice(testing_config["BatchNormBool"])

            curr_config = [layers, batch_size, val_batch_size, learning_rate, dropout, batchnorm]

            """










        layers_u = [[64,32]]#copy.deepcopy(layers)
        activation_layers_u = copy.deepcopy(activation_layers)
        dropout_layers_u = copy.deepcopy(dropout_layers)
        batchnorm_layers_u = copy.deepcopy(batchnorm_layers)


        net = NN_changeable(views=view_names,
                            in_features=dimensions,
                            n_hidden_layers_dims= layers_u,
                            activ_funcs=activation_layers_u,
                            dropout_bool= dropout,
                            dropout_prob= dropout_rate,
                            dropout_layers= dropout_layers_u,
                            batch_norm_bool= batchnorm,
                            batch_norm= batchnorm_layers_u,
                            print_bool=False)




        # Set parameters for NN
        # set optimizer
      #  lr = trial.suggest_float("lr", 1e-5,1e-1, log=True)


        if l2_regularization == True:
            optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=l2_regularization_rate)
        else:
            optimizer = Adam(net.parameters(), lr=learning_rate)

        callbacks = [tt.callbacks.EarlyStopping(patience=10)]


        # TODO : validation & train batch size same size
        # Call model
        model = models.CoxPH(net,optimizer)
        print_loss = False
        print("Split {} : ".format(c_fold + 1))
        # Fit model
        log = model.fit(train_data[c_fold],
                        train_surv,
                        batch_size,
                        n_epochs,
                        callbacks = callbacks,
                        val_data=val_data_full,
                        val_batch_size= batch_size,
                        verbose=print_loss)


        # Plot it
  #      _ = log.plot()

        # Since Cox semi parametric, we calculate a baseline hazard to introduce a time variable
        _ = model.compute_baseline_hazards()


        # Predict based on test data
        surv = model.predict_surv_df(test_data[c_fold])

        # Plot it
   #     surv.iloc[:, :5].plot()
   #     plt.ylabel('S(t | x)')
   #     _ = plt.xlabel('Time')




        # Evaluate with concordance, brier score and binomial log-likelihood
        ev = EvalSurv(surv, test_duration, test_event, censor_surv='km') # censor_surv : Kaplan-Meier

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
        concordances.append(concordance_index)






       #     print("With config : ")
       #     print("Layers : ", curr_config[0])
       #     print("Batch Size : ", curr_config[1])
       #     print("Validation Batch Size : ", curr_config[2])
       #     print("Learning Rate : ", curr_config[3])
        #    print("Dropout Bool : ", curr_config[4])
        #    print("BatchNorm Bool : ", curr_config[5])


    #    if concordance_index >= 0.6:
          #  configs_for_good_concordances[c_fold].append(curr_config)


     #   if concordance_index > curr_concordance:
          #  best_config = curr_config
     #       curr_concordance = concordance_index

    #    all_concordances[c_fold].append(concordance_index)

    #best_concordance_folds.append(curr_concordance)




















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


    return trainset, trainset_feat, valset, valset_feat, testset, testset_feat



"""
def objective(trial):

    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5,1e-1),
        'n_hidden_layers_mRNA_1' : trial.suggest_int('n_hidden_layers_mRNA_1', 1, 1024),
        'n_hidden_layers_mRNA_2' : trial.suggest_int('n_hidden_layers_mRNA_2', 1, 1024),
        'n_hidden_layers_DNA_1' : trial.suggest_int('n_hidden_layers_DNA_1', 1, 1024),
        'n_hidden_layers_DNA_2' : trial.suggest_int('n_hidden_layers_DNA_2', 1, 1024),
        'n_hidden_layers_microRNA_1' : trial.suggest_int('n_hidden_layers_microRNA_1', 1, 1024),
        'n_hidden_layers_microRNA_2' : trial.suggest_int('n_hidden_layers_microRNA_2', 1, 1024),
        'n_hidden_layers_RPPA_1' : trial.suggest_int('n_hidden_layers_RPPA_1', 1, 1024),
        'n_hidden_layers_RPPA_2' : trial.suggest_int('n_hidden_layers_RPPA_2', 1, 1024),
        'activ_funcs_mRNA_1' : trial.suggest_categorical('activ_funcs_mRNA_1', ['relu', 'sigmoid']),
        'activ_funcs_mRNA_2' : trial.suggest_categorical('activ_funcs_mRNA_2', ['relu', 'sigmoid']),
        'activ_funcs_DNA_1' : trial.suggest_categorical('activ_funcs_DNA_1', ['relu', 'sigmoid']),
        'activ_funcs_DNA_2' : trial.suggest_categorical('activ_funcs_DNA_2', ['relu', 'sigmoid']),
        'activ_funcs_microRNA_1' : trial.suggest_categorical('activ_funcs_microRNA_1', ['relu', 'sigmoid']),
        'activ_funcs_microRNA_2' : trial.suggest_categorical('activ_funcs_microRNA_2', ['relu', 'sigmoid']),
        'activ_funcs_RPPA_1' : trial.suggest_categorical('activ_funcs_RPPA_1', ['relu', 'sigmoid']),
        'activ_funcs_RPPA_2' : trial.suggest_categorical('activ_funcs_RPPA_2', ['relu', 'sigmoid']),
        'dropout_prob' : trial.suggest_loguniform('dropout_prob', 0,1)
    }

    model = NN_changeable(params)
"""





