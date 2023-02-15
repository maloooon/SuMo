import copy
import pandas as pd
import torch
from torch import nn
import numpy as np
from pycox import models
import torchtuples as tt
from pycox.evaluation import EvalSurv
from torch.optim import Adam
import os
import optuna


class NN_changeable(nn.Module):
    def __init__(self,views,in_features,n_hidden_layers_dims =None,
                 activ_funcs = None,dropout_prob = None, dropout_layers = None,
                 batch_norm = None, dropout_bool = None, batch_norm_bool = None,print_bool = False,
                 prelu_init = 0.25):
        """
        Fully Connected Neural Net with changeable hyperparameters. Each view has a FCNN itself, finally the output of
        each view is concatenated and passed through a final layer, which compresses values to a single dimensional
        value used for the Proportional Hazards Model.
        :param views: Views (Omes) ; dtype : List of Strings
        :param in_features: Input dimensions for each view : List of Int
        :param n_hidden_layers_dims: Hidden layers for each view : List of Lists of Int
        :param activ_funcs: Activation Functions (for each view) aswell as for the last layer
                           ; dtype : List of Lists of Strings ['relu', 'sigmoid', 'prelu']
        :param dropout : Probability of Neuron Dropouts ; dtype : Int
        :param dropout_layers : Layers in which to apply Dropout ; dtype : List of Lists of Strings ['yes','no']
        :param batch_norm : Layers in which to apply Batch Normalization ; dtype : List of Lists of Strings ['yes','no']
        :param dropout_bool : Decide whether Dropout is to be applied or not ; dtype : Boolean
        :param batch_norm_bool : Decide whether Batch Normalization is to be applied or not ; dtype : Boolean
        :param ae_bool : Check if data input comes from an Autoencoder [needed because of different data structure]
                         ; dtype : Boolean
        :param print_bool : Decide whether to print the model ; dtype : Boolean
        :param prelu_init : Initial Value for PreLU activation ; dtype : Int
        """


        super().__init__()
        self.views =views
        self.in_features = in_features
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
        self.prelu_init = prelu_init



        # If we just input one activation function, use this activation function for each view and also the final layer
        if len(activ_funcs) == 1 and type(activ_funcs[0]) is not list:

            func = activ_funcs[0]
            activ_funcs = [[func] for x in range(len(views) + 1)]



        if len(activ_funcs) == len(views) + 1:

            for c,view in enumerate(activ_funcs):
                # If only one activ function given in sublist, use this for each layer
                if len(activ_funcs[c]) == 1 and c != len(views):
                    # -1 because we already have one activ func in our activ funcs list
                    for x in range(len(n_hidden_layers_dims[c]) -1):
                        activ_funcs[c].append(activ_funcs[c][0])

                for c2,activfunc in enumerate(view):
                    if activfunc.lower() == 'relu':
                        activ_funcs[c][c2] = nn.ReLU()
                    elif activfunc.lower() == 'sigmoid':
                        activ_funcs[c][c2] = nn.Sigmoid()
                    elif activfunc.lower() == 'prelu':
                        activ_funcs[c][c2] = nn.PReLU(init=prelu_init)


        else:
            raise ValueError("Your activation function input seems to be wrong. Check if it is a list of lists with a"
                             " sublist for each view and one list for the output layer or just a single activation function"
                             " value in a list")

        # Assign Layers
        for c,view in enumerate(n_hidden_layers_dims):
            for c2 in range(len(view)):
                if c2 == 0: # First Layer
                    # Batch Normalization
                    if batch_norm_bool == True and batch_norm[c][c2] == 'yes':
                        # Use an activation function
                        if activ_funcs[c][c2] != 'none':
                            self.hidden_layers[c].append(nn.Sequential(nn.Linear(in_features[c],
                                                                                 n_hidden_layers_dims[c][c2]),
                                                                                 nn.BatchNorm1d(n_hidden_layers_dims[c][c2]),
                                                                                 activ_funcs[c][c2]))
                        # Use no activation function
                        else:
                            self.hidden_layers[c].append(nn.Sequential(nn.Linear(in_features[c],
                                                                                 n_hidden_layers_dims[c][c2]),
                                                                       nn.BatchNorm1d(n_hidden_layers_dims[c][c2])
                                                                       ))
                    # No Batch Normalization
                    else:
                        # Use an activation function
                        if activ_funcs[c][c2] != 'none':
                            self.hidden_layers[c].append(nn.Sequential(nn.Linear(in_features[c],
                                                                                 n_hidden_layers_dims[c][c2]),
                                                                                 activ_funcs[c][c2]))
                        # Use no activation function
                        else:
                            self.hidden_layers[c].append(nn.Sequential(nn.Linear(in_features[c],
                                                                                 n_hidden_layers_dims[c][c2])
                                                                       ))


                else: # Other Layers
                    # Batch Normalization
                    if batch_norm_bool == True and batch_norm[c][c2] == 'yes':
                        # Use an activation function
                        if activ_funcs[c][c2] != 'none':
                            self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][c2-1],
                                                                                 n_hidden_layers_dims[c][c2]),
                                                                                 nn.BatchNorm1d(n_hidden_layers_dims[c][c2]),
                                                                                 activ_funcs[c][c2]))
                        # Use no activation function
                        else:
                            self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][c2-1],
                                                                                 n_hidden_layers_dims[c][c2]),
                                                                       nn.BatchNorm1d(n_hidden_layers_dims[c][c2])
                                                                       ))
                    # No Batch Normalization
                    else:
                        # Use an activation function
                        if activ_funcs[c][c2] != 'none':
                            self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][c2-1],
                                                                                 n_hidden_layers_dims[c][c2]),
                                                                                 activ_funcs[c][c2]))
                        # Use no activation function
                        else:
                            self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][c2-1],
                                                                                 n_hidden_layers_dims[c][c2])
                                                                       ))









        sum_dim_last_layers = sum([dim[-1] for dim in n_hidden_layers_dims])

        # Final Layer


        if activ_funcs[-1][0] != 'none':
            # Activation function
            if batch_norm_bool == True and batch_norm[-1][0] == 'yes':
                # Batch Normalization
                self.final_out = nn.Sequential(nn.Linear(sum_dim_last_layers,1),nn.BatchNorm1d(1), activ_funcs[-1][0])
            else:
                # No Batch Normalization
                self.final_out = nn.Sequential(nn.Linear(sum_dim_last_layers,1), activ_funcs[-1][0])

        else:
            # No activation function
            if batch_norm_bool == True and batch_norm[-1][0] == 'yes':
                self.final_out = nn.Sequential(nn.Linear(sum_dim_last_layers,1),nn.BatchNorm1d(1))
            else:
                self.final_out = nn.Sequential(nn.Linear(sum_dim_last_layers,1))




        # Dropout
        self.dropout = nn.Dropout(self.dropout_prob)


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
        Forward function of the Fully Connected Neural Net
        :param x: Data Input (for each view) ; dtype : Tuple/List of Tensor(n_samples_in_batch, n_features)
        :return: "Risk ratio" ; dtype : Tensor(n_samples_in_batch,1) TODO : Namen finden, den man auch in der BA dann benutzt
        """

        if type(x[0]) is list:
            x = tuple(x[0])
        # List of lists to store encoded features for each view
        encoded_features = [[] for x in range(len(self.views))]


        # Data ordered by view
        data_ordered = list(x)

        # Take arbitrary view for batch size, since for each view same batch size
        batch_size = x[0].size(0)

        # Pass data through layers and apply Dropout if wanted
        for c,view in enumerate(self.hidden_layers):
            for c2,encoder in enumerate(view):
                if c2 == 0: #first layer
                    # Apply dropout layer
                    if self.dropout_bool == True and self.dropout_layers[c][c2] == 'yes':
                      #  encoded_features[c][c2] = self.dropout(encoded_features[c][c2])
                        data_ordered[c] = self.dropout(data_ordered[c])
                    encoded_features[c].append(self.hidden_layers[c][c2](data_ordered[c]))


                else : # other layers
                    if self.dropout_bool == True and self.dropout_layers[c][c2] == 'yes':
                        encoded_features[c][c2-1] = self.dropout(encoded_features[c][c2-1])
                    encoded_features[c].append(self.hidden_layers[c][c2](encoded_features[c][c2-1]))




        # Concatenate output for final layer
        final_in = torch.cat(tuple([dim[-1] for dim in encoded_features]), dim=-1)


        if self.dropout_bool == True and self.dropout_layers[-1][0] == 'yes':
            final_in = self.dropout(final_in)

        predict = self.final_out(final_in)


        return predict







def objective(trial):
    """
    Optuna Optimization for Hyperparameters.
    :param trial: Settings of the current trial of Hyperparameters
    :return: Concordance Index ; dtype : Float
    """


    # Load in data
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


    # Split data in feature values, duration, event
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
 #   batch_size = trial.suggest_int("batch_size", 5, 200)
    batch_size = trial.suggest_categorical("batch_size", [8,16,32,64,128,256])
  #  n_epochs = trial.suggest_int("n_epochs", 10,100)
    n_epochs = 100
    dropout_prob = trial.suggest_float("dropout_prob", 0,0.5,step=0.1)
    dropout_bool = trial.suggest_categorical('dropout_bool', [True,False])
    batchnorm_bool = trial.suggest_categorical('batchnorm_bool',[True,False])
    prelu_rate = trial.suggest_float('prelu_rate',0,1,step=0.05)



    layers = []
    activation_functions = []
    dropouts = []
    batchnorms = []

    if 'MRNA' in view_names:
        layers_1_mRNA = trial.suggest_int('layers_1_mRNA', 5, 512)
        layers_2_mRNA = trial.suggest_int('layers_2_mRNA', 5, 512)
        layers_1_mRNA_activfunc = trial.suggest_categorical('layers_1_mRNA_activfunc', ['relu','sigmoid','prelu'])
        layers_2_mRNA_activfunc = trial.suggest_categorical('layers_2_mRNA_activfunc', ['relu','sigmoid','prelu'])
        layers_1_mRNA_dropout = trial.suggest_categorical('layers_1_mRNA_dropout', ['yes','no'])
        layers_2_mRNA_dropout = trial.suggest_categorical('layers_2_mRNA_dropout', ['yes','no'])
        layers_1_mRNA_batchnorm = trial.suggest_categorical('layers_1_mRNA_batchnorm', ['yes', 'no'])
        layers_2_mRNA_batchnorm = trial.suggest_categorical('layers_2_mRNA_batchnorm', ['yes', 'no'])

        layers.append([layers_1_mRNA,layers_2_mRNA])
        activation_functions.append([layers_1_mRNA_activfunc, layers_2_mRNA_activfunc])
        dropouts.append([layers_1_mRNA_dropout, layers_2_mRNA_dropout])
        batchnorms.append([layers_1_mRNA_batchnorm, layers_2_mRNA_batchnorm])

    if 'DNA' in view_names:
        layers_1_DNA = trial.suggest_int('layers_1_DNA', 5, 512)
        layers_2_DNA = trial.suggest_int('layers_2_DNA', 5, 512)
        layers_1_DNA_activfunc = trial.suggest_categorical('layers_1_DNA_activfunc', ['relu','sigmoid','prelu'])
        layers_2_DNA_activfunc = trial.suggest_categorical('layers_2_DNA_activfunc', ['relu','sigmoid','prelu'])
        layers_1_DNA_dropout = trial.suggest_categorical('layers_1_DNA_dropout', ['yes','no'])
        layers_2_DNA_dropout = trial.suggest_categorical('layers_2_DNA_dropout', ['yes','no'])
        layers_1_DNA_batchnorm = trial.suggest_categorical('layers_1_DNA_batchnorm', ['yes', 'no'])
        layers_2_DNA_batchnorm = trial.suggest_categorical('layers_2_DNA_batchnorm', ['yes', 'no'])
        layers.append([layers_1_DNA,layers_2_DNA])
        activation_functions.append([layers_1_DNA_activfunc, layers_2_DNA_activfunc])
        dropouts.append([layers_1_DNA_dropout, layers_2_DNA_dropout])
        batchnorms.append([layers_1_DNA_batchnorm, layers_2_DNA_batchnorm])

    if 'MICRORNA' in view_names:
        layers_1_microRNA = trial.suggest_int('layers_1_microRNA', 5, 512)
        layers_2_microRNA = trial.suggest_int('layers_2_microRNA', 5, 512)
        layers_1_microRNA_activfunc = trial.suggest_categorical('layers_1_microRNA_activfunc', ['relu','sigmoid','prelu'])
        layers_2_microRNA_activfunc = trial.suggest_categorical('layers_2_microRNA_activfunc', ['relu','sigmoid','prelu'])
        layers_1_microRNA_dropout = trial.suggest_categorical('layers_1_microRNA_dropout', ['yes','no'])
        layers_2_microRNA_dropout = trial.suggest_categorical('layers_2_microRNA_dropout', ['yes','no'])
        layers_1_microRNA_batchnorm = trial.suggest_categorical('layers_1_microRNA_batchnorm', ['yes', 'no'])
        layers_2_microRNA_batchnorm = trial.suggest_categorical('layers_2_microRNA_batchnorm', ['yes', 'no'])
        layers.append([layers_1_microRNA,layers_2_microRNA])
        activation_functions.append([layers_1_microRNA_activfunc, layers_2_microRNA_activfunc])
        dropouts.append([layers_1_microRNA_dropout, layers_2_microRNA_dropout])
        batchnorms.append([layers_1_microRNA_batchnorm, layers_2_microRNA_batchnorm])

    if 'RPPA' in view_names:
        layers_1_RPPA = trial.suggest_int('layers_1_microRNA', 5, 512)
        layers_2_RPPA = trial.suggest_int('layers_1_microRNA', 5, 512)
        layers_1_RPPA_activfunc = trial.suggest_categorical('layers_1_RPPA_activfunc', ['relu','sigmoid','prelu'])
        layers_2_RPPA_activfunc = trial.suggest_categorical('layers_2_RPPA_activfunc', ['relu','sigmoid','prelu'])
        layers_1_RPPA_dropout = trial.suggest_categorical('layers_1_RPPA_dropout', ['yes','no'])
        layers_2_RPPA_dropout = trial.suggest_categorical('layers_2_RPPA_dropout', ['yes','no'])
        layers_1_RPPA_batchnorm = trial.suggest_categorical('layers_1_RPPA_batchnorm', ['yes', 'no'])
        layers_2_RPPA_batchnorm = trial.suggest_categorical('layers_2_RPPA_batchnorm', ['yes', 'no'])
        layers.append([layers_1_RPPA,layers_2_RPPA])
        activation_functions.append([layers_1_RPPA_activfunc, layers_2_RPPA_activfunc])
        dropouts.append([layers_1_RPPA_dropout, layers_2_RPPA_dropout])
        batchnorms.append([layers_1_RPPA_batchnorm, layers_2_RPPA_batchnorm])


    # Last layer
    layer_final_activfunc = trial.suggest_categorical('layers_final_activfunc', ['relu','sigmoid','prelu','none'])
    layer_final_dropout = trial.suggest_categorical('layer_final_dropout', ['yes','no'])
    layer_final_batchnorm = trial.suggest_categorical('layer_final_batchnorm', ['yes','no'])
    activation_functions.append([layer_final_activfunc])
    dropouts.append([layer_final_dropout])
    batchnorms.append([layer_final_batchnorm])


    net = NN_changeable(views=view_names,
                              in_features=dimensions,
                              n_hidden_layers_dims=layers,
                              activ_funcs=activation_functions,
                              dropout_prob=dropout_prob,
                              dropout_layers=dropouts,
                              batch_norm=batchnorms,
                              dropout_bool=dropout_bool,
                              batch_norm_bool=batchnorm_bool,
                              print_bool=False,
                              prelu_init= prelu_rate
                              )


    if l2_regularization_bool == True:
        optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=l2_regularization_rate)
    else:
        optimizer = Adam(net.parameters(), lr=learning_rate)

    callbacks = [tt.callbacks.EarlyStopping(patience=10)]


    model = models.CoxPH(net,optimizer)
    print_loss = True

    # Fit model
    log = model.fit(train_data,
                    train_surv,
                    batch_size,
                    n_epochs,
                    callbacks = callbacks,
                    val_data=val_data_full,
                    val_batch_size= batch_size,
                    verbose=print_loss)


    # Plot it
    _ = log.plot()

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

    # Concordance Index ; Used for Optimization
    concordance_index = ev.concordance_td()

    if concordance_index < 0.5:
        concordance_index = 1 - concordance_index


    # These two scores can also be used for Optimization if wanted
    #Brier score
    time_grid = np.linspace(test_duration.min(), test_duration.max(), 100)
    _ = ev.brier_score(time_grid).plot
    brier_score = ev.integrated_brier_score(time_grid)

    #Binomial log-likelihood
    binomial_score = ev.integrated_nbll(time_grid)


    return concordance_index



def optuna_optimization():
    """
    Optuna Optimization for Hyperparameters.
    """


    # Set amount of different trials
    EPOCHS = 500
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials = EPOCHS)

    trial = study.best_trial

    print("Best Concordance", trial.value)
    print("Best Hyperparameters : {}".format(trial.params))



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
          view_names):
    """
    Training Function for the Fully Connected Neural Net, which connects the FCNN with the PH-Model.
    :param train_data: Training Data for each fold for each view  ; dtype : List of Lists [for each view] of Tensors(n_samples,n_features)
    :param val_data: Validation Data for each fold for each view  ; dtype : List of Lists [for each view] of Tensors(n_samples,n_features)
    :param test_data: Test Data for each fold for each view  ; dtype : List of Lists [for each view] of Tensors(n_samples,n_features)
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
    :param layers: Dimension of Layers for each view ; dtype : List of lists of Ints
    :param activation_layers: Activation Functions (for each view) aswell as for the last layer, the last layer can
    have no activation function ['none'] ; dtype : List of Lists of Strings ['relu', 'sigmoid', 'prelu']
    :param dropout: Decide whether Dropout is to be applied or not ; dtype : Boolean
    :param dropout_rate: Probability of Neuron Dropouts ; dtype : Int
    :param dropout_layers:  Layers in which to apply Dropout ; dtype : List of Lists of Strings ['yes','no']
    :param batchnorm: Decide whether Batch Normalization is to be applied or not ; dtype : Boolean
    :param batchnorm_layers: Layers in which to apply Batch Normalization ; dtype : List of Lists of Strings ['yes','no']
    :param view_names: Names of used views ; dtype : List of Strings
    """



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


        layers_u = copy.deepcopy(layers)
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
                            print_bool=False,
                            prelu_init= prelu_rate)



        if l2_regularization == True:
            optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=l2_regularization_rate)
        else:
            optimizer = Adam(net.parameters(), lr=learning_rate)

        callbacks = [tt.callbacks.EarlyStopping(patience=10)]


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










def load_data(data_dir="/Users/marlon/Desktop/Project/PreparedData/"):
    """
    Function to load data. Needed for Optuna Optimization.
    :param data_dir: Directory in which data is stored.
    :return: data and feature offsets (for feature values, duration and event)
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


    return trainset, trainset_feat, valset, valset_feat, testset, testset_feat







