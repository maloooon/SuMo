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

                # Replace strings with actual activation functions
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

    direc_set = 'Desktop'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load in data
    dir = os.path.expanduser('~/{}/Project/PreparedData/'.format(direc_set))
    #    trainset, trainset_feat, valset, valset_feat, testset, testset_feat = load_data()

    trainset_0,trainset_1,trainset_2,trainset_3,trainset_4,valset_0,valset_1,valset_2,valset_3,valset_4,testset_0,testset_1,testset_2,testset_3,testset_4,trainset_feat_0, \
    trainset_feat_1,trainset_feat_2,trainset_feat_3,trainset_feat_4 = load_data(data_dir = dir)



    # Feature offsets need to be the same in train/val/test for each fold, otherwise NN wouldn't work (diff dimension inputs)
    feat_offs = [trainset_feat_0,trainset_feat_1,trainset_feat_2,trainset_feat_3,trainset_feat_4]


    for c2,_ in enumerate(feat_offs):
        feat_offs[c2] = list(feat_offs[c2].values)
        for idx,_ in enumerate(feat_offs[c2]):
            feat_offs[c2][idx] = feat_offs[c2][idx].item()


    # Split data in feature values, duration, event

    trainset = [trainset_0 ,trainset_1,trainset_2,trainset_3,trainset_4]
    valset = [valset_0 ,valset_1,valset_2,valset_3,valset_4]
    testset = [testset_0,testset_1,testset_2,testset_3,testset_4]
    n_folds = len(trainset)
    train_data_folds = []
    train_duration_folds = []
    train_event_folds = []
    val_data_folds = []
    val_duration_folds = []
    val_event_folds = []
    test_data_folds = []


    for c2,_ in enumerate(trainset):
        train_data = []
        for c,feat in enumerate(feat_offs[c2]):
            if c < len(feat_offs[c2]) - 3: # train data views
                data_np = np.array((trainset[c2].iloc[:, feat_offs[c2][c] : feat_offs[c2][c+1]]).values).astype('float32')
                data_tensor = torch.from_numpy(data_np).to(torch.float32)
                data_tensor = data_tensor.to(device)
                train_data.append(data_tensor)
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

        train_data = tuple(train_data)

        val_data = []
        for c,feat in enumerate(feat_offs[c2]):
            if c < len(feat_offs[c2]) - 3: # train data views
                data_np = np.array((valset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')
                data_tensor = torch.from_numpy(data_np).to(torch.float32)
                data_tensor = data_tensor.to(device)
                val_data.append(data_tensor)

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



        test_data = []
        for c,feat in enumerate(feat_offs[c2]):
            if c < len(feat_offs[c2]) - 3: # train data views
                data_np = np.array((testset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')
                data_tensor = torch.from_numpy(data_np).to(torch.float32)
                data_tensor = data_tensor.to(device)
                test_data.append(data_tensor)

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


    views = []
    dir = os.path.expanduser('~/{}/Project/TCGAData/cancerviews.txt'.format(direc_set))
    # dir = os.path.expanduser('/Users/marlon/Desktop/Project/TCGAData/cancerviews.txt')
    read_in = open(dir, 'r')
    for view in read_in:
        views.append(view)

    view_names = [line[:-1] for line in views]


    ######## TESTING PURPOSES ########
 #   view_names = ['MRNA','MICRORNA','RPPA']
    ####### TESTING PURPOSES ########


    # Current fold to be optimized
    c_fold = 0
    # Optimize each fold on its own


    ##################################### HYPERPARAMETER SEARCH SETTINGS ##############################################
    l2_regularization_bool = trial.suggest_categorical('l2_regularization_bool', [True,False])
    learning_rate = trial.suggest_float("learning_rate", 1e-5,1e-1,log=True)
    l2_regularization_rate = trial.suggest_float("l2_regularization_rate", 1e-6,1e-3, log=True)
    #   batch_size = trial.suggest_int("batch_size", 5, 200)
    batch_size = trial.suggest_categorical("batch_size", [32,64,128])
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
        layers_1_mRNA = trial.suggest_int('layers_1_mRNA', 32, 96)
        layers_2_mRNA = trial.suggest_int('layers_2_mRNA', 8, 32)
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
        layers_1_DNA = trial.suggest_int('layers_1_DNA', 32, 96)
        layers_2_DNA = trial.suggest_int('layers_2_DNA', 8, 32)
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
        layers_1_microRNA = trial.suggest_int('layers_1_microRNA', 32, 96)
        layers_2_microRNA = trial.suggest_int('layers_2_microRNA', 8, 32)
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
        layers_1_RPPA = trial.suggest_int('layers_1_RPPA', 32, 96)
        layers_2_RPPA = trial.suggest_int('layers_2_RPPA', 8, 32)
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





    dimensions_train = [x.shape[1] for x in train_data[c_fold]]
    dimensions_val = [x.shape[1] for x in val_data[c_fold]]
    dimensions_test = [x.shape[1] for x in test_data[c_fold]]

    assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/test'

    dimensions = dimensions_train

    # Transforms for PyCox
    train_surv = (train_duration[c_fold], train_event[c_fold])
    val_data_full = (val_data[c_fold], (val_duration[c_fold], val_event[c_fold]))


    layers_u = copy.deepcopy(layers)
    activation_layers_u = copy.deepcopy(activation_functions)
    dropout_layers_u = copy.deepcopy(dropouts)
    batchnorm_layers_u = copy.deepcopy(batchnorms)

    net = NN_changeable(views=view_names,
                        in_features=dimensions,
                        n_hidden_layers_dims=layers_u,
                        activ_funcs=activation_layers_u,
                        dropout_prob=dropout_prob,
                        dropout_layers=dropout_layers_u,
                        batch_norm=batchnorm_layers_u,
                        dropout_bool=dropout_bool,
                        batch_norm_bool=batchnorm_bool,
                        print_bool=False,
                        prelu_init= prelu_rate
                        ).to(device)


    if l2_regularization_bool == True:
        optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=l2_regularization_rate)
    else:
        optimizer = Adam(net.parameters(), lr=learning_rate)

    callbacks = [tt.callbacks.EarlyStopping(patience=10)]


    model = models.CoxPH(net,optimizer)
    model.set_device(torch.device(device))
    print_loss = False


    # Fit model
    log = model.fit(train_data[c_fold],
                    train_surv,
                    batch_size,
                    n_epochs,
                    callbacks = callbacks,
                    val_data=val_data_full,
                    val_batch_size= 5,
                    verbose=print_loss)


    # Plot it
   # _ = log.plot()

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

    # Concordance Index ; Used for Optimization
    concordance_index = ev.concordance_td()

    if concordance_index < 0.5:
        concordance_index = 1 - concordance_index


    # These two scores can also be used for Optimization if wanted
    #Brier score
    #   time_grid = np.linspace(test_duration.min(), test_duration.max(), 100)
    #   _ = ev.brier_score(time_grid).plot
    #   brier_score = ev.integrated_brier_score(time_grid)

    #Binomial log-likelihood
    #   binomial_score = ev.integrated_nbll(time_grid)


    return concordance_index



def optuna_optimization():
    """
    Optuna Optimization for Hyperparameters.
    """


    # Set amount of different trials
    EPOCHS = 100
    study = optuna.create_study(directions=['maximize'],sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials = EPOCHS)
    trial = study.best_trials
   # Show change of c-Index across folds
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show(renderer='browser')
    # Show hyperparameter importance
    fig = optuna.visualization.plot_param_importances(study)
    fig.show(renderer='browser')

    # Save the best trial for each fold
    direc_set = 'Desktop'
    dir = os.path.expanduser(r'~/{}/Project/Trial/FCNN_KIRC3_Standardize_PCA_BEST_3.txt'.format(direc_set))
    with open(dir, 'w') as fp:
        for item in trial:
            # write each item on a new line
            fp.write("%s\n" % item)


    # Save all trials in dataframe
#  df = study.trials_dataframe()
#    df = df.sort_values('value')
#  df.to_csv("~/SUMO/Project/Trial/FCNN_KIRC3_Standardize_PCA.csv")

# print("Best Concordance Sum", trial.value)
# print("Best Hyperparameters : {}".format(trial.params))


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


    for c,fold in enumerate(train_data):
        # Need tuple structure for PyCox
        train_data[c] = tuple(train_data[c])
        val_data[c] = tuple(val_data[c])
        test_data[c] = tuple(test_data[c])



    ############################# FOLD X ###################################
    for c_fold,fold in enumerate(train_data):

        for c2,view in enumerate(fold):

            # For GPU acceleration, we need to have everything as tensors for the training loop, but pycox EvalSurv
            # Needs duration & event to be numpy arrays, thus at the start we set duration/event to tensors
            # and before EvalSurv to numpy
            try:
                test_duration = torch.from_numpy(test_duration).to(torch.float32)
                test_event = torch.from_numpy(test_event).to(torch.float32)
            except TypeError:
                pass


            for c,fold in enumerate(train_data):
                try:
                    train_duration[c] = torch.from_numpy(train_duration[c]).to(torch.float32)
                    train_event[c] = torch.from_numpy(train_event[c]).to(torch.float32)
                    val_duration[c] = torch.from_numpy(val_duration[c]).to(torch.float32)
                    val_event[c] = torch.from_numpy(val_event[c]).to(torch.float32)
                except TypeError:
                    pass

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
                        val_batch_size= 5,
                        verbose=print_loss)


        # Plot it
  #      _ = log.plot()

        # Change for EvalSurv-Function
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


def load_data(data_dir):
    """
    Function to load data. Needed for Optuna Optimization.
    :param data_dir: Directory in which data is stored.
    :return: data and feature offsets (for feature values, duration and event)
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


    #testset_feat = pd.read_csv(
    #    os.path.join(data_dir + "TestDataFeatOffs.csv"), index_col=0)


    return trainset_0,trainset_1,trainset_2,trainset_3,trainset_4,valset_0,valset_1,valset_2,valset_3,valset_4,\
           testset_0,testset_1,testset_2,testset_3,testset_4, \
           trainset_feat_0,trainset_feat_1,trainset_feat_2,trainset_feat_3,trainset_feat_4

    # return trainset, trainset_feat, valset, valset_feat, testset, testset_feat







