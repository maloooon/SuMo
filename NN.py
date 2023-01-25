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


class NN_changeable(nn.Module):
    def __init__(self,views,in_features, n_hidden_layers_dims =None,
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
        self.dropout = nn.Dropout(dropout_prob)


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
          n_grid_search_iterations = 100):
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

        # Call NN
        # TODO : gleiche Durchläufe verhindern (check curr_config != new_config)
        for grid_count in range(n_grid_search_iterations):
            # Grid Search
            curr_config = []
            mRNA_layers = []
            DNA_layers = []
            microRNA_layers = []
            RPPA_layers = []
            for i in config["Layers_mRNA"]:
                curr_layer_size = random.choice(i)
                mRNA_layers.append(curr_layer_size)
            for i in config["Layers_DNA"]:
                curr_layer_size = random.choice(i)
                DNA_layers.append(curr_layer_size)
            for i in config["Layers_microRNA"]:
                curr_layer_size = random.choice(i)
                microRNA_layers.append(curr_layer_size)
            for i in config["Layers_RPPA"]:
                curr_layer_size = random.choice(i)
                RPPA_layers.append(curr_layer_size)

            # Set layers to views we currently look at
            layers = [mRNA_layers, DNA_layers]

            batch_size = random.choice(config["BatchSize"])
            val_batch_size = random.choice(config["BatchSizeVal"])
            learning_rate = random.choice(config["LearningRate"])
            dropout = random.choice(config["DropoutBool"])
            batchnorm = random.choice(config["BatchNormBool"])

            curr_config = [layers, batch_size, val_batch_size, learning_rate, dropout, batchnorm]











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
                                print_bool=False)




            # Set parameters for NN
            # set optimizer
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
                            val_batch_size= val_batch_size,
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

            print("With config : ")
            print("Layers : ", curr_config[0])
            print("Batch Size : ", curr_config[1])
            print("Validation Batch Size : ", curr_config[2])
            print("Learning Rate : ", curr_config[3])
            print("Dropout Bool : ", curr_config[4])
            print("BatchNorm Bool : ", curr_config[5])


            if concordance_index >= 0.6:
                configs_for_good_concordances[c_fold].append(curr_config)


            if concordance_index > curr_concordance:
                best_config = curr_config
                curr_concordance = concordance_index

            all_concordances[c_fold].append(curr_concordance)

        best_concordance_folds.append(curr_concordance)
        best_config_folds.append(best_config)



    print("Best concordances across folds : ", best_concordance_folds, "with config : ", best_config_folds)

    for c_fold in range(len(all_concordances)):
        print("Average concordance across fold for all grid search iterations",
              str(c_fold + 1), ": ", sum(all_concordances[c_fold])/len(all_concordances[c_fold]))
        print("Configs with concordances over 0.6 for current fold : ", len(configs_for_good_concordances[c_fold]))
        print("Respective Config : ", configs_for_good_concordances[c_fold])












def load_data(data_dir="/Users/marlon/Desktop/Project/PreparedData/Fold", n_fold = 1):

    trainset = pd.read_csv(
        os.path.join(data_dir + str(n_fold) + "_TrainData.csv"), index_col=0)

    valset = pd.read_csv(
        os.path.join(data_dir + str(n_fold) + "_ValData.csv"), index_col=0)

    testset = pd.read_csv(
        os.path.join(data_dir + str(n_fold) + "_TestData.csv"), index_col=0)


    return trainset, valset, testset









    """

    # Setup all the data
    n_train_samples, n_test_samples,view_names = module.setup()



    #Select method for feature selection
    module.feature_selection(method=feature_select_method,
                             components= components,
                             thresholds= thresholds,
                             feature_names= feature_names)

    # Load Dataloaders


    trainloader = module.train_dataloader(batch_size=n_train_samples)
    testloader =module.test_dataloader(batch_size=n_test_samples)

    # Load data and set device to cuda if possible

    #Train
    for train_data, train_duration, train_event in trainloader:
        for view in range(len(train_data)):
            train_data[view] = train_data[view].to(device=device)


        train_duration.to(device=device)
        train_event.to(device=device)


    for view in range(len(view_names)):
        print("Train data shape for view {} after feature selection {}".format(view_names[view],train_data[view].shape))


    #Test
    for test_data, test_duration, test_event in testloader:
        for view in range(len(test_data)):
            test_data[view] = test_data[view].to(device=device)
     #       test_data[view] = test_data[view].numpy()


        test_duration.to(device=device)
        test_event.to(device=device)


    for view in range(len(view_names)):
        print("Test data shape for view {} after feature selection {}".format(view_names[view],test_data[view].shape))




    # For PyCox, we need to change the structure so that all the data of different views is wrapped in a tuple
    train_data = tuple(train_data)
    test_data = tuple(test_data)




    # Input dimensions (features for each view) for NN based on different data (train/test)
    # Need to be the same for NN to work

    dimensions_train = [x.shape[1] for x in train_data]
    dimensions_test = [x.shape[1] for x in test_data]


    assert(dimensions_train == dimensions_test), "Dimensions have to be the same size in train and test for each view."



    dimensions = dimensions_train

    # Cross Validation:
    # We first need to concatenate all the data (train,test) together.
    # Since each view has a different feature size, we do that in the cross validation loop ;
    # duration & event can be done here already.

    full_data = []
    for view_idx in range(len(view_names)):
        data = torch.cat(tuple([train_data[view_idx],test_data[view_idx]]), dim=0)
        full_data.append(data)


    full_duration = torch.cat(tuple([train_duration,test_duration]),dim=0)
    full_event = torch.cat(tuple([train_event,test_event]),dim=0)

    # k is the number of folds
    k = number_folds
    # TODO : skicit learn split --> stratify for event / stratifiedKFold
    # TODO : gradient clipping
    # TODO : sonst auch nur train/test mit 1 fold --> train/test split
    splits = KFold(n_splits=k,shuffle=True,random_state=42)

    n_all_samples = n_train_samples + n_test_samples

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(n_all_samples))):
        torch.manual_seed(0)

        print('Fold {}'.format(fold + 1))


        event_boolean = False

        while event_boolean == False:
            np.random.seed(seed=0)
            train_sampler = SubsetRandomSampler(train_idx) # np.array e.g. [0 2 4 8 18 22 ..]
            test_sampler = SubsetRandomSampler(val_idx)






            # we take a subset of our train samples for the validation set
            # 20% of train is validation set
            # We first need to turn the train_sampler into a np.array (else the method won't work)
            train_sampler_array = np.empty(len(train_sampler))

            for c,idx in enumerate(train_sampler):
                np.put(train_sampler_array,c,idx)

            val_sampler = np.random.choice(train_sampler_array, int(0.2 * len(train_sampler)))

            #change dtype
            val_sampler = val_sampler.astype('int32')
            train_sampler_array = train_sampler_array.astype('int32')

            # get indices as list
            val_sampler_list = val_sampler.tolist()
            # remove these from train sampler
            train_sampler_array = np.setdiff1d(train_sampler_array, val_sampler_list)

            # Get duration & events
            train_duration = []
            train_event = []
            for idx in train_sampler_array:
                train_duration.append(full_duration[idx])
                train_event.append(full_event[idx])

            train_duration = torch.stack(tuple(train_duration))
            train_event = torch.stack(tuple(train_event))

            train_duration = train_duration.numpy()
            train_event = train_event.numpy()


            val_duration = []
            val_event = []
            for idx in val_sampler:
                val_duration.append(full_duration[idx])
                val_event.append(full_event[idx])

            val_duration = torch.stack(tuple(val_duration))
            val_event = torch.stack(tuple(val_event))

            val_duration = val_duration.numpy()
            val_event = val_event.numpy()

            # Check that each set contains atleast one event that is not censored




            test_duration = []
            test_event = []
            for idx in test_sampler:

                test_duration.append(full_duration[idx])
                test_event.append(full_event[idx])

            test_duration = torch.stack(tuple(test_duration))
            test_event = torch.stack(tuple(test_event))

            test_duration = test_duration.numpy()
            test_event = test_event.numpy()


            print("val non censored samples : ", np.count_nonzero(val_event))
            print("train non censored samples : ", np.count_nonzero(train_event))
            print("test non censored samples : ", np.count_nonzero(test_event))


            if np.count_nonzero(val_event) != 0 \
                    and np.count_nonzero(train_event) != 0 \
                    and np.count_nonzero(test_event) != 0:

                event_boolean = True


        # Numpy transforms for PyCox


        train_surv = (train_duration, train_event)




        # Get necessary data for each view
        train_data_full = []
        val_data_full = []
        test_data_full = []
        for view_idx in range(len(view_names)):

            # get training data
            train_data = []

            for idx in train_sampler_array:
                train_data.append(full_data[view_idx][idx])


            train_data = torch.stack(tuple(train_data))

            train_data_full.append(train_data.numpy())


            # get validation data
            val_data = []

            for idx in val_sampler:
                val_data.append(full_data[view_idx][idx])


            val_data = torch.stack(tuple(val_data))

            val_data_full.append(val_data.numpy())



            # get testing data
            test_data = []

            for idx in test_sampler:
                test_data.append(full_data[view_idx][idx])


            test_data = torch.stack(tuple(test_data))

            test_data_full.append(test_data.numpy())





        #Change into tuple structure
        train_data_full = tuple(train_data_full)
        val_data_full = tuple(val_data_full)
        test_data_full = tuple(test_data_full)


        # for PyCox
        val_data = (val_data_full, (val_duration, val_event))



        torch.manual_seed(0)

        # Call NN
        if fold == 0:
            net = NN_changeable(view_names,dimensions,[[8,4] for i in range(len(view_names))],
                                [['relu'],['relu'], ['relu'],['none']], dropout_rate,
                                dropout_per_view,
                                [['yes','yes'],['yes','yes'],['yes','yes'],['yes','yes']],
                                dropout,batch_norm_bool=False,print_bool=True)
        else:
            net = NN_changeable(view_names,dimensions, [[8,4] for i in range(len(view_names))],
                                [['relu'],['relu'], ['relu'],['none']], dropout_rate,
                                dropout_per_view,
                                [['yes','yes'],['yes','yes'],['yes','yes'],['yes','yes']],
                                dropout,batch_norm_bool=False,print_bool=False)


        # Dropout makes performance worse !
        # Batch norm only works if each batch has enough samples --> check that the last batch in an epoch (which may
        # get cut off since we don't have enough samples anymore) is bigger than 3-5 (?) samples for BatchNorm to work





        # Set parameters for NN
        # set optimizer
        if l2_regularization == True:
            optimizer = Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
        else:
            optimizer = Adam(net.parameters(), lr=0.001)

        callbacks = [tt.callbacks.EarlyStopping(patience=10)]



        # Call model
        model = models.CoxPH(net,optimizer)


        # Fit model
        log = model.fit(train_data_full,
                        train_surv,
                        batch_size,
                        n_epochs,
                        callbacks = callbacks,
                        val_data=val_data,
                        val_batch_size= val_batch_size,
                        verbose=True)



        # Plot it
        _ = log.plot()

        # Since Cox semi parametric, we calculate a baseline hazard to introduce a time variable
        _ = model.compute_baseline_hazards()


        # Predict based on test data
        surv = model.predict_surv_df(test_data_full)

        # Plot it
        surv.iloc[:, :5].plot()
        plt.ylabel('S(t | x)')
        _ = plt.xlabel('Time')




        # Evaluate with concordance, brier score and binomial log-likelihood
        ev = EvalSurv(surv, test_duration, test_event, censor_surv='km') # censor_surv : Kaplan-Meier

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
    """













