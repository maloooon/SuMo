import torch
import pandas as pd
from torch import nn
import numpy as np
from pycox import models
import DataInputNew
import torchtuples as tt
import matplotlib.pyplot as plt
from pycox.evaluation import EvalSurv
import ReadInData
from torch.optim import Adam
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler


class NN_changeable(nn.Module):
    def __init__(self,views,in_features,feature_offsets = None, n_hidden_layers_dims =None,
                 activ_funcs = None,dropout_prob = None, dropout_layers = None,
                 batch_norm = None, dropout_bool = None, batch_norm_bool = None, ae_bool = False, print_bool = False):
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
        self.feature_offsets = feature_offsets
        self.n_hidden_layers_dims = n_hidden_layers_dims
        self.activ_funcs = activ_funcs
        self.dropout_prob = dropout_prob
        self.dropout_layers = dropout_layers
        self.batch_norm = batch_norm
        self.dropout_bool = dropout_bool
        self.batch_norm_bool = batch_norm_bool
        self.ae_bool = ae_bool
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
                                                                                                self.in_features[0]))

            print("Dropout : {}, Batch Normalization : {}".format(dropout_bool, batch_norm_bool))
            for c,_ in enumerate(self.views):
                print("The view {} has the following pipeline : {}, dropout in layers : {}".format(_, self.hidden_layers[c],
                                                                                            dropout_layers[c]))

            print("Finally, the last output of each layer is summed up ({} features) and casted to a single element, "
                  "the hazard".format(sum_dim_last_layers))




    def forward(self,x):
        """

        :param x: data input
        :return: hazard
        """

        # list of lists to store encoded features for each view
        encoded_features = [[] for x in range(len(self.views))]

        #order data by views for diff. hidden layers
        data_ordered = []


        if self.ae_bool == False:
            #Get batch size
            batch_size = x.size(0)

            for view in range(len(self.views)):

                # Calculate the amount of features for current view via offset
                feature_size = self.feature_offsets[view+1] - self.feature_offsets[view]

                # Intialize an empty tensor to store data for current view
                temp = torch.empty(batch_size,feature_size)

                # fill empty tensor with according features for view for each sample in batch
                for i in range(batch_size):
                    temp[i, :] = x[i][self.feature_offsets[view] : self.feature_offsets[view+1]]

                data_ordered.append(temp)
        else:
            # Output of AE is already "processed"

            if type(x) is list:
                data_ordered = x
            else:
                data_ordered.append(x)

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
          device,
          batch_size =128,
          n_epochs = 512,
          l2_regularization = False,
          val_batch_size = 16,
          number_folds = 5): # TODO :CHANGE ÜBERALL
    """

    :param module: basically the dataset to be used
    :param batch_size: batch size for training
    :param n_epochs: number of epochs
    :param lr_scheduler_type: type of lr_scheduler : 'lambda','mulitplicative','step','multistep','exponential',...
    """




    # Setup all the data
    n_train_samples, n_test_samples, view_names = module.setup()



    #Select method for feature selection
    module.feature_selection(method='ae')

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


    print("Train data shape after feature selection {}".format(train_data[0].shape))


    #Test
    for test_data, test_duration, test_event in testloader:
        for view in range(len(test_data)):
            test_data[view] = test_data[view].to(device=device)


        test_duration.to(device=device)
        test_event.to(device=device)


    print("Test data shape after feature selection {}".format(test_data[0].shape))




    # Input dimensions (features for each view) for NN based on different data (train/test)
    # Need to be the same for NN to work
    dimensions_train = [x.size(1) for x in train_data]
    dimensions_test = [x.size(1) for x in test_data]

    assert (dimensions_train == dimensions_test), 'Feature mismatch between train/test'   # dimensions_val

    dimensions = dimensions_train


    # transforming data input for pycox : all views as one tensor, views accessible via feature offset

    # Get feature offsets for train/test
    # Need to be the same for NN to work
    feature_offsets_train = [0] + np.cumsum(dimensions_train).tolist()
    feature_offsets_test = [0] + np.cumsum(dimensions_test).tolist()

    assert (feature_offsets_train == feature_offsets_test), 'Feature offset mismatch between train/test'

    feature_offsets = feature_offsets_train

    # Number of all features (summed up) for train/validation/test
    # These need to be the same, otherwise NN won't work
    feature_sum_train = feature_offsets_train[-1]
    feature_sum_test = feature_offsets_test[-1]

    assert (feature_sum_train == feature_sum_test), 'Feature sum mismatch between train/test'


    feature_sum = feature_sum_train

    # Initialize empty tensors to store the data for train/validation/test
    train_data_pycox = torch.empty(n_train_samples, feature_sum).to(torch.float32)
    test_data_pycox = torch.empty(n_test_samples, feature_sum).to(torch.float32)



    # Fill up tensors with data

    # Train
    for idx_view,view in enumerate(train_data):
        for idx_sample, sample in enumerate(view):
            train_data_pycox[idx_sample][feature_offsets_train[idx_view]:
                                         feature_offsets_train[idx_view+1]] = sample

    # Test
    for idx_view,view in enumerate(test_data):
        for idx_sample, sample in enumerate(view):
            test_data_pycox[idx_sample][feature_offsets_test[idx_view]:
                                         feature_offsets_test[idx_view+1]] = sample



    # Cross Validation:
    # We first need to concatenate all the data (train,val,test) together again
    full_data = torch.cat(tuple([train_data_pycox,test_data_pycox]), dim=0)
    full_duration = torch.cat(tuple([train_duration,test_duration]),dim=0)
    full_event = torch.cat(tuple([train_event,test_event]),dim=0)

    # k is the number of folds
    k = number_folds

    splits = KFold(n_splits=k,shuffle=True,random_state=42)


    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(full_data)))):
        # TODO : assert that we have enough non censored data in each split ?

        print('Fold {}'.format(fold + 1))

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


        # get training data

        train_data = []
        train_duration = []
        train_event = []

        for idx in train_sampler_array:
            train_data.append(full_data[idx])
            train_duration.append(full_duration[idx])
            train_event.append(full_event[idx])

        train_data = torch.stack(tuple(train_data))
        train_duration = torch.stack(tuple(train_duration))
        train_event = torch.stack(tuple(train_event))

        # get validation data
        val_data = []
        val_duration = []
        val_event = []
        for idx in val_sampler:
            val_data.append(full_data[idx])
            val_duration.append(full_duration[idx])
            val_event.append(full_event[idx])

        val_data = torch.stack(tuple(val_data))
        val_duration = torch.stack(tuple(val_duration))
        val_event = torch.stack(tuple(val_event))


        # get testing data
        test_data = []
        test_duration = []
        test_event = []
        for idx in test_sampler:
            test_data.append(full_data[idx])
            test_duration.append(full_duration[idx])
            test_event.append(full_event[idx])

        test_data = torch.stack(tuple(test_data))
        test_duration = torch.stack(tuple(test_duration))
        test_event = torch.stack(tuple(test_event))


        val_full = (val_data, (val_duration,val_event))
        train_surv = (train_duration, train_event)
        # test_dur and test_ev need to be numpy arrays for EvalSurv()
        test_duration = test_duration.numpy()
        test_event = test_event.numpy()




        torch.manual_seed(0)
        # Call NN
        if fold == 0:
            net = NN_changeable(view_names,dimensions,feature_offsets,[[64,32,16] for i in range(len(view_names))],
                                [['relu'],['relu'],['relu'],['none']], 0.1,
                                [['yes','yes','yes'],['yes','yes','yes'],['yes', 'yes','yes'],['no','no','no']],
                                [['yes','yes'],['yes','yes'],['yes','yes'],['yes','yes']],
                                dropout_bool=False,batch_norm_bool=False,print_bool=True)
        else:
            net = NN_changeable(view_names,dimensions,feature_offsets,[[64,32,16] for i in range(len(view_names))],
                                [['relu'],['relu'],['relu'],['none']], 0.1,
                                [['yes','yes','yes'],['yes','yes','yes'],['yes', 'yes','yes'],['no','no','no']],
                                [['yes','yes'],['yes','yes'],['yes','yes'],['yes','yes']],
                                dropout_bool=False,batch_norm_bool=False,print_bool=False)


        # Dropout makes performance worse !
        # Batch norm only works if each batch has enough samples --> check that the last batch in an epoch (which may
        # get cut off since we don't have enough samples anymore) is bigger than 3-5 (?) samples for BatchNorm to work





        # Set parameters for NN
        # set optimizer
        if l2_regularization == True:
            optimizer = Adam(net.parameters(), lr=0.01, weight_decay=0.0001)
        else:
            optimizer = Adam(net.parameters(), lr=0.01)

        callbacks = [tt.callbacks.EarlyStopping(patience=10)]



        # Call model
        model = models.CoxPH(net,optimizer)


        # Fit model
        log = model.fit(train_data,
                        train_surv,
                        batch_size,
                        n_epochs,
                        callbacks = callbacks,
                        val_data=val_full,
                        val_batch_size= val_batch_size,
                        verbose=True)



        # Plot it
        _ = log.plot()

        # Since Cox semi parametric, we calculate a baseline hazard to introduce a time variable
        _ = model.compute_baseline_hazards()


        # Predict based on test data
        surv = model.predict_surv_df(test_data)

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










        # LR scheduler
#      if lr_scheduler_type.lower() == 'lambda':
#          lambda1 = lambda epoch: 0.65 ** epoch
#          scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda1)



#      if lr_scheduler_type.lower() == 'onecyclecos':
#          scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10)

#      lrs = []

#      for i in range(10):
#          optimizer.step()
#          lrs.append(optimizer.param_groups[0]["lr"])

#          scheduler.step()
