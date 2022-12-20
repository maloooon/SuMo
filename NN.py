import torch
import pandas as pd
import os
import DataInputNew
from torch import nn
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
import torch.nn.functional as F
from pycox import models
import DataInputNew
import torchtuples as tt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pycox.evaluation import EvalSurv

class NN_changeable(nn.Module):
    def __init__(self,views,in_features,feature_offsets, n_hidden_layers_dims, activ_funcs):
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
        """
        super().__init__()
        self.views =views
        self.in_features = in_features
        self.feature_offsets = feature_offsets
        self.n_hidden_layers_dims = n_hidden_layers_dims
        self.activ_funcs = activ_funcs

        # Create list of lists which will store each hidden layer call for each view
        self.hidden_layers = [[] for x in range(len(in_features))]

        # Produce activation functions list of lists


        if len(activ_funcs) == 1:

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
                    self.hidden_layers[c].append(nn.Sequential(nn.Linear(in_features[c],
                                                                         n_hidden_layers_dims[c][c2]),
                                                                         activ_funcs[c][c2]))
                else: # other layers
                    self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][c2-1],
                                                                         n_hidden_layers_dims[c][c2]),
                                                                         activ_funcs[c][c2]))




        # The final layer takes each output from each hidden layer sequence of each view, concatenates them together
        # and reduces them to output dim of 1
        sum_dim_last_layers = sum([dim[-1] for dim in n_hidden_layers_dims])

        print(sum_dim_last_layers)
        self.final_out = nn.Sequential(nn.Linear(sum_dim_last_layers,1), activ_funcs[-1][0])



    def forward(self,x):
        """

        :param x: data input
        :return: hazard
        """

        # list of lists to store encoded features for each view
        encoded_features = [[] for x in range(len(self.views))]

        #order data by views for diff. hidden layers
        data_ordered = []


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

        for c,view in enumerate(self.hidden_layers):
            for c2,encoder in enumerate(view):
                if c2 == 0: #first layer
                    encoded_features[c].append(self.hidden_layers[c][c2](data_ordered[c]))
                else : # other layers
                    encoded_features[c].append(self.hidden_layers[c][c2](encoded_features[c][c2-1]))




        final_in = torch.cat(tuple([dim[-1] for dim in encoded_features]), dim=-1)



        predict = self.final_out(final_in)



        return predict













class NN_simple(nn.Module):
    def __init__(self,in_features,feature_offsets):
        """Simple NN with hidden layer structure for each view, concatenate everything in the end"""
        super().__init__()
        self.in_features = in_features # list of input features for each view
        self.feature_offsets = feature_offsets

        self.hidden_1 = [] # hidden layers for each view after input
        self.hidden_2 = [] # hidden layers for each view after hidden layers 1

        for view in in_features:
            layer = nn.Sequential(nn.Linear(view, 28), nn.ReLU())
            self.hidden_1.append(layer)
            layer_2 = nn.Sequential(nn.Linear(28,5), nn.ReLU())
            self.hidden_2.append(layer_2)

        self.out = nn.Sequential(nn.Linear(20,1), nn.ReLU()) # 1 output feature (hazard)



    def forward(self,x):
        """

        :param x: data input
        :return: predicted hazard
        """
        encoded_hidden_1 = []
        encoded_hidden_2 = []
        batch_size = x.size(0)
        for view in range(len(self.in_features)):

            feature_size = self.feature_offsets[view+1] - self.feature_offsets[view] # calculating amount of features for current view via offset
            temp = torch.empty(batch_size,feature_size)

            # fill empty tensor with according features for view for each sample in batch
            for i in range(batch_size):
                temp[i, :] = x[i][self.feature_offsets[view] : self.feature_offsets[view+1]]

            # temp is now a tensor which has only the features for a specific view for the whole batch
            # apply to hidden layer in column 1 for according view
            encoded_hidden_1.append(self.hidden_1[view](temp))

            # take the latent features from hidden column 1 for this view and pass to hidden column 2
            encoded_hidden_2.append(self.hidden_2[view](encoded_hidden_1[view]))


        # after everything is done, take all the outputs from hidden column 2 and pass to output layer


        final_in = torch.cat(tuple(encoded_hidden_2), dim=-1) # so that it becomes a single tensor of dim [batch_size, hidden column 2 out features]

        hazard_predict = self.out(final_in)



        return hazard_predict







def train(module, batch_size =5, n_epochs = 512, output_dim=1):
    """

    :param module: basically the dataset to be used
    :param batch_size: batch size for training
    :return:
    """

    views = ['mRNA', 'DNA', 'microRNA', 'RPPA']

    # Setup all the data
    n_train_samples, n_test_samples, n_val_samples = module.setup()

    #Select method for feature selection
    module.feature_selection(method='pca')

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

    print("Train data shape after feature selection {},{},{},{}".format(train_data[0].shape,
                                                                        train_data[1].shape,
                                                                        train_data[2].shape,
                                                                        train_data[3].shape))

    #Validation
    for val_data, val_duration, val_event in valloader:
        for view in range(len(val_data)):
            val_data[view] = val_data[view].to(device=device)

        val_duration.to(device=device)
        val_event.to(device=device)

    print("Validation data shape after feature selection {},{},{},{}".format(val_data[0].shape,
                                                                             val_data[1].shape,
                                                                             val_data[2].shape,
                                                                             val_data[3].shape))

    #Test
    for test_data, test_duration, test_event in testloader:
        for view in range(len(test_data)):
            test_data[view] = test_data[view].to(device=device)


        test_duration.to(device=device)
        test_event.to(device=device)

    print("Test data shape after feature selection {},{},{},{}".format(test_data[0].shape,
                                                                        test_data[1].shape,
                                                                        test_data[2].shape,
                                                                        test_data[3].shape))

    # test_dur and test_ev need to be numpy arrays for EvalSurv()
    test_duration = test_duration.numpy()
    test_event = test_event.numpy()
    print(test_duration)
    # Input dimensions (features for each view) for NN based on different data (train/validation/test)
    # Need to be the same for NN to work
    dimensions_train = [x.size(1) for x in train_data]
    dimensions_val = [x.size(1) for x in val_data]
    dimensions_test = [x.size(1) for x in test_data]

    assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/val/test'

    dimensions = dimensions_train

    # We'll use a fake event placeholder, bc current data has too many censored patients
    # which doesn't work with PyCox implementation
    # all events = 1 (no patient censored)
    event_temporary_placeholder_train = (torch.ones(n_train_samples).to(torch.int32)).to(device=device)
    event_temporary_placeholder_val = (torch.ones(n_val_samples).to(torch.int32)).to(device=device)
    event_temporary_placeholder_test = (torch.ones(n_test_samples).to(torch.int32)).to(device=device).numpy()
    print(event_temporary_placeholder_test)

    # transforming data input for pycox : all views as one tensor, views accessible via feature offset

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

    # Fill up tensors with data

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
    train_de_pycox = (train_duration, event_temporary_placeholder_train)
    val_dde_pycox = val_data_pycox, (val_duration, event_temporary_placeholder_val)
    test_d_pycox = test_data_pycox
    #test_dde_pycox = torch.cat((test_data_pycox,
    #                            test_duration.unsqueeze(dim=1),
    #                            event_temporary_placeholder_test.unsqueeze(dim=1)), dim=1)

    test_d_pycox_df = pd.DataFrame(test_d_pycox.numpy())



 #   net_test = NN_simple(dimensions, feature_offsets)


    # Call NN
    net = NN_changeable(views,dimensions,feature_offsets,[[25,12,6],[25,12,6],[25],[25,12]],
                        [['relu'],['relu','relu','relu'],['relu'],['relu','relu'],['relu']])

    # Set parameters for NN
    optimizer = torch.optim.Adam(net.parameters())
    callbacks = [tt.callbacks.EarlyStopping()]



    # Call model
    model = models.CoxPH(net,optimizer)

    # Set learning rate
    model.optimizer.set_lr(0.01)

    # Fit model
    log = model.fit(train_data_pycox,train_de_pycox,batch_size,n_epochs,callbacks,
                    verbose=True,val_data=val_dde_pycox, val_batch_size=5)


    # Plot it
    _ = log.plot()

    # Since Cox semi parametric, we calculate a baseline hazard to introduce a time variable
    _ = model.compute_baseline_hazards()

    # Predict based on test data
    surv = model.predict_surv_df(test_d_pycox)

    # Plot it
    surv.iloc[:, :5].plot()
    plt.ylabel('S(t | x)')
    _ = plt.xlabel('Time')


    # Evaluate with concordance, brier score and binomial log-likelihood
    ev = EvalSurv(surv, test_duration, event_temporary_placeholder_test, censor_surv='km') # censor_surv : Kaplan-Meier

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
    module = DataInputNew.multimodule
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(module= module)