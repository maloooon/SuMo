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
from torchvision import models as tmodels
from torchsummary import summary
import ReadInData
import HelperFunctions as HF





from pycox.datasets import metabric
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

#TODO : how to check whether dropout layer, batch norm actually have an impact ? --> run with same data and with dropout_prob = 0
# TODO : didn't have wanted effect, didn't train at all (stopped at epoch 0)

class NN_changeable(nn.Module):
    def __init__(self,views,in_features,feature_offsets = None, n_hidden_layers_dims =None,
                 activ_funcs = None,dropout_prob = None, dropout_layers = None,
                 batch_norm = None, dropout_bool = None, batch_norm_bool = None, ae_bool = False):
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
        self.hidden_layers = [[] for x in range(len(in_features))] #TODO: add nn.ParameterList like in AE implementation !


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

         #   if type(data_ordered[0]) is list:
         #       # If AE has type 'none' , we need to flatten the list (as we have a list of lists of tensors)
         #       data_ordered = HF.flatten(data_ordered)



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












def train(module,device, batch_size =128, n_epochs = 512, lr_scheduler_type = 'onecyclecos'): # TODO :CHANGE ÜBERALL
    """

    :param module: basically the dataset to be used
    :param batch_size: batch size for training
    :param n_epochs: number of epochs
    :param lr_scheduler_type: type of lr_scheduler : 'lambda','mulitplicative','step','multistep','exponential',...
    """




    # Setup all the data
    n_train_samples, n_test_samples, n_val_samples, view_names = module.setup() # TODO : CHANGE ÜBERALL



    #Select method for feature selection
    module.feature_selection(method='variance')

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

    # test_dur and test_ev need to be numpy arrays for EvalSurv()
    test_duration = test_duration.numpy()
    test_event = test_event.numpy()
    # Input dimensions (features for each view) for NN based on different data (train/validation/test)
    # Need to be the same for NN to work
    dimensions_train = [x.size(1) for x in train_data]
    dimensions_val = [x.size(1) for x in val_data]
    dimensions_test = [x.size(1) for x in test_data]

    assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/val/test'

    dimensions = dimensions_train


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
    train_de_pycox = (train_duration, train_event)
    val_dde_pycox = val_data_pycox, (val_duration, val_event)
    test_d_pycox = test_data_pycox
    #test_dde_pycox = torch.cat((test_data_pycox,
    #                            test_duration.unsqueeze(dim=1),
    #                            event_temporary_placeholder_test.unsqueeze(dim=1)), dim=1)

    test_d_pycox_df = pd.DataFrame(test_d_pycox.numpy())



    # Call NN
    net = NN_changeable(view_names,dimensions,feature_offsets,[[10,5] for i in range(len(view_names))],
                        [['relu'],['relu'],['relu'],['none']], 0.1,
                        [['yes','yes'],['yes','yes',],['yes', 'yes'],['yes','yes']],
                        [['yes','yes'],['yes','yes'],['yes','yes'],['yes','yes']],
                        dropout_bool=False,batch_norm_bool=False)
    #TODO : Batch norm problem (see AE)

#    net = NN_changeable(views, dimensions, feature_offsets_train,
#                        n_hidden_layers_dims = [[10,5]], activ_funcs = [['relu'],['relu']],
#                        dropout_prob= 0.2,dropout_layers = [['yes' ,'yes']],
#                        batch_norm = [['yes','yes']], dropout_bool= True, batch_norm_bool= False)



    # Set parameters for NN
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    callbacks = [tt.callbacks.EarlyStopping()]


    # LR scheduler

    #TODO : working ?
    if lr_scheduler_type.lower() == 'lambda':
        lambda1 = lambda epoch: 0.65 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda1)



    if lr_scheduler_type.lower() == 'onecyclecos':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10)



    lrs = []

    for i in range(10):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])

        scheduler.step()




    # Call model
    model = models.CoxPH(net,optimizer)
  #  model2 = models.CoxPH(net_test,optimizer)

    # Set learning rate
  #  model.optimizer.set_lr(0.01)
 #   model2.optimizer.set_lr(0.01)





    # Fit model
    log = model.fit(train_data_pycox,train_de_pycox,batch_size,n_epochs,callbacks = callbacks,
                    verbose=True,val_data=val_dde_pycox, val_batch_size=10)

 #   log2 = model2.fit(train_data_pycox,train_de_pycox,batch_size,n_epochs,callbacks,
 #                     verbose=True,val_data=val_dde_pycox, val_batch_size=5)


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





if __name__ == '__main__':
    cancer_data = ReadInData.readcancerdata()
    multimodule = DataInputNew.SurvMultiOmicsDataModule(cancer_data[0][0],cancer_data[0][1],cancer_data[0][2],onezeronorm_bool=False)
   # views = cancer_data[0][2] # TODO: CHANGE ÜBERALL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(module= multimodule, device=device, batch_size=32, n_epochs=100) # TODO : CHANGE ÜBERALL (views raus)

