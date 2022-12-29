import torch
import numpy as np
from torch import nn
import ReadInData
import DataInputNew
from torch.optim import Adam


class AE(nn.Module):
    """AE module, input is each view itself (AE for each view)"""
    def __init__(self, views, in_features, feature_offsets, n_hidden_layers_dims,
                 activ_funcs,dropout_prob, dropout_layers, batch_norm, dropout_bool, batch_norm_bool, type):
        """
        :param views: list of views (strings)
        :param in_features: list of input features
        :param feature_offsets: List of feature offsets over all views
        :param n_hidden_layers_dims: List of lists containing output_dim of each hidden layer for each view,
                                     where the length of each list is the amount of encoder layers. For the decoding
                                     layers, the list will be read backwards.
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
        :param type : concat, cross or both (concross)
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
        self.type = type
        # Create list of lists which will store each hidden layer call for each view (encoding & decoding stage)
        self.hidden_layers = nn.ParameterList([nn.ParameterList([]) for x in range(len(in_features))])
        self.middle_dims = []


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


        # Produce hidden layer list of lists for encoding stage
        # For each view, we add the dimensions of the hidden layers for the encoder backwards to the list (for decoding)
        # Accordingly, we need to do the same for activation functions, batch norm and dropout layers (mirroring it for decoding)

        for c,view in enumerate(n_hidden_layers_dims):
            # For output purposes, we save dim of the last layer for the encoder in a list
            self.middle_dims.append(n_hidden_layers_dims[c][-1])
            # Create copy of hidden layer of current view
            temp_hidden = n_hidden_layers_dims[c].copy()
            temp_activation = activ_funcs[c].copy()
            temp_batch = batch_norm[c].copy()
            temp_dropout = dropout_layers[c].copy()
            # reverse it
            temp_hidden.reverse()
            temp_activation.reverse()
            temp_batch.reverse()
            temp_dropout.reverse()
            # concatenate temp to original list starting at first element (otherwise we would have the
            # element in the middle 2 times
            n_hidden_layers_dims[c] += temp_hidden[1:]
            activ_funcs[c] += temp_activation[1:]
            batch_norm[c] += temp_batch[1:]
            dropout_layers[c] += temp_dropout[1:]

            # Now we also need to add the final layer of the decoder, which produces the dimension of our original input



            for c2 in range(len(view) + 1):
                if c2 == 0: # first layer
                    if batch_norm_bool == True and batch_norm[c][c2] == 'yes':
                        self.hidden_layers[c].append(nn.Sequential(nn.Linear(in_features[c],
                                                                             n_hidden_layers_dims[c][c2]),
                                                                   nn.BatchNorm1d(n_hidden_layers_dims[c][c2]),
                                                                   activ_funcs[c][c2]))
                    else:
                        self.hidden_layers[c].append(nn.Sequential(nn.Linear(in_features[c],
                                                                             n_hidden_layers_dims[c][c2]),
                                                                             activ_funcs[c][c2]))


                elif c2 == len(view): # last layer
                    if batch_norm_bool == True: # TODO : own input whether batch norm, dropout layer .. for last layer
                        self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][-1],
                                                                             in_features[c]),
                                                                   nn.BatchNorm1d(in_features[c]),
                                                                   activ_funcs[c][-1]))
                    else:
                        self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][-1],
                                                                             in_features[c]),
                                                                   activ_funcs[c][-1]))



                else: # other layers
                    if batch_norm_bool == True and batch_norm[c][c2] == 'yes':
                        self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][c2-1],
                                                                             n_hidden_layers_dims[c][c2]),
                                                                   nn.BatchNorm1d(n_hidden_layers_dims[c][c2]),
                                                                   activ_funcs[c][c2]))
                    else:
                        self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][c2-1],
                                                                             n_hidden_layers_dims[c][c2]),
                                                                   activ_funcs[c][c2]))







        if type.lower() == 'concat':
            # Concatenate the output, which will then be passed to a NN for survival analysis
            # the final output we're interested in (middle between encoder and decoder) was therefore already
            # saved in middle_dims list

            concatenated_features = sum([dim for dim in self.middle_dims])


        if type.lower() == 'cross':
            #TODO : To be implemented
            pass



        # Dropout
        self.dropout = nn.Dropout(dropout_prob)

            # Print the model
        print("Data input has the following views : {}, each containing {} features.".format(self.views,
                                                                                             self.in_features[0]))

        print("Dropout : {}, Batch Normalization : {}".format(dropout_bool, batch_norm_bool))
        for c,_ in enumerate(self.views):
            print("The view {} has the following pipeline : {}, dropout in layers : {}".format(_, self.hidden_layers[c],
                                                                                               dropout_layers[c]))

        print("Finally, the output of each view between encoder and decoder  is summed up ({} features) "
              "and will be passed to a NN for survival analysis".format(concatenated_features))


    def forward(self,x):

        # list of lists to store encoded features for each view
        # Note that encoded features will have BOTH features for encoding stage and decoding stage !
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
            for c2 in range(len(view)): # view contains all layers (also the last to recreate input dim, as we already have defined it in __init__

                if c2 == 0: #first layer
                    encoded_features[c].append(self.hidden_layers[c][c2](data_ordered[c]))
                    # Apply dropout layer
                    if self.dropout_bool == True and self.dropout_layers[c][c2] == 'yes':
                        encoded_features[c][c2] = self.dropout(encoded_features[c][c2])

                elif c2 == len(view) - 1: # last layer ; not same structure as in init, bc now the last layer is already in view!
                    encoded_features[c].append(self.hidden_layers[c][-1](encoded_features[c][-1]))
                    # no droput layer TODO : user input whether dropout layer


                else : # other layers
                    encoded_features[c].append(self.hidden_layers[c][c2](encoded_features[c][c2-1]))

                    if self.dropout_bool == True and self.dropout_layers[c][c2] == 'yes':
                        encoded_features[c][c2] = self.dropout(encoded_features[c][c2])

        # The data we're interested in is in the middle of the encoding and decoding stage

        data_middle = []

        for c,view in enumerate(encoded_features):
            middle = len(view) // 2
            data_middle.append(view[middle])

        if self.type.lower() == 'concat':

            # Concatenate the output, which will then be passed to a NN for survival analysis
            concatenated_features = torch.cat(tuple(data_middle), dim=-1)


        if self.type.lower() == 'cross':
            pass


        # For training purposes, we will also need the final decoder output of the AE
        # Concatenate everything so that we have the same structure as the input tensors (one tensor stores one whole
        # sample, access features via feature offsets)
        final_out = torch.cat(tuple([dim[-1] for dim in encoded_features]), dim=-1)


        return concatenated_features, final_out





def train(module,views, batch_size =5, n_epochs = 512, lr_scheduler_type = 'onecyclecos'):
    """

    :param module: basically the dataset to be used
    :param batch_size: batch size for training
    :param n_epochs: number of epochs
    :param lr_scheduler_type: type of lr_scheduler : 'lambda','mulitplicative','step','multistep','exponential',...
    """




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



    # Input dimensions (features for each view) for NN based on different data (train/validation/test)
    # Need to be the same for NN to work
    dimensions_train = [x.size(1) for x in train_data]
    dimensions_val = [x.size(1) for x in val_data]
    dimensions_test = [x.size(1) for x in test_data]

    assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/val/test'

    dimensions = dimensions_train

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


    # Call AE
    net = AE(views,dimensions,feature_offsets,[[10,5,2],[10,5,2],[10,5,2],[10,5,2]],
                        [['relu'],['relu','relu','relu'],['relu'],['relu','relu','relu'],['relu']], 0.5,
                        [['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes']],
                        [['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes']],
                        dropout_bool=False,batch_norm_bool=True,type='concat')

    # set optimizer
    optimizer = Adam(net.parameters(), lr=0.01)
    print(net.parameters())
    # set loss function
    criterion = nn.MSELoss() # reconstrution loss
    # data loader for AE
    ae_dataloader = module.train_dataloader(batch_size=batch_size)






    compressed = []
    for epoch in range(n_epochs):
        loss = 0
        for train_data_ae, train_duration_ae, train_event_ae in ae_dataloader:
            #changing data structure of loaded data (needed for pycox)
            train_data_pycox_batch = torch.empty(batch_size, feature_sum_train).to(torch.float32)
            for view in range(len(train_data_ae)):
                train_data_ae[view] = train_data_ae[view].to(device=device)

            train_duration_ae.to(device=device)
            train_event_ae.to(device=device)

            for idx_view,view in enumerate(train_data_ae):
                for idx_sample, sample in enumerate(view):
                    train_data_pycox_batch[idx_sample][feature_offsets_train[idx_view]:
                                                       feature_offsets_train[idx_view+1]] = sample

            optimizer.zero_grad()

            # compressed features is what we are interested in
            compressed_feats, final_out = net(train_data_pycox_batch)
            if epoch == n_epochs - 1: # save compressed_features of last epoch for each batch
                compressed.append(compressed_feats) # list of tensors of compressed for each batch

            train_loss = criterion(final_out, train_data_pycox_batch)

            train_loss.backward()

            optimizer.step()

            loss += train_loss.item()



        loss = loss / len(ae_dataloader)

        print("epoch : {}/{}, loss = {:.6f} ".format(epoch + 1, n_epochs, loss))






if __name__ == '__main__':
    cancer_data = ReadInData.readcancerdata()
    multimodule = DataInputNew.SurvMultiOmicsDataModule(cancer_data[0][0],cancer_data[0][1],cancer_data[0][2])
    views = cancer_data[0][2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(module= multimodule,views= views)






