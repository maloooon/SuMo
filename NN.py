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




class NN(nn.Module):
    """Inspired by SALMON
       https://github.com/huangzhii/SALMON"""
    def __init__(self, idx_hidden, in_features, feature_offset, dropout_rate = .3, output_dim = 1):
        super().__init__()
        self.idx_hidden = idx_hidden # choose which data to send through hidden layer (of type list), e.g. [0,2] for view 0 and 2
        self.dropout_rate = dropout_rate
        self.in_features = in_features # list of ints containing input feature sizes for each view
        self.output_dim = output_dim # size of the output dimension
        self.hidden_layer_sizes = []
        self.encoders = []
        self.feature_offset = feature_offset

        for idx in idx_hidden:
            hidden = round(self.in_features[idx] / 3)   # TODO : Hyperparameter tuning
            self.hidden_layer_sizes.append((idx,hidden))

      #  print("hidden layer sizes : {}".format(self.hidden_layer_sizes))


        for size in self.hidden_layer_sizes:
            self.encoder = nn.Sequential(nn.Linear(self.in_features[size[0]], size[1]), nn.ReLU()) # TODO : ReLU?
            self.encoders.append(self.encoder)


        s_hidden_sum = sum([s_hidden[1] for s_hidden in self.hidden_layer_sizes])

        # sum over all features which are not going through a hidden layer
        self.s_direct_feat = []
        for c,dim in enumerate(in_features):
            if c not in idx_hidden:
                self.s_direct_feat.append((c,dim))

        s_direct_feat_sum = sum([s_direct[1] for s_direct in self.s_direct_feat])



        self.classifier = nn.Sequential(nn.Linear(s_hidden_sum + s_direct_feat_sum,
                                                  self.output_dim),nn.Sigmoid())




    def forward(self, x):
        # x is tensor of size [batch_size, features]

        encoded = []
        # apply hidden layers to all views which have a hidden layer
        # we need to apply hidden layers to only specific parts of the tensor (specific views)
        # create a new tensor (temp) to store only this views feature data which will be send through hidden layer


        for c, encoder in enumerate(self.encoders):
            idx = self.idx_hidden[c]
            # temp tensor of size [batch size, feature size for this view]
            batch_size = x.size(0)
            feature_size = self.feature_offset[idx+1] - self.feature_offset[idx]
            temp = torch.empty(batch_size,feature_size)

            # fill with values
            for i in range(x.size(0)):
                temp[i, :] = x[i][self.feature_offset[idx] : self.feature_offset[idx+1]]



            encoded.append(encoder(temp))



         #   for sample in range(x.size(0)):
         #       encoded.append(encoder(x[sample][self.feature_offset[idx] : self.feature_offset[idx+1]]))



        #    print("layer 1 for views at indices {} : {}".format(self.idx_hidden,encoded))

        non_encoded = []
        # non encoded features that we directly pass to the final layer
        for features in self.s_direct_feat:
            idx = features[0]
            feature_size = self.feature_offset[idx+1] - self.feature_offset[idx]
            temp = torch.empty(batch_size, feature_size)
            # fill with values
            for i in range(x.size(0)):
                temp[i, :] = x[i][self.feature_offset[idx] : self.feature_offset[idx+1]]
            non_encoded.append(temp)




        final_in = tuple(encoded + non_encoded)
        #   print(final_in[0].shape)
        #   print(final_in[1].shape)
        #   print(final_in[2].shape)
        #   print(final_in[3].shape)
      #  print(final_in) # NaN values for second view every time ??

        label_prediction = self.classifier(torch.cat(final_in, dim= -1)) # TODO dim -1 ? Macht das richtige, aber finde nicht viel im Internet dazu, warum

       # print(label_prediction)
        return label_prediction



def train(module, batch_size =50, n_epochs = 512, output_dim=1):
    """

    :param module: basically the dataset to be used
    :param batch_size: batch size for training
    :return:
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
    event_temporary_placeholder_test = (torch.ones(n_test_samples).to(torch.int32)).to(device=device)


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


    # Set indexes for views for SALMON-NN implementation, which are to be passed through their own hidden layer
    idx_hidden = [0,1] # 0 : mRNA, 1:DNA, 2:microRNA, 3:RPPA

    # NN call
 #   net = NN(idx_hidden=idx_hidden,
 #            in_features=dimensions_train,
 #            output_dim=output_dim,
 #            feature_offset=feature_offsets_train)

    net_test = NN_simple(dimensions, feature_offsets)


    # Set parameters for NN

    optimizer = torch.optim.Adam(net_test.parameters())
    callbacks = [tt.callbacks.EarlyStopping()]


    # Call model
    model = models.CoxPH(net_test,optimizer)

    # Set learning rate
    model.optimizer.set_lr(0.01)

    # Fit model
    log = model.fit(train_data_pycox,train_de_pycox,batch_size,n_epochs,callbacks,
                    verbose=True,val_data=val_dde_pycox, val_batch_size=30)

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




if __name__ == '__main__':
    module = DataInputNew.multimodule
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(module= module)





""" 



    for epoch in range(n_epochs):
        for data,duration,event in trainloader_x:


            # Convert data structure for NN (cause PyCox needs data like this)
            current_batch = torch.empty(data[0].size(0), feature_offsets[-1]).to(torch.float32) # -1 largest index --> gives number of all features together

            for idx_view,view in enumerate(data):
                for idx_sample, sample in enumerate(view):
                    current_batch[idx_sample][feature_offsets[idx_view]:feature_offsets[idx_view+1]] = sample

            #forward

            # event tensor size [batch,1], while target is [batch]
            event = torch.squeeze(event)
            hazard_predict = net_test(current_batch)
            loss = criterion(hazard_predict, event)

            #backward
            optimizer.zero_grad()
            loss.backward()

            #ADAM step
            optimizer.step()


"""