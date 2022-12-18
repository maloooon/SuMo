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
    def __init__(self,idx_hidden,input_dim_per_view,feature_offset,dropout_rate = .3, output_dim = 1):
        super().__init__()
        self.idx_hidden = idx_hidden # choose which data to send through hidden layer (of type list), e.g. [0,2] for view 0 and 2
        self.dropout_rate = dropout_rate
        self.input_dim_per_view = input_dim_per_view # list of ints containing input feature sizes for each view
        self.output_dim = output_dim # size of the output dimension
        self.hidden_layer_sizes = []
        self.encoders = []
        self.feature_offset = feature_offset

        for idx in idx_hidden:
            hidden = round(self.input_dim_per_view[idx] / 3)   # TODO : Hyperparameter tuning
            self.hidden_layer_sizes.append((idx,hidden))

      #  print("hidden layer sizes : {}".format(self.hidden_layer_sizes))


        for size in self.hidden_layer_sizes:
            self.encoder = nn.Sequential(nn.Linear(self.input_dim_per_view[size[0]],size[1]), nn.ReLU()) # TODO : ReLU?
            self.encoders.append(self.encoder)


        s_hidden_sum = sum([s_hidden[1] for s_hidden in self.hidden_layer_sizes])

        # sum over all features which are not going through a hidden layer
        self.s_direct_feat = []
        for c,dim in enumerate(input_dim_per_view):
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



    n_train_samples, n_test_samples = module.setup()
    module.feature_selection(method='pca')
    trainloader = module.train_dataloader(batch_size=398) # all training examples
    testloader =module.test_dataloader(batch_size=100)

    # load just for size measures
    for data, duration, event in trainloader:
        train_data = data
        print("train" , train_data[0].shape, train_data[1].shape, train_data[2].shape, train_data[3].shape)
        for view in range(len(train_data)):
            data[view] = data[view].to(device=device)

        duration = duration.to(device=device)
        event = event.to(device=device)

   # for data,duration,event in testloader:
   #     print("test" , data[0].shape, data[1].shape, data[2].shape, data[3].shape) # TODO: wrong sizes ! --> might need to do feature selection on all data and then split afterwards !!
        # TODO : because pca finds other feature sizes for views based on its own data --> gets problematic with feature offsets data input, also in NN


        # We'll use a fake event placeholder, bc current data has too many censored patients
        event_temporary_placeholder = (torch.ones(398).to(torch.int32)).to(device=device)
        duration_structure_train_split = torch.unsqueeze(duration, dim=1)
        event_temporary_placeholder_structure_train_split = torch.unsqueeze(event_temporary_placeholder, dim=1)
        survival_structure_train_split = torch.cat((duration_structure_train_split, event_temporary_placeholder_structure_train_split), dim=1)
        survival = (duration,event_temporary_placeholder)




    input_dim_per_view = [x.size(1) for x in train_data]

    # transforming data input for pycox : all views as one tensor, views accessible via feature offset
    feature_offsets = [0] + np.cumsum(input_dim_per_view).tolist()
    # train_data[0].size(0) --> number training samples
    train_data_all = torch.empty(train_data[0].size(0), feature_offsets[-1]).to(torch.float32) # -1 largest index --> gives number of all features together


    # TODO : chck if correct
    for idx_view,view in enumerate(train_data):
        for idx_sample, sample in enumerate(view):
            train_data_all[idx_sample][feature_offsets[idx_view]:feature_offsets[idx_view+1]] = sample



    # split train set and duration/event values into train and validation
    x_train, x_val, y_train, y_val = train_test_split(train_data_all, survival_structure_train_split, test_size=0.25, random_state=1)


    # turn y into tuple(duration,event), each of type tensor (switch dimensions)
    y_train = y_train.transpose(1,0)
    y_train_f = (y_train[0], y_train[1]) # duration,event
    y_val = y_val.transpose(1,0)
    y_val_f = (y_val[0], y_val[1]) # duration, event
    val = x_val, y_val_f


    idx_hidden = [0,1] # mRNA, DNA
    net = NN(idx_hidden=idx_hidden, input_dim_per_view=input_dim_per_view,output_dim=output_dim,feature_offset=feature_offsets)


    net_test = NN_simple(input_dim_per_view, feature_offsets)



    optimizer = torch.optim.Adam(net_test.parameters())
    callbacks = [tt.callbacks.EarlyStopping()]

    #basically DeepSurv
   # model = models.CoxPH(net, optimizer=optimizer)
    model = models.CoxPH(net_test,optimizer)
    model.optimizer.set_lr(0.01)
    print(val)

    # TODO : val loss doesn't change at all for SALMON NN
    # TODO : train loss for SALMON not significant or gets even higher
    # TODO : NN_simple better already !
    log = model.fit(x_train,y_train_f,batch_size,n_epochs,callbacks, verbose=True,val_data=val, val_batch_size=30) # .fit has problems with multi-omics : cant pass a list of tensors of all views !

 #   _ = model.compute_baseline_hazards() # Cox semi parametric -->  baseline hazard introduced time variable
 #   testloader = module.test_dataloader(batch_size=100)
 #   for data,duration,event in testloader:
 #       test_data = data
 #       test_duration = duration
 #       test_event = event


 #   print(test_data[0].shape)
 #   print(test_duration.shape)
 #   print(test_event)
 #   test_data_all = torch.empty(test_data[0].size(0), feature_offsets[-1]).to(torch.float32) # -1 largest index --> gives number of all features together


    # TODO : chck if correct
 #   for idx_view,view in enumerate(test_data):
 #       for idx_sample, sample in enumerate(view):
 #           test_data_all[idx_sample][feature_offsets[idx_view]:feature_offsets[idx_view+1]] = sample




 #   x_test = torch.cat((test_data_all,test_duration,test_event), dim=1)

 #   x_test_df = pd.DataFrame(x_test.numpy())

 #   surv = model.predict_surv_df(x_test_df)
 #   surv.iloc[:, :5].plot()
 #   plt.ylabel('S(t | x)')
 #   _ = plt.xlabel('Time')


"""

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net_test.parameters())

    trainloader_x = module.train_dataloader(batch_size=5)


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

if __name__ == '__main__':
    module = DataInputNew.multimodule
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(module= module)





