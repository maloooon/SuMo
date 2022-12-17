import torch
import pandas
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

       # print("hidden layer sizes : {}".format(self.hidden_layer_sizes))


        for size in self.hidden_layer_sizes:
            self.encoder = nn.Sequential(nn.Linear(self.input_dim_per_view[size[0]],size[1]), nn.Sigmoid()) # TODO : ReLU?
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
        print(final_in) # NaN values for second view every time ??

        label_prediction = self.classifier(torch.cat(final_in, dim= -1)) # TODO dim -1 ? Macht das richtige, aber finde nicht viel im Internet dazu, warum

        print(label_prediction)
        return label_prediction



def train(module, batch_size =5, learning_rate = 0.001, n_epochs = 512, output_dim=1):
    """

    :param module: basically the dataset to be used
    :param batch_size: batch size for training
    :return:
    """



    n_train_samples, n_test_samples = module.setup()
    module.feature_selection(method='pca')
    trainloader = module.train_dataloader(batch_size=398) # all training examples

    # load just for size measures
    for data, duration, event in trainloader:
        train_data = data
        for view in range(len(train_data)):
            data[view] = data[view].to(device=device)

        duration = duration.to(device=device)
        event = event.to(device=device)
        survival = (duration,event)



    input_dim_per_view = [x.size(1) for x in train_data]

    # transforming data input for pycox : all views as one tensor, views accessible via feature offset
    feature_offsets = [0] + np.cumsum(input_dim_per_view).tolist()
    # train_data[0].size(0) --> number training samples
    train_data_all = torch.empty(train_data[0].size(0), feature_offsets[-1]).to(torch.float32) # -1 largest index --> gives number of all features together


    # TODO : chck if correct
    for idx_view,view in enumerate(train_data):
        for idx_sample, sample in enumerate(view):
            train_data_all[idx_sample][feature_offsets[idx_view]:feature_offsets[idx_view+1]] = sample






    idx_hidden = [0,1] # mRNA, DNA
    net = NN(idx_hidden=idx_hidden, input_dim_per_view=input_dim_per_view,output_dim=output_dim,feature_offset=feature_offsets)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    callbacks = [tt.callbacks.EarlyStopping()]

    #basically DeepSurv
    model = models.CoxPH(net, optimizer=optimizer)
    model.optimizer.set_lr(learning_rate)

    log = model.fit(train_data_all,survival,batch_size,n_epochs,callbacks, verbose=True) # .fit has problems with multi-omics : cant pass a list of tensors of all views !






if __name__ == '__main__':
    module = DataInputNew.multimodule
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(module= module)





