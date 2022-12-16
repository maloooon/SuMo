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



class NN(LightningModule):
    """Inspired by SALMON
       https://github.com/huangzhii/SALMON"""
    def __init__(self,idx_hidden,input_dim_per_view,dropout_rate = .3, output_dim = 1): # TODO : cox model pycox ; output_dim größer
        super().__init__()
        self.idx_hidden = idx_hidden # choose which data to send through hidden layer (of type list), e.g. [0,2] for view 0 and 2
        self.dropout_rate = dropout_rate
        self.input_dim_per_view = input_dim_per_view # list of ints containing input feature sizes for each view
        self.output_dim = output_dim # size of the output dimension
        self.hidden_layer_sizes = []
        self.encoders = []

        for idx in idx_hidden:
            hidden = round(self.input_dim_per_view[idx] / 3)   # TODO : Hyperparameter tuning
            self.hidden_layer_sizes.append((idx,hidden))

     #   print("hidden layer sizes : {}".format(self.hidden_layer_sizes))


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

        encoded = []
        # apply hidden layers to all views which have a hidden layer

        for c, encoder in enumerate(self.encoders):
            encoded.append(encoder(x[self.idx_hidden[c]]))


    #    print("layer 1 for views at indices {} : {}".format(self.idx_hidden,encoded))

        non_encoded = []

        # non encoded features that we directly pass to the final layer
        for features in self.s_direct_feat:
            non_encoded.append(x[features[0]])





        final_in = tuple(encoded + non_encoded)
     #   print(final_in[0].shape)
     #   print(final_in[1].shape)
     #   print(final_in[2].shape)
     #   print(final_in[3].shape)


        label_prediction = self.classifier(torch.cat(final_in, dim= -1)) # TODO dim -1 ? Macht das richtige, aber finde nicht viel im Internet dazu, warum


        return label_prediction



    def training_step(self,batch, batch_idx):
        data, duration, event = batch
        logits = (self.forward(data))
        logits = logits.reshape(logits.size(0))



        loss = F.mse_loss(logits, event) # no loss directly --> we feed into cox model and calculate loss from there
        return {'loss': loss}

    def validation_step(self, val_batch, val_batch_idx):
        pass


    def configure_optimizers(self):
        # weight decay : factor for regularization value in loss function ? aber ADAM ist doch gradient descent method ?
        # usage : reduce complexity and avoid overfitting of model but keep lots of parameters
        return torch.optim.Adam(self.parameters(), lr= 1e-3, weight_decay=0)





def train(module, batch_size=50):
    """

    :param module: basically the dataset to be used
    :param batch_size: batch size for training
    :return:
    """

    n_train_samples, n_test_samples = module.setup()
    module.feature_selection(method='pca')
    trainloader = module.train_dataloader(batch_size=batch_size)

    # load just for size measures
    for data, duration, event in trainloader:
        break

    input_dim_per_view = []

    for view in data:
        input_dim_per_view.append(view.size(1))

    idx_hidden = [0,1] # mRNA, DNA
    model = NN(idx_hidden=idx_hidden, input_dim_per_view=input_dim_per_view)
    trainer = Trainer(max_epochs=20, gpus=1, accelerator='cpu')
    trainer.fit(model, trainloader)




if __name__ == '__main__':
    module = DataInputNew.multimodule
    train(module= module)






