import torch
import pandas
import os
import DataInputNew
from torch import nn
from tqdm import tqdm
import numpy as np



class NN(nn.Module):
    """Inspired by SALMON
       https://github.com/huangzhii/SALMON"""
    def __init__(self,idx_hidden,input_dim_per_view,dropout_rate = .3, output_dim = 1):
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

        print("hidden layer sizes : {}".format(self.hidden_layer_sizes))


        for size in self.hidden_layer_sizes:
            self.encoder = nn.Sequential(nn.Linear(self.input_dim_per_view[size[0]],size[1]), nn.Sigmoid())
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


        print("layer 1 for views at indices {} : {}".format(self.idx_hidden,encoded))

        non_encoded = []

        # non encoded features that we directly pass to the final layer
        for features in self.s_direct_feat:
            non_encoded.append(x[features[0]])





        final_in = tuple(encoded + non_encoded)
        print(final_in[0].shape)
        print(final_in[1].shape)
        print(final_in[2].shape)
        print(final_in[3].shape)

        print(final_in)

        label_prediction = self.classifier(torch.cat(final_in, dim= -1)) # TODO dim -1 ? Macht das richtige, aber finde nicht viel im Internet dazu, warum


        return label_prediction




def train(module, batch_size=20, cuda=False, learning_rate=0.01, n_epochs=50):
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

    if cuda:
        model.cuda()
    # weight decay : factor for regularization value in loss function ? aber ADAM ist doch gradient descent method ?
    # usage : reduce complexity and avoid overfitting of model but keep lots of parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)



        # TODO : replace label with event
    for epoch in tqdm(range(n_epochs)):

        for c, data,duration,event in enumerate(trainloader):

            for tensor in range(len(data)):

                data[tensor] = data[tensor].to(torch.float32) # convert to float32 bc of precision problem with float64 in linear layer

            optimizer.zero_grad() # zero gradient buffer

            label_prediction = model(data)

            if c == 0:
                event_prediction_all = label_prediction
                event_all = event
                duration_all = duration
            else:
                event_prediction_all = torch.cat((event_prediction_all, label_prediction), dim=-1)
                event_all = torch.cat((event_all, event), dim=-1)
                duration_all = torch.cat((duration_all, duration), dim=-1)
                # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
                # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data

                # Needed for Cox PH loss
                current_batch_len = len(duration)
                R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
                for i in range(current_batch_len):
                    for j in range(current_batch_len):
                        R_matrix_train[i,j] = duration[j] >= duration[i]

                train_R = torch.FloatTensor(R_matrix_train)
                if cuda:
                    train_R = train_R.cuda()


            # reshape from
            #tensor([[..],
            #        [..]]
            #to tensor([.., ..])
            theta = label_prediction.reshape(-1)

            # Cox Loss
            loss_nn = -torch.mean( (theta - torch.log(torch.sum( torch.exp(theta)*train_R ,dim=1))) * event.float() )

# Regularization in SALMON
#            l1_reg = None
#            for W in model.parameters():
#                if l1_reg is None:
#                    l1_reg = torch.abs(W).sum()
#                else:
#                    l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)














if __name__ == '__main__':
    module = DataInputNew.multimodule
    train(module= module)






