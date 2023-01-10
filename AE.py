import torch
import numpy as np
from torch import nn
import ReadInData
import DataInputNew
from torch.optim import Adam
import statistics
from pycox import models
import torchtuples as tt
import NN
import matplotlib.pyplot as plt
from pycox.evaluation import EvalSurv


class AE_Hierarichal(nn.Module):
    """Wrapper for hierarichal AE implementation : AE followed by another AE followed by NN for survival analysis.
       The second AE may be used without an additional Decoder."""
    def __init__(self,models,types):
        """

        :param models: models to be used (AE, AE, NN)
        :param types: different types of integration methods for the AE implementations (none = take
                      middle layer of each view); input as list
        :param decoding_bool: bool whether decoder should be used for the second AE or not
        """

        super().__init__()
        self.models = models
        self.types = types


    def forward(self, x):
        # First AE
        ae_1 = self.models[0]
        # Second AE
        ae_2 = self.models[1]
        # NN
        nn = self.models[2]


        if self.types[0] == 'none' and self.types[1] == 'concat':
            view_data, final_out_1 = ae_1(x)
            integrated,final_out_2 = ae_2(view_data)
            hazard = nn(integrated)

            return final_out_1, final_out_2, hazard, view_data











class AE_NN(nn.Module):
    """Wrapper so we can train AE & NN together by using Pycox PH model"""
    def __init__(self, models, type):
        super().__init__()
        self.models = models
        self.type = type

    def forward(self, x):
        # AE call
        ae = self.models[0]
        # NN call
        nn = self.models[1]


        if self.type == 'cross':
            element_wise_avg, final_out_cross, final_out = ae(x)
            hazard = nn(element_wise_avg)
            return final_out, final_out_cross, hazard
        else:
            integrated, final_out = ae(x)
            hazard = nn(integrated)
            return final_out, hazard







class AE(nn.Module):
    """AE module, input is each view itself (AE for each view)"""
    def __init__(self, views, in_features, feature_offsets = None, n_hidden_layers_dims = None,
                 activ_funcs = None,dropout_prob = None, dropout_layers= None, batch_norm = None,
                 dropout_bool = False, batch_norm_bool= False, type = None,cross_mutation = None,
                 ae_hierarichcal_bool = False, ae_hierarichcal_decoding_bool = False):
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
        :param cross_mutation : list of integers of length views, deciding which crosses should be applied,
                                e.g [1,3,0,2] will cross hidden feats of view 1 with decoder of view 2,
                                hidden of 4 with dec of view 2 etc..
        :param ae_hierarichcal_bool : choose whether this AE call is to be in the manner of hierarichcal ae implementation
                                      (for the second AE)
        :param ae_hierarichcal_decoding_bool : choose whether in hierarichcal AE setting, the second AE has a decoder
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
        self.cross_mutation = cross_mutation
        self.ae_hierarichcal_bool = ae_hierarichcal_bool
        self.ae_hierarichcal_decoding_bool = ae_hierarichcal_decoding_bool



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

        decoding_hidden = [[] for x in range(len(in_features))]
        decoding_activation = [[] for x in range(len(in_features))]
        decoding_batch = [[] for x in range(len(in_features))]
        decoding_dropout = [[] for x in range(len(in_features))]



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
            # For CrossAE, we'll need decoding informations
            decoding_hidden[c].append(temp_hidden)
            decoding_activation[c].append(temp_activation)
            decoding_batch[c].append(temp_batch)
            decoding_dropout[c].append(temp_dropout)
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


        # mean/max/min output dimension element wise
        mmm_output_dimension = max(self.middle_dims)






        if type.lower() == 'concat':
            # Concatenate the output, which will then be passed to a NN for survival analysis
            # the final output we're interested in (middle between encoder and decoder) was therefore already
            # saved in middle_dims list

            concatenated_features = sum([dim for dim in self.middle_dims])


        if type.lower() == 'cross':
            # For the crossAE implementation, we now need to cross middle hidden features and encoders.
            # for n views there are n! possible mutations (each view has n-1 possible encoder it can take for crossAE)
            # To have the strongest learning effect, the user can choose which mutation he wants in the beginning
            # or it is randomized and then kept for the remainder of all epochs of training.

            # We are only interested in the decoding stages, thus we only look at the hidden layers starting at the
            # middle position
            middle_pos = []
            for x in n_hidden_layers_dims:
                middle_pos.append(len(x)//2)

            # For crossAE implementation, we might need a helping layer to set the dimensions of the hidden feats of view
            # i to the input dim of the first decoding layer of view j
            self.helping_layer = nn.ParameterList(nn.ParameterList([]) for x in range(len(in_features)))
            # boolean list saving whether view needs additional helping layer or not
            self.needs_help_bool = [False for x in range(len(in_features))]

            for c, view in enumerate(n_hidden_layers_dims):
                # if the middle hidden feat dim size of current view is not the same as the hidden feat dim size
                # of view which decoder will be taken to decode the current view hidden dim feats
                if n_hidden_layers_dims[c][middle_pos[c]] != n_hidden_layers_dims[cross_mutation[c]][middle_pos[cross_mutation[c]]]:
                    self.needs_help_bool[c] = True
                    # In this case we need a helping layer
                    # no activation func, no batch norm etc. --> only fit the data so it can be passed on to the chosen decoder
                    self.helping_layer[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][middle_pos[c]],
                                                                         n_hidden_layers_dims[cross_mutation[c]][middle_pos[cross_mutation[c]]])))







        # Dropout
        self.dropout = nn.Dropout(dropout_prob)

            # Print the model
        print("Data input has the following views : {}, each containing {} features.".format(self.views,
                                                                                             self.in_features[0]))

        print("Dropout : {}, Batch Normalization : {}".format(dropout_bool, batch_norm_bool))
        for c,_ in enumerate(self.views):
            print("The view {} has the following pipeline : {}, dropout in layers : {}".format(_, self.hidden_layers[c],dropout_layers[c]))


        if type.lower() == 'none':
            print("The output of each view between encoder and decoder will be passed to a NN for survival analysis"
                  "or another AE before that. Note that the Input Dimensions need to be the same size for each view.")

        if type.lower() == 'concat':
            print("Finally, for ConcatAE, the output of each view between encoder and decoder  is concatenated  ({} features) "
                  "and will be passed to a NN for survival analysis".format(concatenated_features))

        if type.lower() == 'cross':
            for c, _ in enumerate(self.helping_layer):
                print( "For CrossAE implementation we have the following helping layers : {} for view {}"
                   .format(self.helping_layer[c], self.views[c]))
            print("Finally, for CrossAE, the output of each view between encoder and decoder  is averaged element-wise"
                  ", thus {} elements  will be passed to a NN for survival analysis".format(mmm_output_dimension))

        if type.lower() == 'elementwisemean' or type.lower() == 'elementwiseavg':
            print("Finally, for EMeanAE (EAvgAE), the output of each view between encoder and decoder is averaged ({} features)"
                  "and will be passed to a NN for survival analysis".format(mmm_output_dimension))

        if type.lower() == 'overallmean' or type.lower() == 'overallavg':
            print("Finally, for OMeanAE (OAvgAE), the mean of the output of each view between encoder and decoder is calculated"
                  "(1 feature) and then passed to a NN for survival analysis")

        if type.lower() == 'overallmax':
            print("Finally, for OMaxAE, the max of the output of each view between encoder and decoder is calculated"
                  "(1 feature) and then passed to a NN for survival analysis")

        if type.lower() == 'elementwisemax':
            print("Finally, for EMaxAE, the output of each view between encoder and decoder is averaged ({} features)"
                  "and will be passed to a NN for survival analysis".format(mmm_output_dimension))

        if type.lower() == 'overallmin':
            print("Finally, for OMinAE, the min of the output of each view between encoder and decoder is calculated"
                  "(1 feature) and then passed to a NN for survival analysis")

        if type.lower() == 'elementwisemin':
            print("Finally, for EMinAE, the output of each view between encoder and decoder is averaged ({} features)"
                  "and will be passed to a NN for survival analysis".format(mmm_output_dimension))









    def forward(self,x):

        # list of lists to store encoded features for each view
        # Note that encoded features will have BOTH features for encoding stage and decoding stage !
        encoded_features = [[] for x in range(len(self.views))]
        help_features = []
        cross_features = [[] for x in range(len(self.views))]

        #order data by views for diff. hidden layers
        data_ordered = []



        if self.ae_hierarichcal_bool == False:
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
            # Data already processed
            if type(x) is list:
                data_ordered = x
            else:
                data_ordered.append(x)

            batch_size = data_ordered[0].size(0)




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
        middle_pos = []

        for c,view in enumerate(self.n_hidden_layers_dims):
            middle = (len(view) // 2)
            middle_pos.append(middle)
            data_middle.append(encoded_features[c][middle])






        # Concatenate the output, which will then be passed to a NN for survival analysis
        concatenated_features = torch.cat(tuple(data_middle), dim=-1)




        if self.type.lower() == 'overallmean' or self.type.lower() == 'overallavg':
            overall_mean = torch.mean(concatenated_features,1,True)

        if self.type.lower() == 'overallmax':
            overall_max = torch.amax(concatenated_features,1,True)


        if self.type.lower() == 'overallmin':
            overall_min = torch.amin(concatenated_features,1,True)


        # element wise avg needs all the views to have the same dim (so we can take the avg over each value),
        # If they don't, we take the averages where we can and else just use singleton values of views as "avg"
        # find largest middle hidden feat dim
        size = 0
        hidden_sizes = []
        for _ in data_middle:
            # store sizes of each middle hidden feat dim
            hidden_sizes.append(_.size(1))
            if _.size(1) > size:
                size = _.size(1)


        element_wise_avg = torch.empty(batch_size,size)
        element_wise_max = torch.empty(batch_size,size)
        element_wise_min = torch.empty(batch_size,size)



        for x in range(size):

            for i in range(batch_size):
                temp = []
                for _ in data_middle:
                    # element for averaging exists :
                    if _.size(1) - 1 >= x: # -1 due to indexing
                        temp.append(_[i][x])
                if self.type.lower() == 'cross' or self.type.lower() == 'elementwisemean' or self.type.lower() == 'elementwiseavg':
                    mean = torch.mean(torch.stack(temp))
                    element_wise_avg[i][x] = mean
                if self.type.lower() == 'elementwisemax':
                    max = torch.amax(torch.stack(temp))
                    element_wise_max[i][x] = max
                if self.type.lower() == 'elementwisemin':
                    min = torch.amin(torch.stack(temp))
                    element_wise_min[i][x] = min


        if self.type.lower() == 'cross':
            for c,view in enumerate(self.hidden_layers):
                # If we need to apply the helping layer ...
                if self.needs_help_bool[c] == True:
                    # Take data in middle of current view and give to helping layer
                    help_features.append(self.helping_layer[c][0](data_middle[c]))
                else:
                    # Otherwise just pass to help_features (just so we have everything neatly in one list)
                    help_features.append(data_middle[c])

                # Now pass to according decoder
                for c2 in range(len(view) -1):
                    # wait for decoding stage ...
                    if c2 < middle_pos[c]:
                        continue
                    if c2 == middle_pos[c]:
                        # "first" (middle) layer
                        cross_features[c].append(self.hidden_layers[self.cross_mutation[c]][c2 + 1](help_features[c]))
                    else:
                        # all other layers
                        cross_features[c].append(self.hidden_layers[self.cross_mutation[c]][c2 + 1](cross_features[c][-1]))

            final_out_cross = torch.cat(tuple([dim[-1] for dim in cross_features]), dim=-1)









        # For training purposes, we will also need the final decoder output of the AE
        # Concatenate everything so that we have the same structure as the input tensors (one tensor stores one whole
        # sample, access features via feature offsets)
        final_out = torch.cat(tuple([dim[-1] for dim in encoded_features]), dim=-1)


        if self.type.lower() == 'concat':
            # Finally, we will pass the concatenated features to a NN to get the hazard ratio.
            # The final output (final_out) will be used to train the AE.


            return concatenated_features, final_out

        if self.type.lower() == 'cross':
            # element_wise_avg will be passed to a NN to get the hazard ratio
            # final_out_cross will be used to train the AE
            return element_wise_avg, final_out_cross, final_out


        if self.type.lower() == 'none':
            # Change structure of data so it can fit
            return data_middle, final_out

        if self.type.lower() == 'elementwiseavg' or self.type.lower() == 'elementwisemean':

            return element_wise_avg, final_out

        if self.type.lower() == 'overallavg' or self.type.lower() == 'overallmean':

            return overall_mean, final_out

        if self.type.lower() == 'elementwisemax':

            return element_wise_max, final_out

        if self.type.lower() == 'overallmax':

            return overall_max, final_out

        if self.type.lower() == 'elementwisemin':

            return element_wise_min, final_out

        if self.type.lower() == 'overallmin':

            return overall_min, final_out



class LossHierarichcalAE(nn.Module):
    def __init__(self,alpha, decoding_bool = True):
        """

        :param alpha: alpha is a list of 3 values, need to be 1 in sum
        """
        super().__init__()
        assert sum(alpha), 'alpha needs to be 1 in sum'
        self.alpha = alpha
        self.loss_surv = models.loss.CoxPHLoss()
        self.loss_ae = nn.MSELoss()
        self.decoding_bool = decoding_bool


    def forward(self, final_out, final_out_2, hazard, view_data, survival, input_data):
        """

        :param final_out: decoder output AE 1
        :param final_out_2: decoder output AE 2
        :param hazard: hazard
        :param view_data: data between encoder and decoder of AE 1 (input of AE 2)
        :param survival: duration,event
        :param input_data: input_data for AE 1
        :return:
        """
        duration,event = survival
        loss_surv = self.loss_surv(hazard, duration, event)
        loss_ae_1 = self.loss_ae(final_out, input_data)

        # view data is a list of tensors for each view ; as final_out_2 has structure of just one tensor, we
        # need to change structure accordingly
        view_data = torch.cat(tuple(view_data), dim=1)
        if self.decoding_bool == True:
            loss_ae_2 = self.loss_ae(final_out_2, view_data)
            return self.alpha[0] * loss_surv + self.alpha[1] * loss_ae_1 + self.alpha[2] * loss_ae_2
        else:
            # TODO : input vom loss darf hierfür nicht 3geteilt sein, nur 2 Zahlen ! oder wie bei anderen Losses dann nur 1 Zahl alpha ?
            return self.alpha[0] * loss_surv + self.alpha[1] * loss_ae_1





class LossAEConcatHazard(nn.Module):
    def __init__(self,alpha):
        """

        :param alpha:
        """
        super().__init__()
        assert (alpha >= 0) and (alpha <= 1), 'Need alpha in [0,1]'
        self.alpha = alpha
        self.loss_surv = models.loss.CoxPHLoss()
        self.loss_ae = nn.MSELoss()



    """First argument of output needs to be output of the net, second same structure as tuple structure of targets in
       dataset """
    def forward(self, final_out, hazard, survival, input_data):
        """

        :param concatenated_features: Output of middle layer between encoded & decoded (x features from each view for one sample concatenated)
        :param final_out: decoded final output
        :param duration: duration
        :param event : event
        :param input_data: covariate input into AE
        :return: combined loss
        """
        duration,event = survival
        loss_surv = self.loss_surv(hazard, duration,event)
        loss_ae = self.loss_ae(final_out, input_data)
        return self.alpha * loss_surv + (1- self.alpha) * loss_ae



class LossAECrossHazard(nn.Module):
    def __init__(self,alpha):
        """

        :param alpha:
        """
        super().__init__()
        assert (alpha >= 0) and (alpha <= 1), 'Need alpha in [0,1]'
        self.alpha = alpha
        self.loss_surv = models.loss.CoxPHLoss()
        self.loss_ae = nn.MSELoss()


    """First argument of output needs to be output of the net, second same structure as tuple structure of targets in
       dataset """
    def forward(self, final_out, final_out_cross, hazard, survival, input_data):
        """

        :param concatenated_features: Output of middle layer between encoded & decoded (x features from each view for one sample concatenated)
        :param final_out: decoded final output
        :param duration: duration
        :param event : event
        :param input_data: covariate input into AE
        :return: combined loss
        """
        duration,event = survival
        loss_surv = self.loss_surv(hazard, duration,event)
        loss_ae = self.loss_ae(final_out, input_data) + self.loss_ae(final_out_cross, input_data)
        return self.alpha * loss_surv + (1- self.alpha) * loss_ae






def train(module,views, batch_size =25, n_epochs = 512, lr_scheduler_type = 'onecyclecos', l2_regularization = False):
    """

    :param module: basically the dataset to be used
    :param batch_size: batch size for training
    :param n_epochs: number of epochs
    :param lr_scheduler_type: type of lr_scheduler : 'lambda','mulitplicative','step','multistep','exponential',...
    :param l2_regularization : bool whether should be applied or not
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

    # Initialize empty tensors to store the data for train/validation/test
    train_data_pycox = torch.empty(n_train_samples, feature_sum_train).to(torch.float32)
    val_data_pycox = torch.empty(n_val_samples, feature_sum_val).to(torch.float32)
    test_data_pycox = torch.empty(n_test_samples, feature_sum_test).to(torch.float32)

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
    train_duration_numpy = train_duration.detach().cpu().numpy()
    train_event_numpy = train_event.detach().cpu().numpy()
    val_duration_numpy = val_duration.detach().cpu().numpy()
    val_event_numpy = val_event.detach().cpu().numpy()



    train_de_pycox = (train_duration, train_event)
    val_dde_pycox = val_data_pycox, (val_duration, val_event)
    val_de_pycox = (val_duration, val_event)
    test_d_pycox = test_data_pycox

    train_data_pycox_numpy = train_data_pycox.detach().cpu().numpy()
    val_data_pycox_numpy = val_data_pycox.detach().cpu().numpy()
    train_de_pycox_numpy = (train_duration_numpy, train_event_numpy)#event_temporary_placeholder_train_numpy)
    val_de_pycox_numpy = (val_duration_numpy, val_event_numpy)
    train_ded_pycox = tt.tuplefy(train_de_pycox, train_data_pycox) # TODO : Problem hier wegen (221,) und (221,20) als shapes

    full_train = tt.tuplefy(train_data_pycox, (train_de_pycox, train_data_pycox))
    full_validation = tt.tuplefy(val_data_pycox, (val_de_pycox, val_data_pycox))



    all_models = nn.ModuleList()
    all_models.append(AE(views,dimensions,feature_offsets,[[10,5,4],[10,5,4],[10,5,4],[10,5,4]],
                         [['relu'],['relu','relu','relu'],['relu'],['relu','relu','relu'],['relu']], 0.2,
                         [['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes']],
                         [['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes']],
                         dropout_bool=False,batch_norm_bool=True,type='none'))

#    all_models.append(AE(views,dimensions,feature_offsets,[[10,5,2]],
#                            [['relu'],['relu']], 0.5,
#                            [['yes','yes','yes']],
#                            [['yes','yes','yes']],
#                            dropout_bool=False,batch_norm_bool=True,type='none'))

    all_models.append(AE(views,in_features=[4,4,4,4], n_hidden_layers_dims= [[10,5,4],[10,5,4],[10,5,4],[10,5,4]],
                         activ_funcs = [['relu'],['relu','relu','relu'],['relu'],['relu','relu','relu'],['relu']],
                         dropout_prob= 0.2,
                         dropout_layers =[['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes']],
                         batch_norm = [['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes']],
                         dropout_bool=False,batch_norm_bool=True,type='concat', ae_hierarichcal_bool= True))



    # Note : For Concat, the in_feats for NN need to be the sum of all last output layer dimensions.
    #        For Cross, the in_feats for NN need to be the size of the largest output layer dim, as we take the
    #                    element wise avg
    #        For overall(mean/max/min), the in_feats for NN is 1 (as we take the overall average)
    #        For elementwise(mean/max/min), the in_feats for NN must be size of the largest output layer dim

    # TODO : some problem with batch_norm, perhaps : https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
    all_models.append(NN.NN_changeable(views = ['AE'],in_features = [16],
                                       n_hidden_layers_dims= [[8, 4]],
                                       activ_funcs = [['relu'], ['relu']],dropout_prob=0.2,dropout_layers=[['yes','yes']],
                                       batch_norm = [['yes','yes']],
                                       dropout_bool=True,batch_norm_bool=True,ae_bool=True))

  #  all_models.append(NN.NN_changeable(views = views, in_features=[4,4,4,4], n_hidden_layers_dims=[[4,2], [4,2], [4,2], [4,2]],
  #                                     activ_funcs = ['relu'], dropout_prob=0.2, dropout_layers=[['yes','yes'],['yes','yes'],
  #                                                                                               ['yes','yes'],['yes','yes']],
  #                                     batch_norm=[['yes','yes'],['yes','yes'],
  #                                                 ['yes','yes'],['yes','yes']],dropout_bool=True, batch_norm_bool=True,
  #                                    ae_bool=True))

    full_net = AE_Hierarichal(all_models, types=['none','concat'])

  #  full_net = AE_NN(all_models, type='none')


    # set optimizer
    if l2_regularization == True:
        optimizer = Adam(full_net.parameters(), lr=0.01, weight_decay=0.0001)
    else:
        optimizer = Adam(full_net.parameters(), lr=0.01)

    callbacks = [tt.callbacks.EarlyStopping()]

 #   model = models.CoxPH(full_net,optimizer, loss=LossAEConcatHazard(0.6)) # Change Loss here
    model = models.CoxPH(full_net, optimizer, loss=LossHierarichcalAE(alpha= [0.5,0.5], decoding_bool=False))


  #  log = model.fit(*full_train,batch_size,n_epochs,verbose=True, val_data=full_validation, callbacks=callbacks)
    log = model.fit(train_data_pycox,train_ded_pycox,batch_size,n_epochs,verbose=True, val_data=full_validation,
                    callbacks=callbacks)

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

    """
    # Call AE
    net = AE(views,dimensions,feature_offsets,[[10,5,2],[10,5,2],[10,5,2],[10,5,2]],
             [['relu'],['relu','relu','relu'],['relu'],['relu','relu','relu'],['relu']], 0.5,
             [['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes']],
             [['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes']],
             dropout_bool=False,batch_norm_bool=True,type='concat')#,cross_mutation=[1,2,3,0])


    # set optimizer
    optimizer = Adam(net.parameters(), lr=0.001)

    # set loss function
    criterion = nn.MSELoss() # reconstrution loss
    # data loader for AE
    ae_dataloader = module.train_dataloader(batch_size=batch_size)


    model = models.CoxPH(net,optimizer, loss=LossAEConcatHazard(0.6))


   # log = model.fit(*full_train,batch_size,n_epochs,verbose=True,val_data=val_dde_pycox,val_batch_size=5)



    compressed = []
    for epoch in range(n_epochs):
        loss = 0
        loss_cross = 0
        for train_data_ae, train_duration_ae, train_event_ae in ae_dataloader:

        #    if torch.count_nonzero(train_event_ae) == train_event_ae.size(0): # We skip iteration if all event values are 0, because log partial likelihood will drive itself to NaN and we get no loss
        #        continue

            #changing data structure of loaded data (needed for pycox)
            train_data_pycox_batch = torch.empty(batch_size, feature_sum_train).to(torch.float32)
            for view in range(len(train_data_ae)):
                train_data_ae[view] = train_data_ae[view].to(device=device)

            train_duration_ae.to(device=device)
            train_event_ae.to(device=device)

            train_de_pycox = (train_duration_ae, train_event_ae)




            for idx_view,view in enumerate(train_data_ae):
                for idx_sample, sample in enumerate(view):
                    train_data_pycox_batch[idx_sample][feature_offsets_train[idx_view]:
                                                       feature_offsets_train[idx_view+1]] = sample

            optimizer.zero_grad()

            # compressed features is what we are interested in
       #     compressed_feats, final_out, final_out_cross, element_wise_avg = net(train_data_pycox_batch)
            compressed_features, final_out = net(train_data_pycox_batch)
            if epoch == n_epochs - 1: # save compressed_features of last epoch for each batch
                compressed.append(compressed_features) # list of tensors of compressed for each batch









            train_loss = criterion(final_out, train_data_pycox_batch)

         #   train_loss_2 = criterion(final_out_cross, train_data_pycox_batch)


         #   full_loss = train_loss + train_loss_2
         #   full_loss.backward()
            train_loss.backward()
            optimizer.step()

        #    train_loss_2.backward()

        #    optimizer.step()

            loss += train_loss.item()
         #   loss_cross += train_loss_2.item()




        loss = loss / len(ae_dataloader)
     #   loss_cross = loss_cross / len(ae_dataloader)

        print("epoch : {}/{}, loss first stage = {:.6f} ".format(epoch + 1, n_epochs, loss))
     #   print("epoch : {}/{}, loss cross stage = {:.6f} ".format(epoch + 1, n_epochs, loss_cross))




    """

if __name__ == '__main__':
    cancer_data = ReadInData.readcancerdata()
    multimodule = DataInputNew.SurvMultiOmicsDataModule(cancer_data[0][0],cancer_data[0][1],cancer_data[0][2])
    views = cancer_data[0][2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(module= multimodule,views= views, l2_regularization=True)






