import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from pycox import models
import torchtuples as tt
import NN
import matplotlib.pyplot as plt
from pycox.evaluation import EvalSurv
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
import random
import copy
import optuna
import pandas as pd
import os


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
        # First AE
        self.ae_1 = self.models[0]
        # Second AE
        self.ae_2 = self.models[1]
        # NN
        self.nn = self.models[2]



    def forward(self, *x):




        if self.types[0] == 'cross':
            element_wise_avg, final_out_1, final_out_1_cross, input_data_1 = self.ae_1(*x)
            if self.types[1] == 'cross':
                raise Exception("Cross 2 times does not work, since the first AE gives back an element wise average"
                                " of all the views, thus we can't cross mutate over different views anymore")
            else:
                integrated, final_out_2, input_data_2 = self.ae_2(element_wise_avg)
                hazard = self.nn(integrated)
                return final_out_1, final_out_2, final_out_1_cross, hazard, input_data_2, input_data_1

        else:
            view_data, final_out_1, input_data_1 = self.ae_1(*x)
            if self.types[1] == 'cross':
                if self.types[0] != 'none':
                    raise Exception("Cross on the second AE only works if the output of the first AE still has"
                                    "different views and is not concatenated or averaged/minned/maxxed. Thus only "
                                    "works if we set the type of the first AE to none")

                else:
                    integrated, final_out_2, final_out_2_cross, input_data_2 = self.ae_2(*tuple(view_data))
                    hazard = self.nn(integrated)
                    return final_out_1, final_out_2, final_out_2_cross, hazard, input_data_2, input_data_1


            else:
                if self.types[0] != 'none':
                    integrated, final_out_2, input_data_2 = self.ae_2(view_data)
                    hazard = self.nn(integrated)


                    return final_out_1, final_out_2, hazard, input_data_2, input_data_1

                else:
                    integrated, final_out_2, input_data_2 = self.ae_2(*tuple(view_data))
                    hazard = self.nn(integrated)


                    return final_out_1, final_out_2, hazard, input_data_2 , input_data_1




    def predict(self,*x):
        # Will be used by model.predict later.


        if self.types[0] == 'cross':
            element_wise_avg, final_out_1, final_out_1_cross, input_data_1 = self.ae_1(*x)
            if self.types[1] == 'cross':
                raise Exception("Cross 2 times does not work, since the first AE gives back an element wise average"
                                " of all the views, thus we can't cross mutate over different views anymore")

            else:
                integrated, final_out_2, input_data_2 = self.ae_2(element_wise_avg)
                hazard = self.nn(integrated)

        else:
            view_data, final_out_1, input_data_1 = self.ae_1(*x)
            if self.types[1] == 'cross':
                if self.types[0] != 'none':
                    raise Exception("Cross on the second AE only works if the output of the first AE still has"
                                    "different views and is not concatenated or averaged/minned/maxxed. Thus only "
                                    "works if we set the type of the first AE to none")
                else:
                    integrated, final_out_2, final_out_2_cross, input_data_2 = self.ae_2(*view_data) # *? weil hier noch versch. views
                    hazard = self.nn(integrated)

            else:
                if self.types[0] != 'none':
                    integrated, final_out_2, input_data_2 = self.ae_2(view_data)
                    hazard = self.nn(integrated)
                else:
                    integrated, final_out_2, input_data_2 = self.ae_2(*tuple(view_data)) # ?
                    hazard = self.nn(integrated)


        return hazard



class AE_NN(nn.Module):
    """Wrapper so we can train AE & NN together by using Pycox PH model"""
    def __init__(self, models, type):
        super().__init__()
        self.models = models
        self.type = type
        # AE call
        self.ae = self.models[0]
        # NN call
        self.nn = self.models[1]


    def forward(self, *x):


        if self.type == 'cross':
            element_wise_avg, final_out_cross, final_out, input_data = self.ae(*x)
            hazard = self.nn(element_wise_avg)
            return final_out, final_out_cross, hazard, input_data
        else:
            integrated, final_out, input_data = self.ae(*x)
            hazard = self.nn(integrated)
            return final_out, hazard,input_data


    def predict(self,*x):
        # Will be used by model.predict later.
        if self.type == 'cross':
            element_wise_avg, final_out_cross, final_out, input_data = self.ae(*x)
            hazard = self.nn(element_wise_avg)
            return hazard
        else:
            integrated, final_out, input_data = self.ae(*x)
            hazard = self.nn(integrated)
            return hazard



class AE(nn.Module):
    """AE module, input is each view itself (AE for each view)"""
    def __init__(self, views, in_features, n_hidden_layers_dims = None,
                 activ_funcs = None, dropout_prob = None, dropout_layers= None, batch_norm = None,
                 dropout_bool = False, batch_norm_bool= False, type_ae = None, cross_mutation = None,
                 ae_hierarichcal_bool = False, ae_hierarichcal_decoding_bool = False, print_bool = False):
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
        :param type_ae : concat, cross or both (concross)
        :param cross_mutation : list of integers of length views, deciding which crosses should be applied,
                                e.g [1,3,0,2] will cross hidden feats of view 1 with decoder of view 2,
                                hidden of 4 with dec of view 2 etc..
        :param ae_hierarichcal_bool : choose whether this AE call is to be in the manner of hierarichcal ae implementation
                                      (for the second AE)
        :param ae_hierarichcal_decoding_bool : choose whether in hierarichcal AE setting, the second AE has a decoder
        :param print_bool : choose whether to print the models structure or not
        """
        super().__init__()
        self.views =views
        self.in_features = in_features
        self.n_hidden_layers_dims = n_hidden_layers_dims
        self.activ_funcs = activ_funcs
        self.dropout_prob = dropout_prob
        self.dropout_layers = dropout_layers
        self.batch_norm = batch_norm
        self.dropout_bool = dropout_bool
        self.batch_norm_bool = batch_norm_bool
        self.type_ae = type_ae
        # Create list of lists which will store each hidden layer call for each view (encoding & decoding stage)
        self.hidden_layers = nn.ParameterList([nn.ParameterList([]) for x in range(len(in_features))])
        self.middle_dims = []
        self.cross_mutation = cross_mutation
        self.ae_hierarichcal_bool = ae_hierarichcal_bool
        self.ae_hierarichcal_decoding_bool = ae_hierarichcal_decoding_bool



        # Produce activation functions list of lists


        if len(activ_funcs) == 1 and type(activ_funcs[0]) is not list:

            func = activ_funcs[0]
            activ_funcs = [[func] for x in range(len(views))] # + 1)]


        if len(activ_funcs) == len(views): #  + 1:

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
                    if batch_norm_bool == True:
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






        if type_ae.lower() == 'concat':
            # Concatenate the output, which will then be passed to a NN for survival analysis
            # the final output we're interested in (middle between encoder and decoder) was therefore already
            # saved in middle_dims list

            concatenated_features = sum([dim for dim in self.middle_dims])


        if type_ae.lower() == 'cross':
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






        if print_bool == True:
            # Print the model
            print("Data input has the following views : {}, each containing {} features.".format(self.views,
                                                                                                 self.in_features[0]))

            print("Dropout : {}, Batch Normalization : {}".format(dropout_bool, batch_norm_bool))
            for c,_ in enumerate(self.views):
                print("The view {} has the following pipeline : {}".format(_, self.hidden_layers[c]))
                if dropout_bool == True:
                    print("dropout in layers : {}".format(dropout_layers[c]))


            if type_ae.lower() == 'none':
                print("The output of each view between encoder and decoder will be passed to a NN for survival analysis"
                      "or another AE before that. Note that the Input Dimensions need to be the same size for each view.")

            if type_ae.lower() == 'concat':
                print("Finally, for ConcatAE, the output of each view between encoder and decoder  is concatenated  ({} features) "
                      "and will be passed to a NN for survival analysis".format(concatenated_features))

            if type_ae.lower() == 'cross':
                for c, _ in enumerate(self.helping_layer):
                    print( "For CrossAE implementation we have the following helping layers : {} for view {}"
                       .format(self.helping_layer[c], self.views[c]))
                print("Finally, for CrossAE, the output of each view between encoder and decoder  is averaged element-wise"
                      ", thus {} elements  will be passed to a NN for survival analysis".format(mmm_output_dimension))

            if type_ae.lower() == 'elementwisemean' or type_ae.lower() == 'elementwiseavg':
                print("Finally, for EMeanAE (EAvgAE), the output of each view between encoder and decoder is averaged ({} features)"
                      "and will be passed to a NN for survival analysis".format(mmm_output_dimension))

            if type_ae.lower() == 'overallmean' or type_ae.lower() == 'overallavg':
                print("Finally, for OMeanAE (OAvgAE), the mean of the output of each view between encoder and decoder is calculated"
                      "(1 feature) and then passed to a NN for survival analysis")

            if type_ae.lower() == 'overallmax':
                print("Finally, for OMaxAE, the max of the output of each view between encoder and decoder is calculated"
                      "(1 feature) and then passed to a NN for survival analysis")

            if type_ae.lower() == 'elementwisemax':
                print("Finally, for EMaxAE, the output of each view between encoder and decoder is averaged ({} features)"
                      "and will be passed to a NN for survival analysis".format(mmm_output_dimension))

            if type_ae.lower() == 'overallmin':
                print("Finally, for OMinAE, the min of the output of each view between encoder and decoder is calculated"
                      "(1 feature) and then passed to a NN for survival analysis")

            if type_ae.lower() == 'elementwisemin':
                print("Finally, for EMinAE, the output of each view between encoder and decoder is averaged ({} features)"
                      "and will be passed to a NN for survival analysis".format(mmm_output_dimension))









    def forward(self,*x): # x                                                                                                   # X

        # list of lists to store encoded features for each view
        # Note that encoded features will have BOTH features for encoding stage and decoding stage !
        input_data_raw = x
        encoded_features = [[] for x in range(len(self.views))]
        help_features = []
        cross_features = [[] for x in range(len(self.views))]

        #order data by views for diff. hidden layers
    #    data_ordered = []



    #    if self.ae_hierarichcal_bool == False:
    #        #Get batch size
    #        batch_size = x.size(0)


    #        for view in range(len(self.views)):

                # Calculate the amount of features for current view via offset
    #            feature_size = self.feature_offsets[view+1] - self.feature_offsets[view]

                # Intialize an empty tensor to store data for current view
    #            temp = torch.empty(batch_size,feature_size)

                # fill empty tensor with according features for view for each sample in batch
    #            for i in range(batch_size):
    #                temp[i, :] = x[i][self.feature_offsets[view] : self.feature_offsets[view+1]]

    #            data_ordered.append(temp)
    #    else:
            # Data already processed
    #        if type(x) is list:
        data_ordered = x
    #        else:
    #            data_ordered.append(x)

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




        if self.type_ae.lower() == 'overallmean' or self.type_ae.lower() == 'overallavg':
            overall_mean = torch.mean(concatenated_features,1,True)

        if self.type_ae.lower() == 'overallmax':
            overall_max = torch.amax(concatenated_features,1,True)


        if self.type_ae.lower() == 'overallmin':
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
                if self.type_ae.lower() == 'cross' or self.type_ae.lower() == 'elementwisemean' or self.type_ae.lower() == 'elementwiseavg':
                    mean = torch.mean(torch.stack(temp))
                    element_wise_avg[i][x] = mean
                if self.type_ae.lower() == 'elementwisemax':
                    max = torch.amax(torch.stack(temp))
                    element_wise_max[i][x] = max
                if self.type_ae.lower() == 'elementwisemin':
                    min = torch.amin(torch.stack(temp))
                    element_wise_min[i][x] = min


        if self.type_ae.lower() == 'cross':
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

         #   final_out_cross = torch.cat(tuple([dim[-1] for dim in cross_features]), dim=-1)

            final_out_cross = tuple([dim[-1] for dim in cross_features])                                                           # X









        # For training purposes, we will also need the final decoder output of the AE
        # Concatenate everything so that we have the same structure as the input tensors (one tensor stores one whole
        # sample, access features via feature offsets)
      #  final_out = torch.cat(tuple([dim[-1] for dim in encoded_features]), dim=-1)

        final_out = tuple([dim[-1] for dim in encoded_features])                                                                           # X


        if self.type_ae.lower() == 'concat':
            # Finally, we will pass the concatenated features to a NN to get the hazard ratio.
            # The final output (final_out) will be used to train the AE.
            # x : input_data


            return concatenated_features, final_out, input_data_raw

        if self.type_ae.lower() == 'cross':
            # element_wise_avg will be passed to a NN to get the hazard ratio
            # final_out_cross will be used to train the AE
            return element_wise_avg, final_out_cross, final_out, input_data_raw


        if self.type_ae.lower() == 'none':
            # Change structure of data so it can fit
            return data_middle, final_out, input_data_raw

        if self.type_ae.lower() == 'elementwiseavg' or self.type_ae.lower() == 'elementwisemean':

            return element_wise_avg, final_out, input_data_raw

        if self.type_ae.lower() == 'overallavg' or self.type_ae.lower() == 'overallmean':

            return overall_mean, final_out, input_data_raw

        if self.type_ae.lower() == 'elementwisemax':

            return element_wise_max, final_out, input_data_raw

        if self.type_ae.lower() == 'overallmax':

            return overall_max, final_out, input_data_raw

        if self.type_ae.lower() == 'elementwisemin':

            return element_wise_min, final_out, input_data_raw

        if self.type_ae.lower() == 'overallmin':

            return overall_min, final_out, input_data_raw



class LossHierarichcalAESingleCross(nn.Module):
    def __init__(self,alpha, decoding_bool = True, cross_position = 1):
        """
        :param alpha: alpha is a list of 3 values, need to be 1 in sum
        :param cross_position : integer ; gives info whether first AE uses crossAE or second AE
        """
        super().__init__()
        assert sum(alpha), 'alpha needs to be 1 in sum'
        self.alpha = alpha
        self.loss_surv = models.loss.CoxPHLoss()
        self.loss_ae = nn.MSELoss()
        self.decoding_bool = decoding_bool
        self.cross_position = cross_position


    def forward(self, final_out_1, final_out_2, final_out_cross, hazard, element_wise_avg, input_data_1,duration, event):
        """

        :param final_out: decoder output AE 1
        :param final_out_2: decoder output AE 2
        :param hazard: hazard
        :param view_data: data between encoder and decoder of AE 1 (input of AE 2)
        :param survival: duration,event
        :param input_data: input_data for AE 1
        :return:
        """
        loss_surv = self.loss_surv(hazard, duration, event)
        views_1 = len(final_out_1)
        loss_ae_1_full = 0

        if self.cross_position == 1:
            for i in range(views_1):
                loss_ae_1 = self.loss_ae(final_out_1[i],input_data_1[i])
                loss_ae_cross_1 = self.loss_ae(final_out_cross[i], input_data_1[i])
                loss_ae_1_full += (loss_ae_1)
                loss_ae_1_full += (loss_ae_cross_1)

        else:
            for i in range(views_1):
                loss_ae_1 = self.loss_ae(final_out_1[i],input_data_1[i])
                loss_ae_1_full += (loss_ae_1)


     #   loss_ae_1_full = sum(loss_ae_1_full)

        # check if we have list of tensors for each view or tensor as output of first AE
        # as final_out_2 has structure of just one tensor, we
        # need to change structure accordingly


      #  if type(element_wise_avg) is list:
      #      view_data = torch.cat(tuple(element_wise_avg), dim=1)
      #  else:
      #      view_data = element_wise_avg
        view_data = element_wise_avg



        if self.decoding_bool == True:



            if self.cross_position == 2:
                # Check not needed, because cross on second AE will only work if first AE was of type none and thus
                # data still has different view structure
                loss_ae_2_full = 0
                for i in range(len(view_data)):
                    loss_ae_2 = self.loss_ae(final_out_2[i], view_data[i])
                    loss_ae_cross_2 = self.loss_ae(final_out_cross[i], view_data[i])
                    loss_ae_2_full += (loss_ae_2)
                    loss_ae_2_full += (loss_ae_cross_2)
              #  loss_ae_2_full = sum(loss_ae_2_full)
            else:
                # Here we also need no check because if we didnt have a cross in the second AE, we surely had one
                # in the first, thus our data is not of multiple view structure anymore
                loss_ae_2_full = self.loss_ae(final_out_2[0], view_data[0])

            return self.alpha[0] * loss_surv + self.alpha[1] * loss_ae_1_full + self.alpha[2] * loss_ae_2_full
        else:
            if len(self.alpha) != 2:
                raise Exception("Since the second AE has no decoder, alpha only contains 2 elements, not 3!")
            else:
                return self.alpha[0] * loss_surv + self.alpha[1] * loss_ae_1_full


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


    def forward(self, final_out_1, final_out_2, hazard, view_data, input_data_1,duration, event):
        """

        :param final_out: decoder output AE 1
        :param final_out_2: decoder output AE 2
        :param hazard: hazard
        :param view_data: data between encoder and decoder of AE 1 (input of AE 2)
        :param survival: duration,event
        :param input_data: input_data for AE 1
        :return:
        """
        loss_surv = self.loss_surv(hazard, duration, event)
        # final_out_1 will always have multiple view structure as it is the decoder of the first AE, which takes
        # multiple views as input // TODO : check for one view ob richtig übergeben wird oder len(...) dann falsch liest
        views_1 = len(final_out_1)
        loss_ae_1_full = 0
        # AE loss for each view
        for i in range(views_1):
            loss_ae_1 = self.loss_ae(final_out_1[i], input_data_1[i])
            loss_ae_1_full += loss_ae_1

       # loss_ae_1 = self.loss_ae(final_out_1, input_data_1)

        # view data is a list of tensors for each view ; as final_out_2 has structure of just one tensor, we
        # need to change structure accordingly
       # view_data = torch.cat(tuple(view_data), dim=1)
        if self.decoding_bool == True:
            loss_ae_2_full = 0
            # Second AE might not have multiple views as input, so we need to check that first
            # Diff view structure
            if len(view_data) > 1:
                views_2 = len(view_data)
                for i in range(views_2):
                    loss_ae_2 = self.loss_ae(final_out_2[i], view_data[i])
                    loss_ae_2_full += loss_ae_2
            # No diff view structrue
            else:
                # final_out_2[0] as it is a tuple tree
                loss_ae_2_full = self.loss_ae(final_out_2[0], view_data[0])
#            loss_ae_2 = self.loss_ae(final_out_2, view_data)
            return self.alpha[0] * loss_surv + self.alpha[1] * loss_ae_1_full + self.alpha[2] * loss_ae_2_full
        else:
            # TODO : input vom loss darf hierfür nicht 3geteilt sein, nur 2 Zahlen
            if len(self.alpha) != 2:
                raise Exception("Since the second AE has no decoder, alpha only contains 2 elements, not 3!")
            else:
                return self.alpha[0] * loss_surv + self.alpha[1] * loss_ae_1_full


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
    def forward(self, final_out, hazard,input_data, duration,event):
        """

        :param concatenated_features: Output of middle layer between encoded & decoded (x features from each view for one sample concatenated)
        :param final_out: decoded final output
        :param duration: duration
        :param event : event
        :param input_data: covariate input into AE
        :return: combined loss
        """
      #  duration,event = survival
        loss_surv = self.loss_surv(hazard, duration,event)
        views = len(final_out)
        loss_ae_full = 0
        # AE loss for each view
        for i in range(views):
            loss_ae = self.loss_ae(final_out[i], input_data[i])
            loss_ae_full += loss_ae
        return self.alpha * loss_surv + (1- self.alpha) * loss_ae_full



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
    def forward(self, final_out, final_out_cross, hazard, input_data, duration, event):
        """

        :param concatenated_features: Output of middle layer between encoded & decoded (x features from each view for one sample concatenated)
        :param final_out: decoded final output
        :param duration: duration
        :param event : event
        :param input_data: covariate input into AE
        :return: combined loss
        """
        loss_surv = self.loss_surv(hazard, duration,event)
        views = len(final_out)
        loss_ae_full = 0
        loss_ae_cross_full = 0
        # AE loss for each view
        for i in range(views):
            loss_ae = self.loss_ae(final_out[i], input_data[i])
            loss_ae_cross = self.loss_ae(final_out_cross[i], input_data[i])
            loss_ae_full += loss_ae
            loss_ae_cross_full += loss_ae_cross


        loss_ae_all = loss_ae_full + loss_ae_cross_full
        #loss_ae = self.loss_ae(final_out, input_data) + self.loss_ae(final_out_cross, input_data)
        return self.alpha * loss_surv + (1- self.alpha) * loss_ae_all




def objective(trial):


    # Load in data (##### For testing for first fold, later on

    trainset, trainset_feat, valset, valset_feat, testset, testset_feat = load_data()

    trainset_feat = list(trainset_feat.values)
    for idx,_ in enumerate(trainset_feat):
        trainset_feat[idx] = trainset_feat[idx].item()

    valset_feat = list(valset_feat.values)
    for idx,_ in enumerate(valset_feat):
        valset_feat[idx] = valset_feat[idx].item()

    testset_feat = list(testset_feat.values)
    for idx,_ in enumerate(testset_feat):
        testset_feat[idx] = testset_feat[idx].item()


    train_data = []
    for c,feat in enumerate(trainset_feat):
        if c < len(trainset_feat) - 3: # train data views
            train_data.append(np.array((trainset.iloc[:, trainset_feat[c] : trainset_feat[c+1]]).values).astype('float32'))
        elif c == len(trainset_feat) - 3: # duration
            train_duration = (np.array((trainset.iloc[:, trainset_feat[c] : trainset_feat[c+1]]).values).astype('float32')).squeeze(axis=1)
        elif c == len(trainset_feat) -2: # event
            train_event = (np.array((trainset.iloc[:, trainset_feat[c] : trainset_feat[c+1]]).values).astype('float32')).squeeze(axis=1)

    train_data = tuple(train_data)

    val_data = []
    for c,feat in enumerate(valset_feat):
        if c < len(valset_feat) - 3: # train data views
            val_data.append(np.array((valset.iloc[:, valset_feat[c]: valset_feat[c + 1]]).values).astype('float32'))
        elif c == len(valset_feat) - 3: # duration
            val_duration = (np.array((valset.iloc[:, valset_feat[c]: valset_feat[c + 1]]).values).astype('float32')).squeeze(axis=1)
        elif c == len(valset_feat) -2: # event
            val_event = (np.array((valset.iloc[:, valset_feat[c]: valset_feat[c + 1]]).values).astype('float32')).squeeze(axis=1)

    test_data = []

    for c,feat in enumerate(testset_feat):
        if c < len(testset_feat) - 3: # train data views
            test_data.append(np.array((testset.iloc[:, testset_feat[c]: testset_feat[c + 1]]).values).astype('float32'))
        elif c == len(testset_feat) - 3: # duration
            test_duration = (np.array((testset.iloc[:, testset_feat[c]: testset_feat[c + 1]]).values).astype('float32')).squeeze(axis=1)
        elif c == len(testset_feat) -2: # event
            test_event = (np.array((testset.iloc[:, testset_feat[c]: testset_feat[c + 1]]).values).astype('float32')).squeeze(axis=1)



    views = []
    read_in = open('/Users/marlon/Desktop/Project/TCGAData/cancerviews.txt', 'r')
    for view in read_in:
        views.append(view)

    view_names = [line[:-1] for line in views]


    dimensions_train = [x.shape[1] for x in train_data]
    dimensions_val = [x.shape[1] for x in val_data]
    dimensions_test = [x.shape[1] for x in test_data]

    assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/test'

    dimensions = dimensions_train

    # Transforms for PyCox
    train_surv = (train_duration, train_event)
    val_data_full = (val_data, (val_duration, val_event))


    ##################################### HYPERPARAMETER SEARCH SETTINGS ##############################################
    l2_regularization_bool = trial.suggest_categorical('l2_regularization_bool', [True,False])
    learning_rate = trial.suggest_float("learning_rate", 1e-5,1e-1,log=True)
    l2_regularization_rate = trial.suggest_float("l2_regularization_rate", 1e-6,1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 5, 200) # TODO : batch size so wählen, dass train samples/ batch_size und val samples/batch_size nie 1 ergeben können, da sonst Error : noch besser error abfangen und einfach skippen, da selten passiert !
    n_epochs = trial.suggest_int("n_epochs", 10,100)
    dropout_prob = trial.suggest_float("dropout_prob", 0,0.5,step=0.1)
    dropout_bool = trial.suggest_categorical('dropout_bool', [True,False])
    batchnorm_bool = trial.suggest_categorical('batchnorm_bool',[True,False])

    layers_1_mRNA = trial.suggest_int('layers_1_mRNA', 5, 1200)
    layers_2_mRNA = trial.suggest_int('layers_2_mRNA', 5, 1200)
    layers_1_DNA = trial.suggest_int('layers_1_DNA', 5, 1200)
    layers_2_DNA = trial.suggest_int('layers_2_DNA', 5, 1200)
    layers_1_microRNA = trial.suggest_int('layers_1_microRNA', 5, 1200)
    layers_2_microRNA = trial.suggest_int('layers_1_microRNA', 5, 1200)
    layers_1_RPPA = trial.suggest_int('layers_1_microRNA', 5, 1200)
    layers_2_RPPA = trial.suggest_int('layers_1_microRNA', 5, 1200)


    # After ConcatAE FCNN
    layers_1_FCNN = trial.suggest_int('layers_1_FCNN', 5, 1200)
    layers_2_FCNN = trial.suggest_int('layers_2_FCNN', 5, 1200)

    layers_FCNN = [[layers_1_FCNN,layers_2_FCNN]]

    layers_1_FCNN_activfunc = trial.suggest_categorical('layers_1_FCNN_activfunc', ['relu','sigmoid'])
    layers_2_FCNN_activfunc = trial.suggest_categorical('layers_2_FCNN_activfunc', ['relu','sigmoid'])

    FCNN_activation_functions = [layers_1_FCNN_activfunc, layers_2_FCNN_activfunc]

    FCNN_dropout_prob = trial.suggest_float("FCNN_dropout_prob", 0,0.5,step=0.1)
    FCNN_dropout_bool = trial.suggest_categorical('FCNN_dropout_bool', [True,False])
    FCNN_batchnorm_bool = trial.suggest_categorical('FCNN_batchnorm_bool',[True,False])

    layers_1_FCNN_dropout = trial.suggest_categorical('layers_1_FCNN_dropout', ['yes','no'])
    layers_2_FCNN_dropout = trial.suggest_categorical('layers_2_FCNN_dropout', ['yes','no'])

    FCNN_dropouts = [layers_1_FCNN_dropout, layers_2_FCNN_dropout]


    layers_1_FCNN_batchnorm = trial.suggest_categorical('layers_1_FCNN_batchnorm', ['yes', 'no'])
    layers_2_FCNN_batchnorm = trial.suggest_categorical('layers_2_FCNN_batchnorm', ['yes', 'no'])

    FCNN_batchnorms = [layers_1_FCNN_batchnorm, layers_2_FCNN_batchnorm]

    # Survival + MSE-Loss
    loss_2_values = trial.suggest_float("loss_2_values", 0,1)




    layers_1_mRNA_activfunc = trial.suggest_categorical('layers_1_mRNA_activfunc', ['relu','sigmoid'])
    layers_2_mRNA_activfunc = trial.suggest_categorical('layers_2_mRNA_activfunc', ['relu','sigmoid'])
    layers_1_DNA_activfunc = trial.suggest_categorical('layers_1_DNA_activfunc', ['relu','sigmoid'])
    layers_2_DNA_activfunc = trial.suggest_categorical('layers_2_DNA_activfunc', ['relu','sigmoid'])
    layers_1_microRNA_activfunc = trial.suggest_categorical('layers_1_microRNA_activfunc', ['relu','sigmoid'])
    layers_2_microRNA_activfunc = trial.suggest_categorical('layers_2_microRNA_activfunc', ['relu','sigmoid'])
    layers_1_RPPA_activfunc = trial.suggest_categorical('layers_1_RPPA_activfunc', ['relu','sigmoid'])
    layers_2_RPPA_activfunc = trial.suggest_categorical('layers_2_RPPA_activfunc', ['relu','sigmoid'])

    layers_1_mRNA_dropout = trial.suggest_categorical('layers_1_mRNA_dropout', ['yes','no'])
    layers_2_mRNA_dropout = trial.suggest_categorical('layers_2_mRNA_dropout', ['yes','no'])
    layers_1_DNA_dropout = trial.suggest_categorical('layers_1_DNA_dropout', ['yes','no'])
    layers_2_DNA_dropout = trial.suggest_categorical('layers_2_DNA_dropout', ['yes','no'])
    layers_1_microRNA_dropout = trial.suggest_categorical('layers_1_microRNA_dropout', ['yes','no'])
    layers_2_microRNA_dropout = trial.suggest_categorical('layers_2_microRNA_dropout', ['yes','no'])
    layers_1_RPPA_dropout = trial.suggest_categorical('layers_1_RPPA_dropout', ['yes','no'])
    layers_2_RPPA_dropout = trial.suggest_categorical('layers_2_RPPA_dropout', ['yes','no'])

    layers_1_mRNA_batchnorm = trial.suggest_categorical('layers_1_mRNA_batchnorm', ['yes', 'no'])
    layers_2_mRNA_batchnorm = trial.suggest_categorical('layers_2_mRNA_batchnorm', ['yes', 'no'])

    layers_1_DNA_batchnorm = trial.suggest_categorical('layers_1_DNA_batchnorm', ['yes', 'no'])
    layers_2_DNA_batchnorm = trial.suggest_categorical('layers_2_DNA_batchnorm', ['yes', 'no'])

    layers_1_microRNA_batchnorm = trial.suggest_categorical('layers_1_microRNA_batchnorm', ['yes', 'no'])
    layers_2_microRNA_batchnorm = trial.suggest_categorical('layers_2_microRNA_batchnorm', ['yes', 'no'])

    layers_1_RPPA_batchnorm = trial.suggest_categorical('layers_1_RPPA_batchnorm', ['yes', 'no'])
    layers_2_RPPA_batchnorm = trial.suggest_categorical('layers_2_RPPA_batchnorm', ['yes', 'no'])


    # Loss 3 values TODO : HierarichcalAE has 2 MSE losses and partial negative log likelihood, how to select 3 values that sum up to 1 with optuna?

    # TODO : cross-mutation optuna search only works when we can assert that all views have same feature sizes


    layers = []
    activation_functions = []
    dropouts = []
    batchnorms = []

    if 'MRNA' in view_names:
        layers.append([layers_1_mRNA,layers_2_mRNA])
        activation_functions.append([layers_1_mRNA_activfunc, layers_2_mRNA_activfunc])
        dropouts.append([layers_1_mRNA_dropout, layers_2_mRNA_dropout])
        batchnorms.append([layers_1_mRNA_batchnorm, layers_2_mRNA_batchnorm])

    if 'DNA' in view_names:
        layers.append([layers_1_DNA,layers_2_DNA])
        activation_functions.append([layers_1_DNA_activfunc, layers_2_DNA_activfunc])
        dropouts.append([layers_1_DNA_dropout, layers_2_DNA_dropout])
        batchnorms.append([layers_1_DNA_batchnorm, layers_2_DNA_batchnorm])

    if 'MICRORNA' in view_names:
        layers.append([layers_1_microRNA,layers_2_microRNA])
        activation_functions.append([layers_1_microRNA_activfunc, layers_2_microRNA_activfunc])
        dropouts.append([layers_1_microRNA_dropout, layers_2_microRNA_dropout])
        batchnorms.append([layers_1_microRNA_batchnorm, layers_2_microRNA_batchnorm])

    if 'RPPA' in view_names:
        layers.append([layers_1_RPPA,layers_2_RPPA])
        activation_functions.append([layers_1_RPPA_activfunc, layers_2_RPPA_activfunc])
        dropouts.append([layers_1_RPPA_dropout, layers_2_RPPA_dropout])
        batchnorms.append([layers_1_RPPA_batchnorm, layers_2_RPPA_batchnorm])


    # Create List of models to be used
    all_models = nn.ModuleList()


    model_types = ['concat','elementwisemax']

  #  print("MODEL TYPES : ", model_types)

    out_sizes = []
    for c_layer in range(len(layers)):
        out_sizes.append(layers[c_layer][-1])

    in_feats_second_NN_concat = sum(out_sizes)


    # AE's


    all_models.append(AE(views = view_names,
                         in_features= dimensions,
                         n_hidden_layers_dims= layers,
                         activ_funcs=activation_functions,
                         dropout_bool= dropout_bool,
                         dropout_prob= dropout_prob,
                         dropout_layers= dropouts,
                         batch_norm_bool= batchnorm_bool,
                         batch_norm= batchnorms,
                         type_ae=model_types[0],
                         cross_mutation=[1,0],
                         print_bool=False))


    all_models.append(NN.NN_changeable(views = ['AE'],
                                       trial= trial,
                                       in_features = [in_feats_second_NN_concat],
                                       n_hidden_layers_dims= layers_FCNN,
                                       activ_funcs = [FCNN_activation_functions,['none']],
                                       dropout_prob=FCNN_dropout_prob,
                                       dropout_layers=FCNN_dropouts,
                                       batch_norm = FCNN_batchnorms,
                                       dropout_bool=FCNN_dropout_bool,
                                       batch_norm_bool=FCNN_batchnorm_bool,
                                       print_bool=False))




    #    full_net = AE_Hierarichal(all_models, types=model_types)

    full_net = AE_NN(all_models, type=model_types[0])


    # set optimizer
    if l2_regularization_bool == True:
        optimizer = Adam(full_net.parameters(), lr=learning_rate, weight_decay=l2_regularization_rate)
    else:
        optimizer = Adam(full_net.parameters(), lr=learning_rate)

    callbacks = [tt.callbacks.EarlyStopping(patience=10)]
    #   cross_pos = model_types.index("cross") + 1
    #   model = models.CoxPH(full_net,optimizer, loss=LossHierarichcalAESingleCross(alpha=[0.4,0.4,0.2], decoding_bool=True, cross_position=cross_pos)) # Change Loss here


    # loss : alpha * surv_loss + (1-alpha) * ae_loss
    model = models.CoxPH(full_net,
                         optimizer,
                         loss=LossAEConcatHazard(loss_2_values))
    print_loss = False

    log = model.fit(train_data,
                    train_surv,
                    batch_size,
                    n_epochs,
                    verbose=print_loss,
                    val_data= val_data_full,
                    val_batch_size= batch_size,
                    callbacks=callbacks)

    # Plot it
    #      _ = log.plot()

    # Since Cox semi parametric, we calculate a baseline hazard to introduce a time variable
    _ = model.compute_baseline_hazards()

    # Predict based on test data
    surv = model.predict_surv_df(test_data)

    # Plot it
    #      surv.iloc[:, :5].plot()
    #      plt.ylabel('S(t | x)')
    #      _ = plt.xlabel('Time')


    # Evaluate with concordance, brier score and binomial log-likelihood
    ev = EvalSurv(surv, test_duration, test_event, censor_surv='km') # censor_surv : Kaplan-Meier

    # concordance
    concordance_index = ev.concordance_td()

    if concordance_index < 0.5:
        concordance_index = 1 - concordance_index

    #brier score
    time_grid = np.linspace(test_duration.min(), test_duration.max(), 100)
    _ = ev.brier_score(time_grid).plot
    brier_score = ev.integrated_brier_score(time_grid)

    #binomial log-likelihood
    binomial_score = ev.integrated_nbll(time_grid)


    return concordance_index









def optuna_optimization(fold = 1):


    EPOCHS = 150
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials = EPOCHS)

    trial = study.best_trial

    print("Best Concordance", trial.value)
    print("Best Hyperparamters : {}".format(trial.params))





def train(train_data,
          val_data,
          test_data,
          train_duration,
          train_event,
          val_duration,
          val_event,
          test_duration,
          test_event,
          learning_rate = 0.0001,
          batch_size =128,
          n_epochs = 512,
          l2_regularization = False,
          l2_regularization_rate = 0.000001,
          batchnorm = False,
          batchnorm_layers = None,
          dropout_layers = None,
          val_batch_size = 16,
          dropout_rate = 0.1,
          dropout = False,
          activation_layers = None,
          view_names = None,
          layers = None):
    """

    :param module: basically the dataset to be used
    :param batch_size: batch size for training
    :param n_epochs: number of epochs
    :param lr_scheduler_type: type of lr_scheduler : 'lambda','mulitplicative','step','multistep','exponential',...
    :param l2_regularization : bool whether should be applied or not
    """





    # Cast to numpy arrays if necessary(if we get an error, we already have numpy arrays --> no need to cast)
    try:
        test_duration = test_duration.numpy()
        test_event = test_event.numpy()
    except AttributeError:
        pass


    for c,fold in enumerate(train_data):
        try:
            train_duration[c] = train_duration[c].numpy()
            train_event[c] = train_event[c].numpy()
            val_duration[c] = val_duration[c].numpy()
            val_event[c] = val_event[c].numpy()
        except AttributeError: # in this case already numpy arrays
            pass



        for c2,view in enumerate(fold):
            try:
                train_data[c][c2] = (train_data[c][c2]).numpy()
                val_data[c][c2] = (val_data[c][c2]).numpy()
                test_data[c][c2] = (test_data[c][c2]).numpy()
            except AttributeError:
                pass



        # Need tuple structure for PyCox
        train_data[c] = tuple(train_data[c])
        val_data[c] = tuple(val_data[c])
        test_data[c] = tuple(test_data[c])



    ############################# FOLD X ###################################
    for c_fold,fold in enumerate(train_data):
        for c2,view in enumerate(fold):
            print("Train data has shape : {} for view {}".format(train_data[c_fold][c2].shape, view_names[c2]))
            print("Validation data has shape : {} for view {}".format(val_data[c_fold][c2].shape, view_names[c2]))
            print("Test data has shape : {} for view {}".format(test_data[c_fold][c2].shape, view_names[c2]))


        dimensions_train = [x.shape[1] for x in train_data[c_fold]]
        dimensions_val = [x.shape[1] for x in val_data[c_fold]]
        dimensions_test = [x.shape[1] for x in test_data[c_fold]]

        assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/test'

        dimensions = dimensions_train


        # Transforms for PyCox
        train_surv = (train_duration[c_fold], train_event[c_fold])
        val_data_full = (val_data[c_fold], (val_duration[c_fold], val_event[c_fold]))


        # Create List of models to be used
        all_models = nn.ModuleList()


        model_types = ['concat','elementwisemax']

        print("MODEL TYPES : ", model_types)

        curr_concordance = 0

        out_sizes = []
        for c_layer in range(len(layers)):
            out_sizes.append(layers[c_layer][-1])

        in_feats_second_NN_concat = sum(out_sizes)


        # AE's
        layers_u = copy.deepcopy(layers)
        activation_layers_u = copy.deepcopy(activation_layers)
        dropout_layers_u = copy.deepcopy(dropout_layers)
        batchnorm_layers_u = copy.deepcopy(batchnorm_layers)
        all_models.append(AE(views = view_names,
                             in_features= dimensions,
                             n_hidden_layers_dims= layers_u,
                             activ_funcs=activation_layers_u,
                             dropout_bool= dropout,
                             dropout_prob= dropout_rate,
                             dropout_layers= dropout_layers_u,
                             batch_norm_bool= batchnorm,
                             batch_norm= batchnorm_layers_u,
                             type_ae=model_types[0],
                             cross_mutation=[1,0],
                             print_bool=False))


        #    all_models.append(AE(views = ['AE'],in_features=[4], n_hidden_layers_dims= [[10,5,4]],
        #                         activ_funcs = [['relu']],
        #                         dropout_prob= 0.2,
        #                         dropout_layers =[['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes']],
        #                         batch_norm = [['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes']],
        #                         dropout_bool=False,batch_norm_bool=False,type=model_types[1], cross_mutation=[1,0,3,2], ae_hierarichcal_bool= True,print_bool=False))


        # NN

        # Note : For Concat, the in_feats for NN need to be the sum of all last output layer dimensions.
        #        For Cross, the in_feats for NN need to be the size of the largest output layer dim, as we take the
        #                    element wise avg
        #        For overall(mean/max/min), the in_feats for NN is 1 (as we take the overall average)
        #        For elementwise(mean/max/min), the in_feats for NN must be size of the largest output layer dim


        dropout_layers_u = copy.deepcopy(dropout_layers)
        batchnorm_layers_u = copy.deepcopy(batchnorm_layers)
        all_models.append(NN.NN_changeable(views = ['AE'],
                                           trial= None,
                                           in_features = [in_feats_second_NN_concat],
                                           n_hidden_layers_dims= [[64,32]],
                                           activ_funcs = [['relu'],['none']],
                                           dropout_prob=dropout_rate,
                                           dropout_layers=[dropout_layers_u[0]],
                                           batch_norm = [batchnorm_layers_u[0]],
                                           dropout_bool=dropout,
                                           batch_norm_bool=batchnorm,
                                           print_bool=False))




        #############
        # Note that for the Cross Method, the cross_mutation must be set in a way so that the input view x and
        # "crossed" view y have the same feature size : Otherwise, the MSELoss() cant be calculated
        # e.g. 4 views having feature sizes [15,15,5,5] --> cross mutations between first two and last two work,
        # but not e.g. between first and third/fourth





        #    full_net = AE_Hierarichal(all_models, types=model_types)

        full_net = AE_NN(all_models, type=model_types[0])


        # set optimizer
        if l2_regularization == True:
            optimizer = Adam(full_net.parameters(), lr=learning_rate, weight_decay=l2_regularization_rate)
        else:
            optimizer = Adam(full_net.parameters(), lr=learning_rate)

        callbacks = [tt.callbacks.EarlyStopping(patience=10)]
        #   cross_pos = model_types.index("cross") + 1
        #   model = models.CoxPH(full_net,optimizer, loss=LossHierarichcalAESingleCross(alpha=[0.4,0.4,0.2], decoding_bool=True, cross_position=cross_pos)) # Change Loss here


        # loss : alpha * surv_loss + (1-alpha) * ae_loss
        model = models.CoxPH(full_net,
                             optimizer,
                             loss=LossAEConcatHazard(0.6))
        print_loss = False
        print("Split {} : ".format(c_fold + 1))
        log = model.fit(train_data[c_fold],
                        train_surv,
                        batch_size,
                        n_epochs,
                        verbose=print_loss,
                        val_data= val_data_full,
                        val_batch_size= val_batch_size,
                        callbacks=callbacks)

        # Plot it
  #      _ = log.plot()

        # Since Cox semi parametric, we calculate a baseline hazard to introduce a time variable
        _ = model.compute_baseline_hazards()

        # Predict based on test data
        surv = model.predict_surv_df(test_data[c_fold])

        # Plot it
  #      surv.iloc[:, :5].plot()
  #      plt.ylabel('S(t | x)')
  #      _ = plt.xlabel('Time')


        # Evaluate with concordance, brier score and binomial log-likelihood
        ev = EvalSurv(surv, test_duration, test_event, censor_surv='km') # censor_surv : Kaplan-Meier

        # concordance
        concordance_index = ev.concordance_td()

        if concordance_index < 0.5:
            concordance_index = 1 - concordance_index

        #brier score
        time_grid = np.linspace(test_duration.min(), test_duration.max(), 100)
        _ = ev.brier_score(time_grid).plot
        brier_score = ev.integrated_brier_score(time_grid)

        #binomial log-likelihood
        binomial_score = ev.integrated_nbll(time_grid)

        print("Concordance index : {} , Integrated Brier Score : {} , Binomial Log-Likelihood : {}".format(concordance_index,
                                                                                                           brier_score,
                                                                                                           binomial_score))






def load_data(data_dir="/Users/marlon/Desktop/Project/PreparedData/"):

    trainset = pd.read_csv(
        os.path.join(data_dir + "TrainData.csv"), index_col=0)

    trainset_feat = pd.read_csv(
        os.path.join(data_dir +"TrainDataFeatOffs.csv"), index_col=0)


    valset = pd.read_csv(
        os.path.join(data_dir + "ValData.csv"), index_col=0)

    valset_feat = pd.read_csv(
        os.path.join(data_dir +"ValDataFeatOffs.csv"), index_col=0)

    testset = pd.read_csv(
        os.path.join(data_dir +  "TestData.csv"), index_col=0)

    testset_feat = pd.read_csv(
        os.path.join(data_dir + "TestDataFeatOffs.csv"), index_col=0)


    return trainset, trainset_feat, valset, valset_feat, testset, testset_feat