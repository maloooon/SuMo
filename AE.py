import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from pycox import models
import torchtuples as tt
import FCNN
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


        if self.types[0] == 'cross' \
                or self.types[0] == 'cross_concat' \
                or self.types[0] == 'cross_elementwiseavg' \
                or self.types[0] == 'cross_elementwisemean' \
                or self.types[0] == 'cross_elementwisemax' \
                or self.types[0] == 'cross_elementwisemin' \
                or self.types[0] == 'cross_overallavg' \
                or self.types[0] == 'cross_overallmean' \
                or self.types[0] == 'cross_overallmin' \
                or self.types[0] == 'cross_overallmax':

            element_wise_avg, final_out_1, final_out_1_cross, input_data_1 = self.ae_1(*x)

            if self.types[1] == 'cross' \
                    or self.types[1] == 'cross_concat' \
                    or self.types[1] == 'cross_elementwiseavg' \
                    or self.types[1] == 'cross_elementwisemean' \
                    or self.types[1] == 'cross_elementwisemax' \
                    or self.types[1] == 'cross_elementwisemin' \
                    or self.types[1] == 'cross_overallavg' \
                    or self.types[1] == 'cross_overallmean' \
                    or self.types[1] == 'cross_overallmin' \
                    or self.types[1] == 'cross_overallmax':

                raise Exception("Cross 2 times does not work, since the first AE gives back an element wise average"
                                " of all the views, thus we can't cross mutate over different views anymore")
            else:
                integrated, final_out_2, input_data_2 = self.ae_2(element_wise_avg)
                hazard = self.nn(integrated)
                return final_out_1, final_out_2, final_out_1_cross, hazard, input_data_2, input_data_1

        else:
            view_data, final_out_1, input_data_1 = self.ae_1(*x)
            if self.types[1] == 'cross' \
                    or self.types[1] == 'cross_concat' \
                    or self.types[1] == 'cross_elementwiseavg' \
                    or self.types[1] == 'cross_elementwisemean' \
                    or self.types[1] == 'cross_elementwisemax' \
                    or self.types[1] == 'cross_elementwisemin' \
                    or self.types[1] == 'cross_overallavg' \
                    or self.types[1] == 'cross_overallmean' \
                    or self.types[1] == 'cross_overallmin' \
                    or self.types[1] == 'cross_overallmax':
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


        if self.types[0] == 'cross' \
                or self.types[0] == 'cross_concat' \
                or self.types[0] == 'cross_elementwiseavg' \
                or self.types[0] == 'cross_elementwisemean' \
                or self.types[0] == 'cross_elementwisemax' \
                or self.types[0] == 'cross_elementwisemin' \
                or self.types[0] == 'cross_overallavg' \
                or self.types[0] == 'cross_overallmean' \
                or self.types[0] == 'cross_overallmin' \
                or self.types[0] == 'cross_overallmax':

            element_wise_avg, final_out_1, final_out_1_cross, input_data_1 = self.ae_1(*x)

            if self.types[1] == 'cross' \
                    or self.types[1] == 'cross_concat' \
                    or self.types[1] == 'cross_elementwiseavg' \
                    or self.types[1] == 'cross_elementwisemean' \
                    or self.types[1] == 'cross_elementwisemax' \
                    or self.types[1] == 'cross_elementwisemin' \
                    or self.types[1] == 'cross_overallavg' \
                    or self.types[1] == 'cross_overallmean' \
                    or self.types[1] == 'cross_overallmin' \
                    or self.types[1] == 'cross_overallmax':
                raise Exception("Cross 2 times does not work, since the first AE gives back an element wise average"
                                " of all the views, thus we can't cross mutate over different views anymore")

            else:
                integrated, final_out_2, input_data_2 = self.ae_2(element_wise_avg)
                hazard = self.nn(integrated)

        else:
            view_data, final_out_1, input_data_1 = self.ae_1(*x)

            if self.types[1] == 'cross' \
                    or self.types[1] == 'cross_concat' \
                    or self.types[1] == 'cross_elementwiseavg' \
                    or self.types[1] == 'cross_elementwisemean' \
                    or self.types[1] == 'cross_elementwisemax' \
                    or self.types[1] == 'cross_elementwisemin' \
                    or self.types[1] == 'cross_overallavg' \
                    or self.types[1] == 'cross_overallmean' \
                    or self.types[1] == 'cross_overallmin' \
                    or self.types[1] == 'cross_overallmax':
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


        if self.type == 'cross' \
                or self.type == 'cross_concat' \
                or self.type == 'cross_elementwiseavg' \
                or self.type == 'cross_elementwisemean' \
                or self.type == 'cross_elementwisemax' \
                or self.type == 'cross_elementwisemin' \
                or self.type == 'cross_overallavg' \
                or self.type == 'cross_overallmean' \
                or self.type == 'cross_overallmin' \
                or self.type == 'cross_overallmax':
            element_wise_avg, final_out_cross, final_out, input_data = self.ae(*x) # TODO : change element wise name
            hazard = self.nn(element_wise_avg)
            return final_out, final_out_cross, hazard, input_data
        else:
            integrated, final_out, input_data = self.ae(*x)
            hazard = self.nn(integrated)
            return final_out, hazard,input_data


    def predict(self,*x):
        # Will be used by model.predict later.
        if self.type == 'cross' \
                or self.type == 'cross_concat' \
                or self.type == 'cross_elementwiseavg' \
                or self.type == 'cross_elementwisemean' \
                or self.type == 'cross_elementwisemax' \
                or self.type == 'cross_elementwisemin' \
                or self.type == 'cross_overallavg' \
                or self.type == 'cross_overallmean' \
                or self.type == 'cross_overallmin' \
                or self.type == 'cross_overallmax':
            element_wise_avg, final_out_cross, final_out, input_data = self.ae(*x) # TODO  : change element wise name
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
                 ae_hierarichcal_bool = False, ae_hierarichcal_decoding_bool = False, print_bool = False,
                 prelu_init = 0.25):
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
        self.prelu_init = prelu_init



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
                    elif activfunc.lower() == 'prelu':
                        activ_funcs[c][c2] = nn.PReLU(init=prelu_init)

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
                    if batch_norm_bool == True and batch_norm[c][-1] == 'yes':
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






        if type_ae.lower() == 'concat' or type_ae.lower() == 'cross_concat':
            # Concatenate the output, which will then be passed to a NN for survival analysis
            # the final output we're interested in (middle between encoder and decoder) was therefore already
            # saved in middle_dims list

            concatenated_features = sum([dim for dim in self.middle_dims])


        if type_ae.lower() == 'cross' \
                or type_ae.lower() == 'cross_concat'\
                or type_ae.lower() == 'cross_elementwiseavg' \
                or type_ae.lower() == 'cross_elementwisemean' \
                or type_ae.lower() == 'cross_elementwisemax'\
                or type_ae.lower() == 'cross_elementwisemin' \
                or type_ae.lower() =='cross_overallavg' \
                or type_ae.lower() == 'cross_overallmean' \
                or type_ae.lower() =='cross_overallmax' \
                or type_ae.lower() =='cross_overallmin':
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
                print("Finally, for Concat Integration, the output of each view between encoder and decoder  is concatenated  ({} features) "
                      "and will be passed to a NN for survival analysis".format(concatenated_features))

            if type_ae.lower() == 'cross' \
                or type_ae.lower() == 'cross_concat' \
                or type_ae.lower() == 'cross_elementwiseavg' \
                or type_ae.lower() == 'cross_elementwisemean' \
                or type_ae.lower() == 'cross_elementwisemax' \
                or type_ae.lower() == 'cross_elementwisemin' \
                or type_ae.lower() =='cross_overallavg' \
                or type_ae.lower() =='cross_overallmean' \
                or type_ae.lower() =='cross_overallmax' \
                or type_ae.lower() =='cross_overallmin':
                for c, _ in enumerate(self.helping_layer):
                    print( "For CrossAE implementation we have the following helping layers : {} for view {}"
                       .format(self.helping_layer[c], self.views[c]))
                print("Finally, for CrossAE, the output of each view between encoder and decoder  is integrated" # TODO : cross_concat print auch hier
                      ", thus {} elements  will be passed to a NN for survival analysis or 1 in the case of overall cross implementation".format(mmm_output_dimension))

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

        data_ordered = list(x)


        batch_size = data_ordered[0].size(0)




        for c,view in enumerate(self.hidden_layers):
            for c2 in range(len(view)): # view contains all layers (also the last to recreate input dim, as we already have defined it in __init__

                if c2 == 0: #first layer
                    # Apply dropout layer
                    if self.dropout_bool == True and self.dropout_layers[c][c2] == 'yes':
                       # encoded_features[c][c2] = self.dropout(encoded_features[c][c2])
                        data_ordered[c] = self.dropout(data_ordered[c])
                    encoded_features[c].append(self.hidden_layers[c][c2](data_ordered[c]))


                elif c2 == len(view) - 1: # last layer (decoder); not same structure as in init, bc now the last layer is already in view!
                    if self.dropout_bool == True and self.dropout_layers[c][-1] == 'yes':
                        encoded_features[c][-1] =self.dropout(encoded_features[c][-1])

                    encoded_features[c].append(self.hidden_layers[c][-1](encoded_features[c][-1]))



                else : # other layers
                    if self.dropout_bool == True and self.dropout_layers[c][c2] == 'yes':
                        encoded_features[c][c2-1] = self.dropout(encoded_features[c][c2-1])
                    encoded_features[c].append(self.hidden_layers[c][c2](encoded_features[c][c2-1]))



        # The data we're interested in is in the middle of the encoding and decoding stage

        data_middle = []
        middle_pos = []

        for c,view in enumerate(self.n_hidden_layers_dims):
            middle = (len(view) // 2)
            middle_pos.append(middle)
            data_middle.append(encoded_features[c][middle])






        # Concatenate the output, which will then be passed to a NN for survival analysis
        concatenated_features = torch.cat(tuple(data_middle), dim=-1)




        if self.type_ae.lower() == 'overallmean'\
                or self.type_ae.lower() == 'overallavg' \
                or self.type_ae.lower() == 'cross_overallavg' \
                or self.type_ae.lower() == 'cross_overallmean':
            overall_mean = torch.mean(concatenated_features,1,True)

        if self.type_ae.lower() == 'overallmax' or self.type_ae.lower() == 'cross_overallmax':
            overall_max = torch.amax(concatenated_features,1,True)


        if self.type_ae.lower() == 'overallmin' or self.type_ae.lower() == 'cross_overallmin':
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
                if self.type_ae.lower() == 'cross' \
                        or self.type_ae.lower() == 'elementwisemean' \
                        or self.type_ae.lower() == 'elementwiseavg' \
                        or self.type_ae.lower() == 'cross_elementwisemean':
                    mean = torch.mean(torch.stack(temp))
                    element_wise_avg[i][x] = mean
                if self.type_ae.lower() == 'elementwisemax' or self.type_ae.lower() == 'cross_elementwisemax':
                    max = torch.amax(torch.stack(temp))
                    element_wise_max[i][x] = max
                if self.type_ae.lower() == 'elementwisemin' or self.type_ae.lower() == 'cross_elementwisemin':
                    min = torch.amin(torch.stack(temp))
                    element_wise_min[i][x] = min


        if self.type_ae.lower() == 'cross' \
                or self.type_ae.lower() == 'cross_concat' \
                or self.type_ae.lower() == 'cross_elementwiseavg' \
                or self.type_ae.lower() == 'cross_elementwisemean' \
                or self.type_ae.lower() == 'cross_elementwisemax' \
                or self.type_ae.lower() == 'cross_elementwisemin' \
                or self.type_ae.lower() =='cross_overallavg' \
                or self.type_ae.lower() =='cross_overallmean' \
                or self.type_ae.lower() =='cross_overallmax' \
                or self.type_ae.lower() =='cross_overallmin':

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

        if self.type_ae.lower() == 'cross': # cross element wise ; we need output for each cross type
            # element_wise_avg will be passed to a NN to get the hazard ratio
            # final_out_cross will be used to train the AE
            return element_wise_avg, final_out_cross, final_out, input_data_raw


        if self.type_ae.lower() == 'cross_concat': # cross element wise ; we need output for each cross type
            # element_wise_avg will be passed to a NN to get the hazard ratio
            # final_out_cross will be used to train the AE
            return concatenated_features, final_out_cross, final_out, input_data_raw

        if self.type_ae.lower() == 'cross_elementwisemean' or self.type_ae.lower() == 'cross_elementwiseavg': # cross element wise ; we need output for each cross type
            # element_wise_avg will be passed to a NN to get the hazard ratio
            # final_out_cross will be used to train the AE
            return element_wise_avg, final_out_cross, final_out, input_data_raw

        if self.type_ae.lower() == 'cross_elementwisemax': # cross element wise ; we need output for each cross type
            # element_wise_avg will be passed to a NN to get the hazard ratio
            # final_out_cross will be used to train the AE
            return element_wise_max, final_out_cross, final_out, input_data_raw

        if self.type_ae.lower() == 'cross_elementwisemin': # cross element wise ; we need output for each cross type
            # element_wise_avg will be passed to a NN to get the hazard ratio
            # final_out_cross will be used to train the AE
            return element_wise_min, final_out_cross, final_out, input_data_raw

        if self.type_ae.lower() == 'cross_overallavg' or self.type_ae.lower() == 'cross_overallmean': # cross element wise ; we need output for each cross type
            # element_wise_avg will be passed to a NN to get the hazard ratio
            # final_out_cross will be used to train the AE
            return overall_mean, final_out_cross, final_out, input_data_raw

        if self.type_ae.lower() == 'cross_overallmax': # cross element wise ; we need output for each cross type
            # element_wise_avg will be passed to a NN to get the hazard ratio
            # final_out_cross will be used to train the AE
            return overall_max, final_out_cross, final_out, input_data_raw


        if self.type_ae.lower() == 'cross_overallmin': # cross element wise ; we need output for each cross type
            # element_wise_avg will be passed to a NN to get the hazard ratio
            # final_out_cross will be used to train the AE
            return overall_min, final_out_cross, final_out, input_data_raw

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

    second_decoder_bool = False
    model_types = ['concat']


    ##################################### HYPERPARAMETER SEARCH SETTINGS ##############################################
    l2_regularization_bool = trial.suggest_categorical('l2_regularization_bool', [True,False])
    learning_rate = trial.suggest_float("learning_rate", 1e-5,1e-1,log=True)
    l2_regularization_rate = trial.suggest_float("l2_regularization_rate", 1e-6,1e-3, log=True)
  #  batch_size = trial.suggest_int("batch_size", 5, 200)
    batch_size = trial.suggest_categorical("batch_size", [8,16,32,64,128,256])
  #  n_epochs = trial.suggest_int("n_epochs", 10,100) ########### TESTING
    n_epochs = 100
    dropout_prob = trial.suggest_float("dropout_prob", 0,0.5,step=0.1)
    dropout_bool = trial.suggest_categorical('dropout_bool', [True,False])
    batchnorm_bool = trial.suggest_categorical('batchnorm_bool',[True,False])
    prelu_rate = trial.suggest_float('prelu_rate',0,1,step=0.05)
    if len(model_types) == 2:
        dropout_prob_hierachical = trial.suggest_float("dropout_prob_hierachical", 0,0.5,step=0.1)
        dropout_bool_hierachical = trial.suggest_categorical('dropout_bool_hierachical', [True,False])
        batchnorm_bool_hierachical = trial.suggest_categorical('batchnorm_bool_hierachical',[True,False])





    # After concat FCNN (concatenated input into FCNN)

    # This is needed in nearly every case, just not if we don't integrate the data in the autoencoders in any way
    # (basically just let AE work as feature selection method) ; TODO : add if case for this case
    layers_1_FCNN = trial.suggest_int('layers_1_FCNN', 5, 300)
    layers_2_FCNN = trial.suggest_int('layers_2_FCNN', 5, 300)

    layers_FCNN = [[layers_1_FCNN,layers_2_FCNN]]

    layers_1_FCNN_activfunc = trial.suggest_categorical('layers_1_FCNN_activfunc', ['relu','sigmoid','prelu'])
    layers_2_FCNN_activfunc = trial.suggest_categorical('layers_2_FCNN_activfunc', ['relu','sigmoid','prelu'])

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


    # we only have MSE & survival loss :
    loss_surv = trial.suggest_float("loss_surv",0,1)
    # in Hierarchical implementation with a second decoder which we train, we have 2 MSE-Losses and a survival loss
    if second_decoder_bool == True:
        loss_MSE = trial.suggest_float("loss_MSE", 0,1)
        loss_MSE_2 = trial.suggest_float("loss_MSE_2",0,1)
        # losses need to sum up to 1
        summed_losses = loss_MSE + loss_MSE_2 + loss_surv
        loss_3_values_hierarchical = [loss_MSE/summed_losses, loss_MSE_2/summed_losses, loss_surv/summed_losses]


    layers = []
    layers_hierarchical = []
    layers_hierarchical_integrated = []
    activation_functions = []
    activation_functions_hierarchical = []
    activation_functions_hierarchical_integrated = []
    dropouts = []
    dropouts_hierarchical = []
    dropouts_hierarchical_integrated = []
    batchnorms = []
    batchnorms_hierarchical = []
    batchnorms_hierarchical_integrated = []

    if 'MRNA' in view_names:
        layers_1_mRNA = trial.suggest_int('layers_1_mRNA', 5, 300)
        layers_2_mRNA = trial.suggest_int('layers_2_mRNA', 5, 300)
        layers_1_mRNA_activfunc = trial.suggest_categorical('layers_1_mRNA_activfunc', ['relu','sigmoid','prelu'])
        layers_2_mRNA_activfunc = trial.suggest_categorical('layers_2_mRNA_activfunc', ['relu','sigmoid','prelu'])
        layers_1_mRNA_dropout = trial.suggest_categorical('layers_1_mRNA_dropout', ['yes','no'])
        layers_2_mRNA_dropout = trial.suggest_categorical('layers_2_mRNA_dropout', ['yes','no'])
        layers_1_mRNA_batchnorm = trial.suggest_categorical('layers_1_mRNA_batchnorm', ['yes', 'no'])
        layers_2_mRNA_batchnorm = trial.suggest_categorical('layers_2_mRNA_batchnorm', ['yes', 'no'])
        if len(model_types) == 2 and model_types[0] == 'none':
            layers_1_mRNA_hierarichcal = trial.suggest_int('layers_1_mRNA_hierarichcal', 5, 300)
            layers_2_mRNA_hierarichcal = trial.suggest_int('layers_2_mRNA_hierarichcal', 5, 300)
            layers_1_mRNA_activfunc_hierarichcal = trial.suggest_categorical('layers_1_mRNA_activfunc_hierarichcal', ['relu','sigmoid','prelu'])
            layers_2_mRNA_activfunc_hierarichcal = trial.suggest_categorical('layers_2_mRNA_activfunc_hierarichcal', ['relu','sigmoid','prelu'])
            layers_1_mRNA_dropout_hierarichcal = trial.suggest_categorical('layers_1_mRNA_dropout_hierarichcal', ['yes','no'])
            layers_2_mRNA_dropout_hierarichcal = trial.suggest_categorical('layers_2_mRNA_dropout_hierarichcal', ['yes','no'])
            layers_1_mRNA_batchnorm_hierarichcal = trial.suggest_categorical('layers_1_mRNA_batchnorm_hierarichcal', ['yes', 'no'])
            layers_2_mRNA_batchnorm_hierarichcal = trial.suggest_categorical('layers_2_mRNA_batchnorm_hierarichcal', ['yes', 'no'])
            layers_hierarchical.append([layers_1_mRNA_hierarichcal,layers_2_mRNA_hierarichcal])
            activation_functions_hierarchical.append([layers_1_mRNA_activfunc_hierarichcal, layers_2_mRNA_activfunc_hierarichcal])
            dropouts_hierarchical.append([layers_1_mRNA_dropout_hierarichcal, layers_2_mRNA_dropout_hierarichcal])
            batchnorms_hierarchical.append([layers_1_mRNA_batchnorm_hierarichcal, layers_2_mRNA_batchnorm_hierarichcal])



        layers.append([layers_1_mRNA,layers_2_mRNA])
        activation_functions.append([layers_1_mRNA_activfunc, layers_2_mRNA_activfunc])
        dropouts.append([layers_1_mRNA_dropout, layers_2_mRNA_dropout])
        batchnorms.append([layers_1_mRNA_batchnorm, layers_2_mRNA_batchnorm])


    if 'DNA' in view_names:
        layers_1_DNA = trial.suggest_int('layers_1_DNA', 5, 300)
        layers_2_DNA = trial.suggest_int('layers_2_DNA', 5, 300)
        layers_1_DNA_activfunc = trial.suggest_categorical('layers_1_DNA_activfunc', ['relu','sigmoid','prelu'])
        layers_2_DNA_activfunc = trial.suggest_categorical('layers_2_DNA_activfunc', ['relu','sigmoid','prelu'])
        layers_1_DNA_dropout = trial.suggest_categorical('layers_1_DNA_dropout', ['yes','no'])
        layers_2_DNA_dropout = trial.suggest_categorical('layers_2_DNA_dropout', ['yes','no'])
        layers_1_DNA_batchnorm = trial.suggest_categorical('layers_1_DNA_batchnorm', ['yes', 'no'])
        layers_2_DNA_batchnorm = trial.suggest_categorical('layers_2_DNA_batchnorm', ['yes', 'no'])

        if len(model_types) == 2 and model_types[0] == 'none':
            layers_1_DNA_hierarichcal = trial.suggest_int('layers_1_DNA_hierarichcal', 5, 300)
            layers_2_DNA_hierarichcal = trial.suggest_int('layers_2_DNA_hierarichcal', 5, 300)
            layers_1_DNA_activfunc_hierarichcal = trial.suggest_categorical('layers_1_DNA_activfunc_hierarichcal', ['relu','sigmoid', 'prelu'])
            layers_2_DNA_activfunc_hierarichcal = trial.suggest_categorical('layers_2_DNA_activfunc_hierarichcal', ['relu','sigmoid', 'prelu'])
            layers_1_DNA_dropout_hierarichcal = trial.suggest_categorical('layers_1_DNA_dropout_hierarichcal', ['yes','no'])
            layers_2_DNA_dropout_hierarichcal = trial.suggest_categorical('layers_2_DNA_dropout_hierarichcal', ['yes','no'])
            layers_1_DNA_batchnorm_hierarichcal = trial.suggest_categorical('layers_1_DNA_batchnorm_hierarichcal', ['yes', 'no'])
            layers_2_DNA_batchnorm_hierarichcal = trial.suggest_categorical('layers_2_DNA_batchnorm_hierarichcal', ['yes', 'no'])
            layers_hierarchical.append([layers_1_DNA_hierarichcal,layers_2_DNA_hierarichcal])
            activation_functions_hierarchical.append([layers_1_DNA_activfunc_hierarichcal, layers_2_DNA_activfunc_hierarichcal])
            dropouts_hierarchical.append([layers_1_DNA_dropout_hierarichcal, layers_2_DNA_dropout_hierarichcal])
            batchnorms_hierarchical.append([layers_1_DNA_batchnorm_hierarichcal, layers_2_DNA_batchnorm_hierarichcal])


        layers.append([layers_1_DNA,layers_2_DNA])
        activation_functions.append([layers_1_DNA_activfunc, layers_2_DNA_activfunc])
        dropouts.append([layers_1_DNA_dropout, layers_2_DNA_dropout])
        batchnorms.append([layers_1_DNA_batchnorm, layers_2_DNA_batchnorm])


    if 'MICRORNA' in view_names:
        layers_1_microRNA = trial.suggest_int('layers_1_microRNA', 5, 300)
        layers_2_microRNA = trial.suggest_int('layers_2_microRNA', 5, 300)
        layers_1_microRNA_activfunc = trial.suggest_categorical('layers_1_microRNA_activfunc', ['relu','sigmoid','prelu'])
        layers_2_microRNA_activfunc = trial.suggest_categorical('layers_2_microRNA_activfunc', ['relu','sigmoid','prelu'])
        layers_1_microRNA_dropout = trial.suggest_categorical('layers_1_microRNA_dropout', ['yes','no'])
        layers_2_microRNA_dropout = trial.suggest_categorical('layers_2_microRNA_dropout', ['yes','no'])
        layers_1_microRNA_batchnorm = trial.suggest_categorical('layers_1_microRNA_batchnorm', ['yes', 'no'])
        layers_2_microRNA_batchnorm = trial.suggest_categorical('layers_2_microRNA_batchnorm', ['yes', 'no'])

        if len(model_types) == 2 and model_types[0] == 'none':
            layers_1_microRNA_hierarichcal = trial.suggest_int('layers_1_microRNA_hierarichcal', 5, 300)
            layers_2_microRNA_hierarichcal = trial.suggest_int('layers_2_microRNA_hierarichcal', 5, 300)
            layers_1_microRNA_activfunc_hierarichcal = trial.suggest_categorical('layers_1_microRNA_activfunc_hierarichcal', ['relu','sigmoid', 'prelu'])
            layers_2_microRNA_activfunc_hierarichcal = trial.suggest_categorical('layers_2_microRNA_activfunc_hierarichcal', ['relu','sigmoid', 'prelu'])
            layers_1_microRNA_dropout_hierarichcal = trial.suggest_categorical('layers_1_microRNA_dropout_hierarichcal', ['yes','no'])
            layers_2_microRNA_dropout_hierarichcal = trial.suggest_categorical('layers_2_microRNA_dropout_hierarichcal', ['yes','no'])
            layers_1_microRNA_batchnorm_hierarichcal = trial.suggest_categorical('layers_1_microRNA_batchnorm_hierarichcal', ['yes', 'no'])
            layers_2_microRNA_batchnorm_hierarichcal = trial.suggest_categorical('layers_2_microRNA_batchnorm_hierarichcal', ['yes', 'no'])
            layers_hierarchical.append([layers_1_microRNA_hierarichcal,layers_2_microRNA_hierarichcal])
            activation_functions_hierarchical.append([layers_1_microRNA_activfunc_hierarichcal, layers_2_microRNA_activfunc_hierarichcal])
            dropouts_hierarchical.append([layers_1_microRNA_dropout_hierarichcal, layers_2_microRNA_dropout_hierarichcal])
            batchnorms_hierarchical.append([layers_1_microRNA_batchnorm_hierarichcal, layers_2_microRNA_batchnorm_hierarichcal])

        layers.append([layers_1_microRNA,layers_2_microRNA])
        activation_functions.append([layers_1_microRNA_activfunc, layers_2_microRNA_activfunc])
        dropouts.append([layers_1_microRNA_dropout, layers_2_microRNA_dropout])
        batchnorms.append([layers_1_microRNA_batchnorm, layers_2_microRNA_batchnorm])


    if 'RPPA' in view_names:
        layers_1_RPPA = trial.suggest_int('layers_1_microRNA', 5, 300)
        layers_2_RPPA = trial.suggest_int('layers_1_microRNA', 5, 300)
        layers_1_RPPA_activfunc = trial.suggest_categorical('layers_1_RPPA_activfunc', ['relu','sigmoid','prelu'])
        layers_2_RPPA_activfunc = trial.suggest_categorical('layers_2_RPPA_activfunc', ['relu','sigmoid','prelu'])
        layers_1_RPPA_dropout = trial.suggest_categorical('layers_1_RPPA_dropout', ['yes','no'])
        layers_2_RPPA_dropout = trial.suggest_categorical('layers_2_RPPA_dropout', ['yes','no'])
        layers_1_RPPA_batchnorm = trial.suggest_categorical('layers_1_RPPA_batchnorm', ['yes', 'no'])
        layers_2_RPPA_batchnorm = trial.suggest_categorical('layers_2_RPPA_batchnorm', ['yes', 'no'])

        if len(model_types) == 2 and model_types[0] == 'none':
            layers_1_RPPA_hierarichcal = trial.suggest_int('layers_1_RPPA_hierarichcal', 5, 300)
            layers_2_RPPA_hierarichcal = trial.suggest_int('layers_2_RPPA_hierarichcal', 5, 300)
            layers_1_RPPA_activfunc_hierarichcal = trial.suggest_categorical('layers_1_RPPA_activfunc_hierarichcal', ['relu','sigmoid', 'prelu'])
            layers_2_RPPA_activfunc_hierarichcal = trial.suggest_categorical('layers_2_RPPA_activfunc_hierarichcal', ['relu','sigmoid', 'prelu'])
            layers_1_RPPA_dropout_hierarichcal = trial.suggest_categorical('layers_1_RPPA_dropout_hierarichcal', ['yes','no'])
            layers_2_RPPA_dropout_hierarichcal = trial.suggest_categorical('layers_2_RPPA_dropout_hierarichcal', ['yes','no'])
            layers_1_RPPA_batchnorm_hierarichcal = trial.suggest_categorical('layers_1_RPPA_batchnorm_hierarichcal', ['yes', 'no'])
            layers_2_RPPA_batchnorm_hierarichcal = trial.suggest_categorical('layers_2_RPPA_batchnorm_hierarichcal', ['yes', 'no'])
            layers_hierarchical.append([layers_1_RPPA_hierarichcal,layers_2_RPPA_hierarichcal])
            activation_functions_hierarchical.append([layers_1_RPPA_activfunc_hierarichcal, layers_2_RPPA_activfunc_hierarichcal])
            dropouts_hierarchical.append([layers_1_RPPA_dropout_hierarichcal, layers_2_RPPA_dropout_hierarichcal])
            batchnorms_hierarchical.append([layers_1_RPPA_batchnorm_hierarichcal, layers_2_RPPA_batchnorm_hierarichcal])


        layers.append([layers_1_RPPA,layers_2_RPPA])
        activation_functions.append([layers_1_RPPA_activfunc, layers_2_RPPA_activfunc])
        dropouts.append([layers_1_RPPA_dropout, layers_2_RPPA_dropout])
        batchnorms.append([layers_1_RPPA_batchnorm, layers_2_RPPA_batchnorm])



    if len(model_types) == 2 and model_types[0] != 'none':
    # In this case we already have done integration method and have single omic data structure left
        layers_1_hierarichcal_integrated = trial.suggest_int('layers_1_hierarichcal_integrated', 5, 300)
        layers_2_hierarichcal_integrated = trial.suggest_int('layers_2_hierarichcal_integrated', 5, 300)
        layers_1_activfunc_hierarichcal_integrated = trial.suggest_categorical('layers_1_activfunc_hierarichcal_integrated', ['relu','sigmoid', 'prelu'])
        layers_2_activfunc_hierarichcal_integrated = trial.suggest_categorical('layers_2_activfunc_hierarichcal_integrated', ['relu','sigmoid', 'prelu'])
        layers_1_dropout_hierarichcal_integrated = trial.suggest_categorical('layers_1_dropout_hierarichcal_integrated', ['yes','no'])
        layers_2_dropout_hierarichcal_integrated = trial.suggest_categorical('layers_2_dropout_hierarichcal_integrated', ['yes','no'])
        layers_1_batchnorm_hierarichcal_integrated = trial.suggest_categorical('layers_1_batchnorm_hierarichcal_integrated', ['yes', 'no'])
        layers_2_batchnorm_hierarichcal_integrated = trial.suggest_categorical('layers_2_batchnorm_hierarichcal_integrated', ['yes', 'no'])
        dropout_prob_hierachical_integrated = trial.suggest_float("dropout_prob_hierachical_integrated", 0,0.5,step=0.1)
        dropout_bool_hierachical_integrated = trial.suggest_categorical('dropout_bool_hierachical_integrated', [True,False])
        batchnorm_bool_hierachical_integrated = trial.suggest_categorical('batchnorm_bool_hierachical_integrated',[True,False])
        layers_hierarchical_integrated.append([layers_1_hierarichcal_integrated,layers_2_hierarichcal_integrated])
        activation_functions_hierarchical_integrated.append([layers_1_activfunc_hierarichcal_integrated, layers_2_activfunc_hierarichcal_integrated])
        dropouts_hierarchical_integrated.append([layers_1_dropout_hierarichcal_integrated, layers_2_dropout_hierarichcal_integrated])
        batchnorms_hierarchical_integrated.append([layers_1_batchnorm_hierarichcal_integrated, layers_2_batchnorm_hierarichcal_integrated])



    # Create List of models to be used
    all_models = nn.ModuleList()




    # for cross implementation # TODO : warning bc of data structure, how can this be done in another way with optuna ?
    if len(layers) == 4 and 'cross' in model_types[0]:
        cross_decoder_4_views = trial.suggest_categorical("cross_decoders_4_views",[(0,1,2,3),(0,1,3,2),(1,0,2,3),(1,0,3,2)]) #TODO: fulfill with chatgpt
    if len(layers) == 3 and 'cross' in model_types[0]:
        cross_decoders_3_views = trial.suggest_categorical("cross_decoders_3_views",[(0,1,2),(0,2,1),(2,1,0),(1,2,0),(1,0,2),(2,0,1)])
    if len(layers) == 2 and 'cross' in model_types[0]:
        cross_decoders_2_views = trial.suggest_categorical("cross_decoders_2_views",[(0,1),(1,0)]) # TODO : als string definieren ; dann splitten für NN call




  #  print("MODEL TYPES : ", model_types)

    # for concatenated data input into FCNN (no hierarichcal AE)
    out_sizes = []
    for c_layer in range(len(layers)):
        out_sizes.append(layers[c_layer][-1])

    in_feats_second_NN_concat = sum(out_sizes)

    # for element-wise avg/min/max input into FCNN (no hierarichcal AE)

    # Size is the one of the largest output dim of AE layers
    in_feats_second_NN_element_wise = max([i[-1] for i in layers])

    # After hierarichcal integrated method
    if len(model_types) == 2:
        in_feats_third_NN_element_wise = max([i[-1] for i in layers_hierarchical_integrated])

        out_sizes_hierachical = []

        in_feats_third_NN_concat = layers_hierarchical_integrated[0][-1] # integrated method has only one layer left

    in_feats_second_NN_overall = 1




    # Note that for the Cross Method, the cross_mutation must be set in a way so that the input view x and
    # "crossed" view y have the same feature size : Otherwise, the MSELoss() cant be calculated
    # e.g. 4 views having feature sizes [15,15,5,5] --> cross mutations between first two and last two work,
    # but not e.g. between first and third/fourth

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
                         cross_mutation=[1,2,0],
                         print_bool=False,
                         prelu_init= prelu_rate))


  #  all_models.append(AE(views = ['AE'],
  #                       in_features= [in_feats_second_NN_element_wise],
  #                       n_hidden_layers_dims= layers_hierarchical_integrated,
  #                       activ_funcs=activation_functions_hierarchical_integrated,
  #                       dropout_bool= dropout_bool_hierachical_integrated,
  #                       dropout_prob= dropout_prob_hierachical_integrated,
  #                       dropout_layers= dropouts_hierarchical_integrated,
  #                       batch_norm_bool= batchnorm_bool_hierachical_integrated,
  #                       batch_norm= batchnorms_hierarchical_integrated,
   #                      type_ae=model_types[1],
  #                       cross_mutation=None,
   #                      print_bool=False))


    all_models.append(FCNN.NN_changeable(views = ['AE'],
                                       in_features = [in_feats_second_NN_concat],
                                       n_hidden_layers_dims= layers_FCNN,
                                       activ_funcs = [FCNN_activation_functions,['none']],
                                       dropout_prob=FCNN_dropout_prob,
                                       dropout_layers=FCNN_dropouts,
                                       batch_norm = FCNN_batchnorms,
                                       dropout_bool=FCNN_dropout_bool,
                                       batch_norm_bool=FCNN_batchnorm_bool,
                                       print_bool=False,
                                       prelu_init = prelu_rate))




   # full_net = AE_Hierarichal(all_models, types=model_types)

    full_net = AE_NN(all_models, type=model_types[0])


    # set optimizer
    if l2_regularization_bool == True:
        optimizer = Adam(full_net.parameters(), lr=learning_rate, weight_decay=l2_regularization_rate)
    else:
        optimizer = Adam(full_net.parameters(), lr=learning_rate)

    callbacks = [tt.callbacks.EarlyStopping(patience=10)]
    #   cross_pos = model_types.index("cross") + 1
    #   model = models.CoxPH(full_net,optimizer, loss=LossHierarichcalAESingleCross(alpha=[0.4,0.4,0.2], decoding_bool=True, cross_position=cross_pos)) # Change Loss here

    loss_concat = LossAEConcatHazard(loss_surv)
    loss_cross = LossAECrossHazard(loss_surv)

    # if second decoder, we need to use loss_3 ; else normal 2 valued loss
    if len(model_types) == 2:
        loss_hierachical_no_cross = LossHierarichcalAE(loss_3_values_hierarchical)
        loss_hierachical_cross = LossHierarichcalAESingleCross(loss_3_values_hierarchical, cross_position=1)
    # loss : alpha * surv_loss + (1-alpha) * ae_loss
    model = models.CoxPH(full_net,
                         optimizer,
                         loss=loss_concat)
    print_loss = True

    log = model.fit(train_data,
                    train_surv,
                    batch_size,
                    n_epochs,
                    verbose=print_loss,
                    val_data= val_data_full,
                    val_batch_size= batch_size,
                    callbacks=callbacks)

    # Plot it
    _ = log.plot()

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


def train(train_data,val_data,test_data,
          train_duration,val_duration,test_duration,
          train_event,val_event,test_event,
          n_epochs,
          batch_size,
          l2_regularization,
          l2_regularization_rate,
          learning_rate,
          prelu_rate,
          layers,
          activation_layers,
          dropout,
          dropout_rate,
          dropout_layers,
          batchnorm,
          batchnorm_layers,
          view_names,
          cross_mutation,
          model_types,
          dropout_second,
          dropout_rate_second,
          batchnorm_second,
          layers_second,
          activation_layers_second,
          dropout_layers_second,
          batchnorm_layers_second,
          dropout_third,
          dropout_rate_third,
          batchnorm_third,
          layers_third,
          activation_layers_third,
          dropout_layers_third,
          batchnorm_layers_third):


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


     #   model_types = ['concat','elementwisemax']

        print("MODEL TYPES : ", model_types)


        out_sizes = []
        for c_layer in range(len(layers)):
            out_sizes.append(layers[c_layer][-1])


        in_feats_second_NN_concat = sum(out_sizes)
        in_feats_second_NN_elementwise = max([i[-1] for i in layers])

        out_sizes_second = []
        for c_layer in range(len(layers_second)):
            out_sizes_second.append(layers_second[c_layer][-1])


        in_feats_third_NN_concat = sum(out_sizes_second)
        in_feats_third_NN_elementwise = max([i[-1] for i in layers_second])


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
                             cross_mutation=cross_mutation,
                             print_bool=False,
                             prelu_init= prelu_rate))


        layers_u_2 = copy.deepcopy(layers_second)
        activation_layers_u_2 = copy.deepcopy(activation_layers_second)
        dropout_layers_u_2 = copy.deepcopy(dropout_layers_second)
        batchnorm_layers_u_2 = copy.deepcopy(batchnorm_layers_second)
        all_models.append(AE(views = ['AE'],
                             in_features=[in_feats_second_NN_concat],
                             n_hidden_layers_dims= layers_u_2,
                             activ_funcs = activation_layers_u_2,
                             dropout_prob= dropout_second,
                             dropout_layers = dropout_layers_u_2,
                             batch_norm = batchnorm_layers_u_2,
                             dropout_bool=dropout_second,
                             batch_norm_bool=batchnorm_second,
                             type_ae =model_types[1],
                             cross_mutation=[1,0,3,2],
                             ae_hierarichcal_bool= True,
                             print_bool=False))






        # NN

        # Note : For Concat, the in_feats for NN need to be the sum of all last output layer dimensions.
        #        For Cross, the in_feats for NN need to be the size of the largest output layer dim, as we take the
        #                    element wise avg
        #        For overall(mean/max/min), the in_feats for NN is 1 (as we take the overall average)
        #        For elementwise(mean/max/min), the in_feats for NN must be size of the largest output layer dim

        layers_u_3 = copy.deepcopy(layers_third)
        activation_layers_u_3 = copy.deepcopy(activation_layers_third)
        dropout_layers_u_3 = copy.deepcopy(dropout_layers_third)
        batchnorm_layers_u_3 = copy.deepcopy(batchnorm_layers_third)
        all_models.append(FCNN.NN_changeable(views = ['AE'],
                                           in_features = [in_feats_third_NN_concat],
                                           n_hidden_layers_dims= layers_u_3,
                                           activ_funcs = activation_layers_u_3,
                                           dropout_prob=dropout_third,
                                           dropout_layers=dropout_layers_u_3,
                                           batch_norm =batchnorm_layers_u_3,
                                           dropout_bool=dropout_third,
                                           batch_norm_bool=batchnorm_third,
                                           print_bool=False,
                                           prelu_init= prelu_rate))



        #############
        # Note that for the Cross Method, the cross_mutation must be set in a way so that the input view x and
        # "crossed" view y have the same feature size : Otherwise, the MSELoss() cant be calculated
        # e.g. 4 views having feature sizes [15,15,5,5] --> cross mutations between first two and last two work,
        # but not e.g. between first and third/fourth





        full_net = AE_Hierarichal(all_models, types=model_types)

   #     full_net = AE_NN(all_models, type=model_types[0])


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
                             loss=LossHierarichcalAE(alpha=[0.4,0.4,0.2], decoding_bool=True))
        print_loss = True
        print("Split {} : ".format(c_fold + 1))
        log = model.fit(train_data[c_fold],
                        train_surv,
                        batch_size,
                        n_epochs,
                        verbose=print_loss,
                        val_data= val_data_full,
                        val_batch_size= batch_size,
                        callbacks=callbacks)

        # Plot it
        _ = log.plot()

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