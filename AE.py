import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from pycox import models
import torchtuples as tt
import FCNN
from pycox.evaluation import EvalSurv
import copy
import optuna
import pandas as pd
import os


class AE_Hierarichal(nn.Module):
    """
    Wrapper for Hierarichal AE implementation : AE followed by another AE followed by FCNN.
       The second AE may be used without an decoder.
    """
    def __init__(self,models,types):

        """
        :param models: Models to be used ; dtype : ModuleList
        :param type: Bottlneck manipulation
        ; dtype : String ['concat','elementwisemax','elementwisemin',elementwiseavg','overallmax',
                          'overallmin','overallavg', 'cross' or 'cross_k' where k is on of the
                           mentionted types]
        :param decoding_bool: bool whether decoder should be used for the second AE or not ; dtype : Boolean
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

        """

        :param x: Data input (for each view) ; dtype : Tuple of Tensors(n_samples_in_batch, n_features)
        :return: final_out : Decoded output from AE ; dtype : Tuple of Tensors(n_samples_in_batch, n_features)
                 hazard : "Risk ratio" ; dtype : Tensor(n_samples_in_batch, 1)
                 input data : Data input (same as x) ; dtype : Tuple of Tensors(n_samples_in_batch, n_features)
                 final_out_cross : Cross decoded output from AE ; dtype : Tuple of Tensors(n_samples_in_batch, n_features)
        """

        if 'cross' in self.types[0].lower():

            integrated_element, final_out_1, final_out_1_cross, input_data_1 = self.ae_1(*x)

            if 'cross' in self.types[1].lower():

                raise Exception("Cross 2 times does not work, since the first AE gives back an integrated element"
                                "of all the views, thus we can't cross mutate over different views anymore")
            else:
                integrated, final_out_2, input_data_2 = self.ae_2(integrated_element)
                hazard = self.nn(integrated)
                return final_out_1, final_out_2, final_out_1_cross, hazard, input_data_2, input_data_1

        else:
            view_data, final_out_1, input_data_1 = self.ae_1(*x)
            if 'cross' in self.types[1].lower():

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
        """
         Predict function which will be used by model.predict later.
         :param x: Data input (for each view) ; dtype : Tuple of Tensors(n_samples_in_batch, n_features)
         :return: hazard : "Risk ratio" ; dtype : Tensor(n_samples_in_batch, 1)
         """
        if 'cross' in self.types[0].lower():


            integrated_element, final_out_1, final_out_1_cross, input_data_1 = self.ae_1(*x)

            if 'cross' in self.types[1].lower():

                raise Exception("Cross 2 times does not work, since the first AE gives back an integrated element"
                                "of all the views, thus we can't cross mutate over different views anymore")

            else:
                integrated, final_out_2, input_data_2 = self.ae_2(integrated_element)
                hazard = self.nn(integrated)

        else:
            view_data, final_out_1, input_data_1 = self.ae_1(*x)

            if 'cross' in self.types[1].lower():

                if self.types[0] != 'none':
                    raise Exception("Cross on the second AE only works if the output of the first AE still has"
                                    "different views and is not concatenated or averaged/minned/maxxed. Thus only "
                                    "works if we set the type of the first AE to none")
                else:
                    integrated, final_out_2, final_out_2_cross, input_data_2 = self.ae_2(*view_data)
                    hazard = self.nn(integrated)

            else:
                if self.types[0] != 'none':
                    integrated, final_out_2, input_data_2 = self.ae_2(view_data)
                    hazard = self.nn(integrated)
                else:
                    integrated, final_out_2, input_data_2 = self.ae_2(*tuple(view_data))
                    hazard = self.nn(integrated)


        return hazard



class AE_NN(nn.Module):
    """
    Wrapper for Pipeline of AE followed by FCNN.
    See :
    """
    def __init__(self, models, type):
        """

        :param models: Models to be used ; dtype : ModuleList
        :param type: Bottlneck manipulation
                     ; dtype : String ['concat','elementwisemax','elementwisemin',elementwiseavg','overallmax',
                                       'overallmin','overallavg', 'cross' or 'cross_k' where k is on of the
                                        mentionted types]
        """
        super().__init__()
        self.models = models
        self.type = type
        # AE call
        self.ae = self.models[0]
        # NN call
        self.nn = self.models[1]


    def forward(self, *x):
        """

        :param x: Data input (for each view) ; dtype : Tuple of Tensors(n_samples_in_batch, n_features)
        :return: final_out : Decoded output from AE ; dtype : Tuple of Tensors(n_samples_in_batch, n_features)
                 hazard : "Risk ratio" ; dtype : Tensor(n_samples_in_batch, 1)
                 input data : Data input (same as x) ; dtype : Tuple of Tensors(n_samples_in_batch, n_features)
                 final_out_cross : Cross decoded output from AE ; dtype : Tuple of Tensors(n_samples_in_batch, n_features)
        """
        if 'cross' in self.type.lower():

            integrated_element, final_out_cross, final_out, input_data = self.ae(*x)
            hazard = self.nn(integrated_element)
            return final_out, final_out_cross, hazard, input_data
        else:
            integrated, final_out, input_data = self.ae(*x)
            hazard = self.nn(integrated)
            return final_out, hazard,input_data


    def predict(self,*x):
        """
        Predict function which will be used by model.predict later.
        :param x: Data input (for each view) ; dtype : Tuple of Tensors(n_samples_in_batch, n_features)
        :return: hazard : "Risk ratio" ; dtype : Tensor(n_samples_in_batch, 1)
        """
        if 'cross' in self.type.lower():

            integrated_element, final_out_cross, final_out, input_data = self.ae(*x)
            hazard = self.nn(integrated_element)
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
                 print_bool = False,
                 prelu_init = 0.25):
        """
        Autoencoder with changeable hyperparameters. Each view has an AE itself. The bottleneck representations can be
        maninpulated with different techniques (concaatenation, element-wise/overall average/minimum/maximum), the
        decoding stage can be manipulated with cross decoding, meaning that we decode the input of a specific view
        with an additional decoder. The bottleneck representation can be passed to another AE. Finally, we pass it to
        a FCNN, where it will be passed through a final layer, which compresses values to a single dimensional
        value used for the Proportional Hazards Model.

        :param views: Views (Omes) ; dtype : List of Strings
        :param in_features: Input dimensions for each view : List of Int
        :param n_hidden_layers_dims: Hidden layers for each view : List of Lists of Int
        :param activ_funcs: Activation Functions (for each view) aswell as for the last layer
        ; dtype : List of Lists of Strings ['relu', 'sigmoid', 'prelu']
        :param dropout : Probability of Neuron Dropouts ; dtype : Int
        :param dropout_layers : Layers in which to apply Dropout ; dtype : List of Lists of Strings ['yes','no']
        :param batch_norm : Layers in which to apply Batch Normalization ; dtype : List of Lists of Strings ['yes','no']
        :param dropout_bool : Decide whether Dropout is to be applied or not ; dtype : Boolean
        :param batch_norm_bool : Decide whether Batch Normalization is to be applied or not ; dtype : Boolean
        :param type_ae : Bottlneck manipulation
                         ; dtype : String ['concat','elementwisemax','elementwisemin',elementwiseavg','overallmax',
                                           'overallmin','overallavg', 'cross' or 'cross_k' where k is on of the
                                           mentionted types]
        :param cross_mutation : Choose additional decoder for view ; dtype : List of Int [Indices of decoders for view,
                                e.g [1,3,0,2] will decode features of view 0 with decoder of view 1 (and 0), features of view
                                              1 with decoder of view 3 (and 1) .. ; dtype : List or Tuple
        :param print_bool : Choose whether to print the models structure or not ; dtype : Boolean
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
        self.cross_mutation = cross_mutation
        self.prelu_init = prelu_init
        # Create list of lists which will store each hidden layer call for each view (encoding & decoding stage)
        self.hidden_layers = nn.ParameterList([nn.ParameterList([]) for x in range(len(in_features))])
        self.middle_dims = []





        # If we just input one activation function, use this activation function for each view and also the final layer
        if len(activ_funcs) == 1 and type(activ_funcs[0]) is not list:

            func = activ_funcs[0]
            activ_funcs = [[func] for x in range(len(views))] # + 1)]


        if len(activ_funcs) == len(views): #  + 1:

            for c,view in enumerate(activ_funcs):
                # If only one activ function given in sublist, use this for each layer
                if len(activ_funcs[c]) == 1 and c != len(views):
                    # -1 because we already have one activ func in our activ funcs list
                    for x in range(len(n_hidden_layers_dims[c]) -1):
                        activ_funcs[c].append(activ_funcs[c][0])

                # Replace strings with actual activation functions
                for c2,activfunc in enumerate(view):
                    if activfunc.lower() == 'relu':
                        activ_funcs[c][c2] = nn.ReLU()
                    elif activfunc.lower() == 'sigmoid':
                        activ_funcs[c][c2] = nn.Sigmoid()
                    elif activfunc.lower() == 'prelu':
                        activ_funcs[c][c2] = nn.PReLU(init=prelu_init)

        else:
            raise ValueError("Your activation function input seems to be wrong. Check if it is a list of lists with a"
                             " sublist for each view and one list for the output layer or just a single activation function"
                             " value in a list")


        # Produce hidden layer list of lists for encoding stage

        # For each view, we add the dimensions of the hidden layers for the encoder backwards to the list for the
        # decoding stage (as Decoder mirrors the Decoder and vice versa)
        # We need to do the same for activation functions, batch norm and dropout layers

        decoding_hidden = [[] for x in range(len(in_features))]
        decoding_activation = [[] for x in range(len(in_features))]
        decoding_batch = [[] for x in range(len(in_features))]
        decoding_dropout = [[] for x in range(len(in_features))]



        for c,view in enumerate(n_hidden_layers_dims):
            # Save bottleneck layer size
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

            decoding_hidden[c].append(temp_hidden)
            decoding_activation[c].append(temp_activation)
            decoding_batch[c].append(temp_batch)
            decoding_dropout[c].append(temp_dropout)
            # Concatenate temp to original list starting at first element (otherwise we would have the bottleneck
            # representation twice)
            n_hidden_layers_dims[c] += temp_hidden[1:]
            activ_funcs[c] += temp_activation[1:]
            batch_norm[c] += temp_batch[1:]
            dropout_layers[c] += temp_dropout[1:]


            # Assign Layers
            for c2 in range(len(view) + 1):
                if c2 == 0: # First layer
                    # Batch normalization
                    if batch_norm_bool == True and batch_norm[c][c2] == 'yes':
                        self.hidden_layers[c].append(nn.Sequential(nn.Linear(in_features[c],
                                                                             n_hidden_layers_dims[c][c2]),
                                                                   nn.BatchNorm1d(n_hidden_layers_dims[c][c2]),
                                                                   activ_funcs[c][c2]))
                    # No batch normalization
                    else:
                        self.hidden_layers[c].append(nn.Sequential(nn.Linear(in_features[c],
                                                                             n_hidden_layers_dims[c][c2]),
                                                                   activ_funcs[c][c2]))


                elif c2 == len(view): # Last layer
                    # Batch normalization
                    if batch_norm_bool == True and batch_norm[c][-1] == 'yes':
                        self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][-1],
                                                                             in_features[c]),
                                                                   nn.BatchNorm1d(in_features[c]),
                                                                   activ_funcs[c][-1]))
                    # No batch normalization
                    else:
                        self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][-1],
                                                                             in_features[c]),
                                                                   activ_funcs[c][-1]))



                else: # other layers
                    # Batch normalization
                    if batch_norm_bool == True and batch_norm[c][c2] == 'yes':
                        self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][c2-1],
                                                                             n_hidden_layers_dims[c][c2]),
                                                                   nn.BatchNorm1d(n_hidden_layers_dims[c][c2]),
                                                                   activ_funcs[c][c2]))
                    # No batch normalization
                    else:
                        self.hidden_layers[c].append(nn.Sequential(nn.Linear(n_hidden_layers_dims[c][c2-1],
                                                                             n_hidden_layers_dims[c][c2]),
                                                                   activ_funcs[c][c2]))


        # Mean/Max/Min output dimension element wise
        mmm_output_dimension = max(self.middle_dims)






        if type_ae.lower() == 'concat' or type_ae.lower() == 'cross_concat':
            # Dimensions of concatenated bottleneck representation

            concatenated_features = sum([dim for dim in self.middle_dims])

        if 'cross' in type_ae.lower():


            # We are only interested in the decoding stages, thus we only look at the hidden layers starting at the
            # bottleneck "layer"
            middle_pos = []
            for x in n_hidden_layers_dims:
                middle_pos.append(len(x)//2)

            # We might need a helping layer to set the dimensions of the hidden features of a view i
            # to the input dimension of the first decoding layer of view j, if we want to decode view i with the decoder
            # of view j
            self.helping_layer = nn.ParameterList(nn.ParameterList([]) for x in range(len(in_features)))
            # boolean list saving whether view needs additional helping layer or not
            self.needs_help_bool = [False for x in range(len(in_features))]

            for c, view in enumerate(n_hidden_layers_dims):
                # If the bottleneck dimensional size of view i is not the same size as first decoding layer for view j,
                # if we want to decode view i with decoder of view j... and sizes i != j
                if n_hidden_layers_dims[c][middle_pos[c]] != n_hidden_layers_dims[cross_mutation[c]][middle_pos[cross_mutation[c]]]:
                    self.needs_help_bool[c] = True
                    # In this case we need a helping layer
                    # Fit the data so it can be passed on to the chosen decoder
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

            if 'cross' in type_ae.lower():

                for c, _ in enumerate(self.helping_layer):
                    print( "For CrossAE implementation we have the following helping layers : {} for view {}"
                           .format(self.helping_layer[c], self.views[c]))
                print("Finally, for CrossAE, the output of each view between encoder and decoder  is integrated" 
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









    def forward(self,*x):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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




        if self.type_ae.lower() == 'overallmean' \
                or self.type_ae.lower() == 'overallavg' \
                or self.type_ae.lower() == 'cross_overallavg' \
                or self.type_ae.lower() == 'cross_overallmean':
            overall_mean = torch.mean(concatenated_features,1,True)

        if self.type_ae.lower() == 'overallmax' or self.type_ae.lower() == 'cross_overallmax':
            overall_max = torch.amax(concatenated_features,1,True)


        if self.type_ae.lower() == 'overallmin' or self.type_ae.lower() == 'cross_overallmin':
            overall_min = torch.amin(concatenated_features,1,True)


        # Elementwise integration needs all views to have the same amount of features in the bottleneck layer.
        hidden_sizes = []
        for idx,_ in enumerate(data_middle):
            # store sizes of each middle hidden feat dim
            hidden_sizes.append(_.size(1))
            # Take arbitrary view for size as they need to be of the same size for elementwise integration
            size = _.size(1)

            # Add dimension so we can concatenate tensors by element


        # Create copy of data_middle
        data_middle_temp = []
        for i in data_middle:
            data_middle_temp.append(torch.empty_like(i).copy_(i))
        for c,_ in enumerate(data_middle_temp):
            # Unsqueeze each view so we get single dimensional values
            data_middle_temp[c] = data_middle_temp[c].unsqueeze(2)
        # Concatenate data

        if self.type_ae.lower() == 'cross' \
                or self.type_ae.lower() == 'elementwisemean' \
                or self.type_ae.lower() == 'elementwiseavg' \
                or self.type_ae.lower() == 'cross_elementwisemean' or self.type_ae.lower() == 'cross_elementwiseavg':
            assert (len(hidden_sizes) == hidden_sizes.count(size)), "For elementwise integration, amount of features in each view for the bottleneck layer need to be of the same size."
            before_calc = torch.cat(tuple(data_middle_temp),dim=2)
            element_wise_avg = (torch.mean(before_calc,2))
        if self.type_ae.lower() == 'elementwisemax' or self.type_ae.lower() == 'cross_elementwisemax':
            assert (len(hidden_sizes) == hidden_sizes.count(size)), "For elementwise integration, amount of features in each view for the bottleneck layer need to be of the same size."
            before_calc = torch.cat(tuple(data_middle_temp),dim=2)
            element_wise_max = (torch.amax(before_calc,2))
        if self.type_ae.lower() == 'elementwisemin' or self.type_ae.lower() == 'cross_elementwisemin':
            assert (len(hidden_sizes) == hidden_sizes.count(size)), "For elementwise integration, amount of features in each view for the bottleneck layer need to be of the same size."
            before_calc = torch.cat(tuple(data_middle_temp),dim=2)
            element_wise_min = (torch.amin(before_calc,2))

        if 'cross' in self.type_ae.lower():

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
    """
    Loss function wrapper for AE & AE 2 & FCNN using a cross implementation for one AE.
    The FCNN trains on negative partial lgo likelihood loss, the AEs on
    MSE loss.
    """

    def __init__(self,alpha, decoding_bool = True, cross_position = 1):
        """

        :param alpha: Loss ratio [Negative partial log likelihood loss, MSE loss AE 1, MSE loss AE 2]
                      ; dtype : List [3 values
        :param decoding_bool : Choose whether the second AE has an decoder. Note that this should be true if
                               the cross_position is 2, because otherwise there won't be any cross-decoding ; dtype : Boolean
        :param cross_position : Choose whether first or second AE uses cross implementation
                                ; dtype : Int [1 : first AE, 2 : second AE]
        """
        super().__init__()
        assert sum(alpha), 'alpha needs to be 1 in sum'
        self.alpha = alpha
        self.loss_surv = models.loss.CoxPHLoss()
        self.loss_ae = nn.MSELoss()
        self.decoding_bool = decoding_bool
        self.cross_position = cross_position


    def forward(self, final_out_1, final_out_2, final_out_cross, hazard, integrated_element, input_data_1,duration, event):
        """
        :param final_out: Decoded output of the AE ; dtype : TupleTree of Tensor(n_samples_in_batch, n_features)
        :param final_out_cross : Cross decoded output of the AE ; dtype : TupleTree of Tensor(n_samples_in_batch, n_features)
        :param integrated_element : Input for the second AE ; dtype : List of Tensor(n_samples_in_batch, n_features)
        :param duration: Duration value ; dtype : Tensor(n_samples_in_batch,)
        :param event : Event value ; dtype : Tensor(n_samples_in_batch,)
        :param input_data: Input for the AE ; dtype : TupleTree of Tensor(n_samples_in_batch, n_features)
        :return: Combined Loss (survival (negative partial log likelihood) and sum of MSE loss (from the "normal" decoder
                 and the cross decoder) ; dtype : Float
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


        view_data = integrated_element


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
    """
    Loss function wrapper for AE & AE 2 & FCNN using a non-cross implementation.
    The FCNN trains on negative partial lgo likelihood loss, the AEs on
    MSE loss.
    """
    def __init__(self,alpha, decoding_bool = True):
        """
        :param alpha: Loss ratio [Negative partial log likelihood loss, MSE loss AE 1, MSE loss AE 2]
                      ; dtype : List [3 values
        :param decoding_bool : Choose whether the second AE has an decoder ; dtype : Boolean
        """
        super().__init__()
        assert sum(alpha), 'alpha needs to be 1 in sum'
        self.alpha = alpha
        self.loss_surv = models.loss.CoxPHLoss()
        self.loss_ae = nn.MSELoss()
        self.decoding_bool = decoding_bool


    def forward(self, final_out_1, final_out_2, hazard, view_data, input_data_1,duration, event):
        """
        :param final_out: Decoded output of the AE ; dtype : TupleTree of Tensor(n_samples_in_batch, n_features)
        :param view_data : Input for the second AE ; dtype : List of Tensor(n_samples_in_batch, n_features)
        :param duration: Duration value ; dtype : Tensor(n_samples_in_batch,)
        :param event : Event value ; dtype : Tensor(n_samples_in_batch,)
        :param input_data: Input for the AE ; dtype : TupleTree of Tensor(n_samples_in_batch, n_features)
        :return: Combined Loss (survival (negative partial log likelihood) and sum of MSE loss (from the "normal" decoder
                 and the cross decoder) ; dtype : Float
        """
        loss_surv = self.loss_surv(hazard, duration, event)
        # Final_out_1 will always have multiple view structure as it is the decoder of the first AE, which takes
        # multiple views as input //
        views_1 = len(final_out_1)
        loss_ae_1_full = 0
        # AE loss for each view
        for i in range(views_1):
            loss_ae_1 = self.loss_ae(final_out_1[i], input_data_1[i])
            loss_ae_1_full += loss_ae_1

        # View data is a list of tensors for each view ; as final_out_2 has structure of just one tensor, we
        # need to change structure accordingly
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

            return self.alpha[0] * loss_surv + self.alpha[1] * loss_ae_1_full + self.alpha[2] * loss_ae_2_full
        else:
            if len(self.alpha) != 2:
                raise Exception("Since the second AE has no decoder, alpha only contains 2 elements, not 3!")
            else:
                return self.alpha[0] * loss_surv + self.alpha[1] * loss_ae_1_full


class LossAEConcatHazard(nn.Module):
    """Loss function wrapper for AE & FCNN using a non-cross implementation.
       The FCNN trains on negative partial lgo likelihood loss, the AE on
       MSE loss."""
    def __init__(self,alpha):
        """

        :param alpha: Loss ratio [alpha negative partial log likelihood loss, 1-alpha MSE loss]
                      ; dtype : Float
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
        :param final_out: Decoded output of the AE ; dtype : TupleTree of Tensor(n_samples_in_batch, n_features)
        :param duration: Duration value ; dtype : Tensor(n_samples_in_batch,)
        :param event : Event value ; dtype : Tensor(n_samples_in_batch,)
        :param input_data: Input for the AE ; dtype : TupleTree of Tensor(n_samples_in_batch, n_features)
        :return: Combined Loss (survival (negative partial log likelihood) and MSE loss) ; dtype : Float
        """
        loss_surv = self.loss_surv(hazard, duration,event)
        views = len(final_out)
        loss_ae_full = 0
        # AE loss for each view
        for i in range(views):
            loss_ae = self.loss_ae(final_out[i], input_data[i])
            loss_ae_full += loss_ae
        return self.alpha * loss_surv + (1- self.alpha) * loss_ae_full



class LossAECrossHazard(nn.Module):
    """
    Loss function wrapper for AE & FCNN using a cross implementation.
   The FCNN trains on negative partial lgo likelihood loss, the AE on
   MSE loss.
   """
    def __init__(self,alpha):
        """

        :param alpha: Loss ratio [alpha negative partial log likelihood loss, 1-alpha MSE loss]
                      ; dtype : Float
        """
        super().__init__()
        assert (alpha >= 0) and (alpha <= 1), 'Need alpha in [0,1]'
        self.alpha = alpha
        self.loss_surv = models.loss.CoxPHLoss()
        self.loss_ae = nn.MSELoss()


    def forward(self, final_out, final_out_cross, hazard, input_data, duration, event):
        """
        :param final_out: Decoded output of the AE ; dtype : TupleTree of Tensor(n_samples_in_batch, n_features)
        :param final_out_cross : Cross decoded output of the AE ; dtype : TupleTree of Tensor(n_samples_in_batch, n_features)
        :param duration: Duration value ; dtype : Tensor(n_samples_in_batch,)
        :param event : Event value ; dtype : Tensor(n_samples_in_batch,)
        :param input_data: Input for the AE ; dtype : TupleTree of Tensor(n_samples_in_batch, n_features)
        :return: Combined Loss (survival (negative partial log likelihood) and sum of MSE loss (from the "normal" decoder
                 and the cross decoder) ; dtype : Float
        """
        loss_surv = self.loss_surv(hazard, duration,event)
        views = len(final_out)
        loss_ae_full = 0
        loss_ae_cross_full = 0

        for i in range(views):
            loss_ae = self.loss_ae(final_out[i], input_data[i])
            loss_ae_cross = self.loss_ae(final_out_cross[i], input_data[i])
            loss_ae_full += loss_ae
            loss_ae_cross_full += loss_ae_cross

        loss_ae_all = loss_ae_full + loss_ae_cross_full

        return self.alpha * loss_surv + (1- self.alpha) * loss_ae_all




def objective(trial, n_fold, t_preprocess,feature_selection_type, model_types, decoder_bool,cancer,mode):
    """
    Optuna Optimization for Hyperparameters.
    :param trial: Settings of the current trial of Hyperparameters
    :param n_fold: number of fold to be optimized ; dtype : Int
    :param t_preprocess: type of preprocess ; dtype : String
    :param feature_selection_type : type of feature selection ; dtype : String
    :param model_type : types of AE model(s) ; dtype : List of String(s)
    :param decoder_bool : decide whether second decoder will be applied ; dtype : Boolean
    :param cancer : name of the cancer used to load the data ; dtype : String
    :param mode : Decide whether to use prepared data or if we load in new data
    :return: Concordance Index ; dtype : Float
    """

    print("Fold : {}, preprocess type : {}, feature selection type : {}, model : {}, decoder : {}, cancer : {}".format(n_fold,t_preprocess,feature_selection_type,model_types,decoder_bool,cancer))
    direc_set = 'SUMO'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load in data
    preprocess_type = t_preprocess

    if mode == 'prepared_data':
        dir = os.path.expanduser('~/{}/Project/PreparedData/{}/{}/{}/'.format(direc_set,cancer,feature_selection_type,preprocess_type))
    else:
        dir = os.path.expanduser('~/{}/Project/PreparedData/'.format(direc_set)) #




    trainset_0,trainset_1,trainset_2,trainset_3,trainset_4,valset_0,valset_1,valset_2,valset_3,valset_4, \
    testset_0,testset_1,testset_2,testset_3,testset_4,trainset_feat_0, \
    trainset_feat_1,trainset_feat_2,trainset_feat_3,trainset_feat_4 = load_data(data_dir = dir)



    # Feature offsets need to be the same in train/val/test for each fold, otherwise NN wouldn't work (diff dimension inputs)
    feat_offs = [trainset_feat_0,trainset_feat_1,trainset_feat_2,trainset_feat_3,trainset_feat_4]


    for c2,_ in enumerate(feat_offs):
        feat_offs[c2] = list(feat_offs[c2].values)
        for idx,_ in enumerate(feat_offs[c2]):
            feat_offs[c2][idx] = feat_offs[c2][idx].item()


    # Split data in feature values, duration, event

    trainset = [trainset_0 ,trainset_1,trainset_2,trainset_3,trainset_4]
    valset = [valset_0 ,valset_1,valset_2,valset_3,valset_4]
    testset = [testset_0,testset_1,testset_2,testset_3,testset_4]
    n_folds = len(trainset)
    train_data_folds = []
    train_duration_folds = []
    train_event_folds = []
    val_data_folds = []
    val_duration_folds = []
    val_event_folds = []
    test_data_folds = []

    # LOAD IN DATA
    for c2,_ in enumerate(trainset):
        train_data = []
        for c,feat in enumerate(feat_offs[c2]):
            if c < len(feat_offs[c2]) - 5: # train data views # CHANGED -3 to -5  TO NOT LOOK AT microRNA and RPPA
                data_np = np.array((trainset[c2].iloc[:, feat_offs[c2][c] : feat_offs[c2][c+1]]).values).astype('float32')
                data_tensor = torch.from_numpy(data_np).to(torch.float32)
                data_tensor = data_tensor.to(device)
                train_data.append(data_tensor)
            elif c == len(feat_offs[c2]) - 3: # duration
                duration_np = (np.array((trainset[c2].iloc[:, feat_offs[c2][c] : feat_offs[c2][c+1]]).values).astype('float32')).squeeze(axis=1)
                duration_tensor = torch.from_numpy(duration_np).to(torch.float32)
                duration_tensor = duration_tensor.to(device)
                train_duration = duration_tensor

            elif c == len(feat_offs[c2]) -2: # event
                event_np = (np.array((trainset[c2].iloc[:, feat_offs[c2][c] : feat_offs[c2][c+1]]).values).astype('float32')).squeeze(axis=1)
                event_tensor = torch.from_numpy(event_np).to(torch.float32)
                event_tensor = event_tensor.to(device)
                train_event = event_tensor

        train_data = tuple(train_data)

        val_data = []
        for c,feat in enumerate(feat_offs[c2]):
            if c < len(feat_offs[c2]) - 5: # train data views # CHANGED -3 to -5  TO NOT LOOK AT microRNA and RPPA
                data_np = np.array((valset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')
                data_tensor = torch.from_numpy(data_np).to(torch.float32)
                data_tensor = data_tensor.to(device)
                val_data.append(data_tensor)

            elif c == len(feat_offs[c2]) - 3: # duration
                duration_np = (np.array((valset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')).squeeze(axis=1)
                duration_tensor = torch.from_numpy(duration_np).to(torch.float32)
                duration_tensor = duration_tensor.to(device)
                val_duration = duration_tensor

            elif c == len(feat_offs[c2]) -2: # event
                event_np = (np.array((valset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')).squeeze(axis=1)
                event_tensor = torch.from_numpy(event_np).to(torch.float32)
                event_tensor = event_tensor.to(device)
                val_event = event_tensor



        test_data = []
        for c,feat in enumerate(feat_offs[c2]):
            if c < len(feat_offs[c2]) - 5: # train data views # CHANGED -3 to -5  TO NOT LOOK AT microRNA and RPPA
                data_np = np.array((testset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')
                data_tensor = torch.from_numpy(data_np).to(torch.float32)
                data_tensor = data_tensor.to(device)
                test_data.append(data_tensor)

            elif c == len(feat_offs[c2]) - 3: # duration
                duration_np = (np.array((testset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')).squeeze(axis=1)
                duration_tensor = torch.from_numpy(duration_np).to(torch.float32)
                duration_tensor = duration_tensor.to(device)
                test_duration = duration_tensor

            elif c == len(feat_offs[c2]) -2: # event
                event_np = (np.array((testset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')).squeeze(axis=1)
                event_tensor = torch.from_numpy(event_np).to(torch.float32)
                event_tensor = event_tensor.to(device)
                test_event = event_tensor

        train_data_folds.append(train_data)
        val_data_folds.append(val_data)
        train_duration_folds.append(train_duration)
        val_duration_folds.append(val_duration)
        train_event_folds.append(train_event)
        val_event_folds.append(val_event)
        test_data_folds.append(test_data)

    # Rename so we have same structure as in train function
    train_data = train_data_folds
    train_duration = train_duration_folds
    train_event = train_event_folds
    val_data = val_data_folds
    val_duration = val_duration_folds
    val_event = val_event_folds
    test_data = test_data_folds


    views = []
    dir = os.path.expanduser('~/{}/Project/TCGAData/cancerviews.txt'.format(direc_set))
    # dir = os.path.expanduser('/Users/marlon/Desktop/Project/TCGAData/cancerviews.txt')
    read_in = open(dir, 'r')
    for view in read_in:
        views.append(view)

    view_names = [line[:-1] for line in views]

    ###### TESTING
    view_names = ['MRNA','DNA']



    second_decoder_bool = decoder_bool
    # Current fold to be optimized
    c_fold = n_fold
    #print("Fold:", c_fold, " model type : ", model_types, "second decoder :", second_decoder_bool, " Preprocess Type: ", preprocess_type)
    ##################################### HYPERPARAMETER SEARCH SETTINGS ##############################################
    l2_regularization_bool = trial.suggest_categorical('l2_regularization_bool', [True,False])
    learning_rate = trial.suggest_float("learning_rate", 1e-5,1e-1,log=True)
    l2_regularization_rate = trial.suggest_float("l2_regularization_rate", 1e-6,1e-3, log=True)
    #  batch_size = trial.suggest_int("batch_size", 5, 200)
    batch_size = trial.suggest_categorical("batch_size", [7,17,33,64,128,256])
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
        layers_1_mRNA = trial.suggest_categorical('layers_1_mRNA', [32,64,96,128])
        layers_2_mRNA = trial.suggest_categorical('layers_2_mRNA', [8,16,32])

        layers_1_mRNA_activfunc = trial.suggest_categorical('layers_1_mRNA_activfunc', ['relu','sigmoid','prelu'])
        layers_2_mRNA_activfunc = trial.suggest_categorical('layers_2_mRNA_activfunc', ['relu','sigmoid','prelu'])
        layers_1_mRNA_dropout = trial.suggest_categorical('layers_1_mRNA_dropout', ['yes','no'])
        layers_2_mRNA_dropout = trial.suggest_categorical('layers_2_mRNA_dropout', ['yes','no'])
        layers_1_mRNA_batchnorm = trial.suggest_categorical('layers_1_mRNA_batchnorm', ['yes', 'no'])
        layers_2_mRNA_batchnorm = trial.suggest_categorical('layers_2_mRNA_batchnorm', ['yes', 'no'])

        """
        if len(model_types) == 2 and model_types[0] == 'none':
            layers_1_mRNA_hierarichcal = trial.suggest_int('layers_1_mRNA_hierarichcal', 32, 96)
            layers_2_mRNA_hierarichcal = trial.suggest_int('layers_2_mRNA_hierarichcal', 8, 32)
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
        """


        layers.append([layers_1_mRNA,layers_2_mRNA])
        activation_functions.append([layers_1_mRNA_activfunc, layers_2_mRNA_activfunc])
        dropouts.append([layers_1_mRNA_dropout, layers_2_mRNA_dropout])
        batchnorms.append([layers_1_mRNA_batchnorm, layers_2_mRNA_batchnorm])

    #JUMPER1
    if 'DNA' in view_names:
        layers_1_DNA = trial.suggest_categorical('layers_1_DNA', [32,64,96,128])
        if 'concat' in model_types[0] or 'overall' in model_types[0]:
            layers_2_DNA = trial.suggest_categorical('layers_2_DNA',[8,16,32])
        if 'elementwise' in model_types[0]:
            layers_2_DNA = layers_2_mRNA
        layers_1_DNA_activfunc = trial.suggest_categorical('layers_1_DNA_activfunc', ['relu','sigmoid','prelu'])
        layers_2_DNA_activfunc = trial.suggest_categorical('layers_2_DNA_activfunc', ['relu','sigmoid','prelu'])
        layers_1_DNA_dropout = trial.suggest_categorical('layers_1_DNA_dropout', ['yes','no'])
        layers_2_DNA_dropout = trial.suggest_categorical('layers_2_DNA_dropout', ['yes','no'])
        layers_1_DNA_batchnorm = trial.suggest_categorical('layers_1_DNA_batchnorm', ['yes', 'no'])
        layers_2_DNA_batchnorm = trial.suggest_categorical('layers_2_DNA_batchnorm', ['yes', 'no'])
        """
        if len(model_types) == 2 and model_types[0] == 'none':
            layers_1_DNA_hierarichcal = trial.suggest_int('layers_1_DNA_hierarichcal', 8, 16)
            layers_2_DNA_hierarichcal = trial.suggest_int('layers_2_DNA_hierarichcal', 4, 8)
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
        """

        layers.append([layers_1_DNA,layers_2_DNA])
        activation_functions.append([layers_1_DNA_activfunc, layers_2_DNA_activfunc])
        dropouts.append([layers_1_DNA_dropout, layers_2_DNA_dropout])
        batchnorms.append([layers_1_DNA_batchnorm, layers_2_DNA_batchnorm])


    if 'MICRORNA' in view_names:
        layers_1_microRNA = trial.suggest_categorical('layers_1_microRNA', [32,64,96,128])
        if 'concat' in model_types[0] or 'overall' in model_types[0]:
            layers_2_microRNA = trial.suggest_categorical('layers_2_microRNA',[8,16,32])
        if 'elementwise' in model_types[0]:
            layers_2_microRNA = layers_2_mRNA
        layers_1_microRNA_activfunc = trial.suggest_categorical('layers_1_microRNA_activfunc', ['relu','sigmoid','prelu'])
        layers_2_microRNA_activfunc = trial.suggest_categorical('layers_2_microRNA_activfunc', ['relu','sigmoid','prelu'])
        layers_1_microRNA_dropout = trial.suggest_categorical('layers_1_microRNA_dropout', ['yes','no'])
        layers_2_microRNA_dropout = trial.suggest_categorical('layers_2_microRNA_dropout', ['yes','no'])
        layers_1_microRNA_batchnorm = trial.suggest_categorical('layers_1_microRNA_batchnorm', ['yes', 'no'])
        layers_2_microRNA_batchnorm = trial.suggest_categorical('layers_2_microRNA_batchnorm', ['yes', 'no'])
        """
        if len(model_types) == 2 and model_types[0] == 'none':
            layers_1_microRNA_hierarichcal = trial.suggest_int('layers_1_microRNA_hierarichcal', 8, 16)
            layers_2_microRNA_hierarichcal = trial.suggest_int('layers_2_microRNA_hierarichcal', 4, 8)
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
        """
        layers.append([layers_1_microRNA,layers_2_microRNA])
        activation_functions.append([layers_1_microRNA_activfunc, layers_2_microRNA_activfunc])
        dropouts.append([layers_1_microRNA_dropout, layers_2_microRNA_dropout])
        batchnorms.append([layers_1_microRNA_batchnorm, layers_2_microRNA_batchnorm])


    if 'RPPA' in view_names:
        layers_1_RPPA = trial.suggest_int('layers_1_RPPA', 64, 96)
        layers_2_RPPA = trial.suggest_int('layers_2_RPPA', 16, 32)
        layers_1_RPPA_activfunc = trial.suggest_categorical('layers_1_RPPA_activfunc', ['relu','sigmoid','prelu'])
        layers_2_RPPA_activfunc = trial.suggest_categorical('layers_2_RPPA_activfunc', ['relu','sigmoid','prelu'])
        layers_1_RPPA_dropout = trial.suggest_categorical('layers_1_RPPA_dropout', ['yes','no'])
        layers_2_RPPA_dropout = trial.suggest_categorical('layers_2_RPPA_dropout', ['yes','no'])
        layers_1_RPPA_batchnorm = trial.suggest_categorical('layers_1_RPPA_batchnorm', ['yes', 'no'])
        layers_2_RPPA_batchnorm = trial.suggest_categorical('layers_2_RPPA_batchnorm', ['yes', 'no'])
        """
        if len(model_types) == 2 and model_types[0] == 'none':
            layers_1_RPPA_hierarichcal = trial.suggest_int('layers_1_RPPA_hierarichcal', 8, 16)
            layers_2_RPPA_hierarichcal = trial.suggest_int('layers_2_RPPA_hierarichcal', 4, 8)
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
        """

        layers.append([layers_1_RPPA,layers_2_RPPA])
        activation_functions.append([layers_1_RPPA_activfunc, layers_2_RPPA_activfunc])
        dropouts.append([layers_1_RPPA_dropout, layers_2_RPPA_dropout])
        batchnorms.append([layers_1_RPPA_batchnorm, layers_2_RPPA_batchnorm])












    # for cross implementation
    if len(model_types) == 2:
        if len(layers) == 3 and ('cross' in model_types[0] or 'cross' in model_types[1]):
            cross_decoders_3_views = trial.suggest_categorical("cross_decoders_3_views",[(0,2,1),(2,1,0),(1,2,0),(1,0,2),(2,0,1)])

        if len(layers) == 2 and ('cross' in model_types[0] or 'cross' in model_types[1]):
            # No suggesting since for 2 views we only have one cross possibilty :
            # Decode view 0 with view 1 decoder and vice versa
            cross_decoders_2_views = (1,0)

    else:
        if len(layers) == 3 and ('cross' in model_types[0]):
            cross_decoders_3_views = trial.suggest_categorical("cross_decoders_3_views",[(0,2,1),(2,1,0),(1,2,0),(1,0,2),(2,0,1)])

        if len(layers) == 2 and ('cross' in model_types[0]):
            # No suggesting since for 2 views we only have one cross possibilty :
            # Decode view 0 with view 1 decoder and vice versa
            cross_decoders_2_views = (1,0)



    #  print("MODEL TYPES : ", model_types)

    # for concatenated data input into FCNN (no hierarichcal AE)
    out_sizes = []
    for c_layer in range(len(layers)):
        out_sizes.append(layers[c_layer][-1])

    in_feats_second_NN_concat = [sum(out_sizes)]

    # for element-wise avg/min/max input into FCNN (no hierarichcal AE)

    # Size is the one of the largest output dim of AE layers
    in_feats_second_NN_element_wise = [max([i[-1] for i in layers])]



    # After concat FCNN (concatenated input into FCNN)
    #JUMPER1
    # This is needed in nearly every case, just not if we don't integrate the data in the autoencoders in any way
    # (basically just let AE work as feature selection method)
    if len(model_types) == 1:
        if 'concat' in model_types[0]:
            size_of_fcnn_layer_1 = sum([layers_2_mRNA,layers_2_DNA])
        if 'elementwise' in model_types[0]:
            size_of_fcnn_layer_1 = max(layers_2_mRNA,layers_2_DNA)
        if 'overall' in model_types[0]:
            size_of_fcnn_layer_1 = 1
    if len(model_types) == 2:
        #in this case, the second AE always works on overall settings
        size_of_fcnn_layer_1 = 1
    #   layers_1_FCNN = size_of_overall
    layers_1_FCNN = size_of_fcnn_layer_1
    #layers_1_FCNN = trial.suggest_int('layers_1_FCNN', size_of_overall,size_of_overall)
    # layers_2_FCNN = trial.suggest_int('layers_2_FCNN', 8,size_of_fcnn_layer_1)

    layers_FCNN = [[layers_1_FCNN]]#,layers_2_FCNN]]

    layers_1_FCNN_activfunc = trial.suggest_categorical('layers_1_FCNN_activfunc', ['relu','sigmoid','prelu'])
    # layers_2_FCNN_activfunc = trial.suggest_categorical('layers_2_FCNN_activfunc', ['relu','sigmoid','prelu'])

    FCNN_activation_functions = [layers_1_FCNN_activfunc]#,layers_2_FCNN_activfunc]

    FCNN_dropout_prob = trial.suggest_float("FCNN_dropout_prob", 0,0.5,step=0.1)
    FCNN_dropout_bool = trial.suggest_categorical('FCNN_dropout_bool', [True,False])
    FCNN_batchnorm_bool = trial.suggest_categorical('FCNN_batchnorm_bool',[True,False])

    layers_1_FCNN_dropout = trial.suggest_categorical('layers_1_FCNN_dropout', ['yes','no'])
    # layers_2_FCNN_dropout = trial.suggest_categorical('layers_2_FCNN_dropout', ['yes','no'])

    FCNN_dropouts = [layers_1_FCNN_dropout]#, layers_2_FCNN_dropout]


    layers_1_FCNN_batchnorm = trial.suggest_categorical('layers_1_FCNN_batchnorm', ['yes', 'no'])
    # layers_2_FCNN_batchnorm = trial.suggest_categorical('layers_2_FCNN_batchnorm', ['yes', 'no'])

    FCNN_batchnorms = [layers_1_FCNN_batchnorm]#,layers_2_FCNN_batchnorm]


    if len(model_types) == 2 and model_types[0] != 'none':
        # In this case we already have done integration method and have single omic data structure left
        if 'concat' in model_types[0]:
            feats_in = sum(out_sizes)
        if 'elementwise' in model_types[0]:
            # can be the last layer of aribtrary view from the first AE, as they need to have the same size
            # for elementwise integration
            feats_in = layers_2_DNA

        layers_1_hierarichcal_integrated = feats_in # trial.suggest_int('layers_1_hierarichcal_integrated', feats_in, feats_in)
        layers_2_hierarichcal_integrated = trial.suggest_int('layers_2_hierarichcal_integrated',8, feats_in)
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







    # Note that for the Cross Method, the cross_mutation must be set in a way so that the input view x and
    # "crossed" view y have the same feature size : Otherwise, the MSELoss() cant be calculated
    # e.g. 4 views having feature sizes [15,15,5,5] --> cross mutations between first two and last two work,
    # but not e.g. between first and third/fourth

    # AE's





    # Create List of models to be used
    all_models = nn.ModuleList()


    dimensions_train = [x.shape[1] for x in train_data[c_fold]]
    dimensions_val = [x.shape[1] for x in val_data[c_fold]]
    dimensions_test = [x.shape[1] for x in test_data[c_fold]]

    assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/test'

    dimensions = dimensions_train

    # Transforms for PyCox
    train_surv = (train_duration[c_fold], train_event[c_fold])
    val_data_full = (val_data[c_fold], (val_duration[c_fold], val_event[c_fold]))


    # layers_u = copy.deepcopy(layers)
    # activation_functions_u = copy.deepcopy(activation_functions)
    # dropouts_u = copy.deepcopy(dropouts)
    # batchnorms_u = copy.deepcopy(batchnorms)
    if "cross" in model_types[0]:
        if len(layers) == 2:
            cross_setting = cross_decoders_2_views
        elif len(layers) == 3:
            cross_setting = cross_decoders_3_views
    else:
        cross_setting = None

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
                         cross_mutation=cross_setting,
                         print_bool=False,
                         prelu_init= prelu_rate)).to(device)

    # Feature dimension are the input dimensions for the FCNN, feature dimensions h(ierachical) for the second AE
    if len(model_types) == 1:
        if "overall" in model_types[0]:
            feature_dimension_FCNN = [1]
        elif "elementwise" in model_types[0]:
            feature_dimension_FCNN = in_feats_second_NN_element_wise
        elif "concat" in model_types[0]:
            feature_dimension_FCNN = in_feats_second_NN_concat

    if len(model_types) == 2:
        if "elementwise" in model_types[0] and "overall" in model_types[1]:
            feature_dimension_h = in_feats_second_NN_element_wise
            feature_dimension_FCNN = [1]
        #  feature_dimension = in_feats_third_NN_elementwise_hierachical
        if "concat" in model_types[0] and "overall" in model_types[1]:
            feature_dimension_h = in_feats_second_NN_concat
            feature_dimension_FCNN = [1]
        #   feature_dimension = in_feats_third_NN_concat_hierachical

    # Hierachical AE no integration : in_features must be sizes of bottleneck dimensions of first AE

    if len(model_types) == 2:
        all_models.append(AE(views = ['AE'],
                             in_features= feature_dimension_h,
                             n_hidden_layers_dims= layers_hierarchical_integrated,
                             activ_funcs=activation_functions_hierarchical_integrated,
                             dropout_bool= dropout_bool_hierachical_integrated,
                             dropout_prob= dropout_prob_hierachical_integrated,
                             dropout_layers= dropouts_hierarchical_integrated,
                             batch_norm_bool= batchnorm_bool_hierachical_integrated,
                             batch_norm= batchnorms_hierarchical_integrated,
                             type_ae=model_types[1],
                             cross_mutation=None,
                             print_bool=False,
                             prelu_init= prelu_rate)).to(device)


    #JUMPER1 (layer size)
    #JUMPERFIN

    all_models.append(FCNN.NN_changeable(views = ['AE'],
                                         in_features = feature_dimension_FCNN,
                                         n_hidden_layers_dims= layers_FCNN,
                                         activ_funcs = [FCNN_activation_functions,['none']],
                                         dropout_prob=FCNN_dropout_prob,
                                         dropout_layers=FCNN_dropouts,
                                         batch_norm = FCNN_batchnorms,
                                         dropout_bool=FCNN_dropout_bool,
                                         batch_norm_bool=FCNN_batchnorm_bool,
                                         print_bool=False,
                                         prelu_init = prelu_rate)).to(device)



    if len(model_types) == 2:
        full_net = AE_Hierarichal(all_models, types=model_types)

    if len(model_types) == 1:
        full_net = AE_NN(all_models, type=model_types[0])


    # set optimizer
    if l2_regularization_bool == True:
        optimizer = Adam(full_net.parameters(), lr=learning_rate, weight_decay=l2_regularization_rate)
    else:
        optimizer = Adam(full_net.parameters(), lr=learning_rate)

    callbacks = [tt.callbacks.EarlyStopping(patience=10)]
    #   cross_pos = model_types.index("cross") + 1
    #   model = models.CoxPH(full_net,optimizer, loss=LossHierarichcalAESingleCross(alpha=[0.4,0.4,0.2], decoding_bool=True, cross_position=cross_pos)) # Change Loss here


    if len(model_types) == 1:
        if "cross" in model_types[0]:
            loss_func = LossAECrossHazard(loss_surv)
        else:
            loss_func = LossAEConcatHazard(loss_surv)


    # if second decoder, we need to use loss_3 ; else normal 2 valued loss
    if len(model_types) == 2:
        if second_decoder_bool == False:
            loss_val  = [loss_surv, 1-loss_surv]
            if "cross" in model_types[0]:
                loss_func = LossHierarichcalAESingleCross(loss_val, cross_position=1,decoding_bool=second_decoder_bool)
            else:
                loss_func = LossHierarichcalAE(loss_val,decoding_bool=second_decoder_bool)
        else:
            loss_val = loss_3_values_hierarchical
            if "cross" in model_types[0]:
                loss_func = LossHierarichcalAESingleCross(loss_val, cross_position=1,decoding_bool=second_decoder_bool)
            else:
                loss_func = LossHierarichcalAE(loss_val,decoding_bool=second_decoder_bool)


        # decoding bool False : loss = [loss_surv, 1-loss_surv] ; else loss_3_values_hierarchical
    #   loss_hierachical_no_cross = LossHierarichcalAE(loss_3_values_hierarchical,decoding_bool=second_decoder_bool)
    # loss : alpha * surv_loss + (1-alpha) * ae_loss
    model = models.CoxPH(full_net,
                         optimizer,
                         loss=loss_func)
    model.set_device(torch.device(device))
    print_loss = False

    log = model.fit(train_data[c_fold],
                    train_surv,
                    batch_size,
                    n_epochs,
                    verbose=print_loss,
                    val_data= val_data_full,
                    val_batch_size= batch_size,
                    callbacks=callbacks)

    # Plot it
    #   _ = log.plot()

    # Change for EvalSurv-Function
    try:
        test_duration = test_duration.cpu().detach().numpy()
        test_event = test_event.cpu().detach().numpy()
    except AttributeError:
        pass


    for c,fold in enumerate(train_data):
        try:
            train_duration[c_fold] = train_duration[c_fold].cpu().detach().numpy()
            train_event[c_fold] = train_event[c_fold].cpu().detach().numpy()
            val_duration[c_fold] = val_duration[c_fold].cpu().detach().numpy()
            val_event[c_fold] = val_event[c_fold].cpu().detach().numpy()
        except AttributeError: # in this case already numpy arrays
            pass

    # Since Cox semi parametric, we calculate a baseline hazard to introduce a time variable
    _ = model.compute_baseline_hazards()

    # Predict based on validation data
    surv = model.predict_surv_df(val_data[c_fold])

    # Plot it
    #      surv.iloc[:, :5].plot()
    #      plt.ylabel('S(t | x)')
    #      _ = plt.xlabel('Time')




    # Evaluate with concordance, brier score and binomial log-likelihood
    ev = EvalSurv(surv, val_duration[c_fold], val_event[c_fold], censor_surv='km') # censor_surv : Kaplan-Meier

    # concordance
    concordance_index = ev.concordance_td()

    if concordance_index < 0.5:
        concordance_index = 1 - concordance_index

    # Can also be optimized with brier score or binomial log-likelihood
    #brier score
    #  time_grid = np.linspace(test_duration.min(), test_duration.max(), 100)
    ##  _ = ev.brier_score(time_grid).plot
    #  brier_score = ev.integrated_brier_score(time_grid)

    #binomial log-likelihood
    #   binomial_score = ev.integrated_nbll(time_grid)


    #  # SAVING MODEL POSSIBILITY
    # dir = os.path.expanduser(r'~/SUMO/Project/Trial/Models/Fold_{}_Trial_{}'.format(c_fold,trial.number))

    # torch.save(net,dir)


    return concordance_index





def test_model(n_fold,t_preprocess,feature_selection_type,model_types,decoder_bool,cancer):
    """Function to test the model on optimized hyperparameter settings.
    :param n_fold : Number of the fold to test ; dtype : Int
    :param t_preprocess : Type of preprocessing ; dtype : String
    :param feature_selection_type : Feature selection type ; dtype : String
    :param model_types : Model types of the AE ; dtype : List of Strings
    :param decoder_bool : Choose whether second decoder was applied ; dtype : Boolean
    :param cancer : Name of the cancer folder ; dtype : String"""
    direc_set = 'SUMO'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load in data
    preprocess_type = t_preprocess


    dir = os.path.expanduser('~/{}/Project/PreparedData/{}/{}/{}/'.format(direc_set,cancer,feature_selection_type,preprocess_type))


    trainset_0,trainset_1,trainset_2,trainset_3,trainset_4,valset_0,valset_1,valset_2,valset_3,valset_4, \
    testset_0,testset_1,testset_2,testset_3,testset_4,trainset_feat_0, \
    trainset_feat_1,trainset_feat_2,trainset_feat_3,trainset_feat_4 = load_data(data_dir = dir)



    # Feature offsets need to be the same in train/val/test for each fold, otherwise NN wouldn't work (diff dimension inputs)
    feat_offs = [trainset_feat_0,trainset_feat_1,trainset_feat_2,trainset_feat_3,trainset_feat_4]


    for c2,_ in enumerate(feat_offs):
        feat_offs[c2] = list(feat_offs[c2].values)
        for idx,_ in enumerate(feat_offs[c2]):
            feat_offs[c2][idx] = feat_offs[c2][idx].item()


    # Split data in feature values, duration, event

    trainset = [trainset_0 ,trainset_1,trainset_2,trainset_3,trainset_4]
    valset = [valset_0 ,valset_1,valset_2,valset_3,valset_4]
    testset = [testset_0,testset_1,testset_2,testset_3,testset_4]
    n_folds = len(trainset)
    train_data_folds = []
    train_duration_folds = []
    train_event_folds = []
    val_data_folds = []
    val_duration_folds = []
    val_event_folds = []
    test_data_folds = []

    # LOAD IN DATA
    for c2,_ in enumerate(trainset):
        train_data = []
        for c,feat in enumerate(feat_offs[c2]):
            if c < len(feat_offs[c2]) - 5: # train data views # CHANGED -3 to -5  TO NOT LOOK AT microRNA and RPPA
                data_np = np.array((trainset[c2].iloc[:, feat_offs[c2][c] : feat_offs[c2][c+1]]).values).astype('float32')
                data_tensor = torch.from_numpy(data_np).to(torch.float32)
                data_tensor = data_tensor.to(device)
                train_data.append(data_tensor)
            elif c == len(feat_offs[c2]) - 3: # duration
                duration_np = (np.array((trainset[c2].iloc[:, feat_offs[c2][c] : feat_offs[c2][c+1]]).values).astype('float32')).squeeze(axis=1)
                duration_tensor = torch.from_numpy(duration_np).to(torch.float32)
                duration_tensor = duration_tensor.to(device)
                train_duration = duration_tensor

            elif c == len(feat_offs[c2]) -2: # event
                event_np = (np.array((trainset[c2].iloc[:, feat_offs[c2][c] : feat_offs[c2][c+1]]).values).astype('float32')).squeeze(axis=1)
                event_tensor = torch.from_numpy(event_np).to(torch.float32)
                event_tensor = event_tensor.to(device)
                train_event = event_tensor

        train_data = tuple(train_data)

        val_data = []
        for c,feat in enumerate(feat_offs[c2]):
            if c < len(feat_offs[c2]) - 5: # train data views # CHANGED -3 to -5  TO NOT LOOK AT microRNA and RPPA
                data_np = np.array((valset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')
                data_tensor = torch.from_numpy(data_np).to(torch.float32)
                data_tensor = data_tensor.to(device)
                val_data.append(data_tensor)

            elif c == len(feat_offs[c2]) - 3: # duration
                duration_np = (np.array((valset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')).squeeze(axis=1)
                duration_tensor = torch.from_numpy(duration_np).to(torch.float32)
                duration_tensor = duration_tensor.to(device)
                val_duration = duration_tensor

            elif c == len(feat_offs[c2]) -2: # event
                event_np = (np.array((valset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')).squeeze(axis=1)
                event_tensor = torch.from_numpy(event_np).to(torch.float32)
                event_tensor = event_tensor.to(device)
                val_event = event_tensor



        test_data = []
        for c,feat in enumerate(feat_offs[c2]):
            if c < len(feat_offs[c2]) - 5: # train data views # CHANGED -3 to -5  TO NOT LOOK AT microRNA and RPPA
                data_np = np.array((testset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')
                data_tensor = torch.from_numpy(data_np).to(torch.float32)
                data_tensor = data_tensor.to(device)
                test_data.append(data_tensor)

            elif c == len(feat_offs[c2]) - 3: # duration
                duration_np = (np.array((testset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')).squeeze(axis=1)
                duration_tensor = torch.from_numpy(duration_np).to(torch.float32)
                duration_tensor = duration_tensor.to(device)
                test_duration = duration_tensor

            elif c == len(feat_offs[c2]) -2: # event
                event_np = (np.array((testset[c2].iloc[:, feat_offs[c2][c]: feat_offs[c2][c + 1]]).values).astype('float32')).squeeze(axis=1)
                event_tensor = torch.from_numpy(event_np).to(torch.float32)
                event_tensor = event_tensor.to(device)
                test_event = event_tensor

        train_data_folds.append(train_data)
        val_data_folds.append(val_data)
        train_duration_folds.append(train_duration)
        val_duration_folds.append(val_duration)
        train_event_folds.append(train_event)
        val_event_folds.append(val_event)
        test_data_folds.append(test_data)

    # Rename so we have same structure as in train function
    train_data = train_data_folds
    train_duration = train_duration_folds
    train_event = train_event_folds
    val_data = val_data_folds
    val_duration = val_duration_folds
    val_event = val_event_folds
    test_data = test_data_folds


    views = []
    dir = os.path.expanduser('~/{}/Project/TCGAData/cancerviews.txt'.format(direc_set))
    # dir = os.path.expanduser('/Users/marlon/Desktop/Project/TCGAData/cancerviews.txt')
    read_in = open(dir, 'r')
    for view in read_in:
        views.append(view)

    view_names = [line[:-1] for line in views]

    ###### TESTING
    view_names = ['MRNA','DNA']



    second_decoder_bool = decoder_bool
    # Current fold to be optimized
    c_fold = n_fold

    # Create List of models to be used
    all_models = nn.ModuleList()


    dimensions_train = [x.shape[1] for x in train_data[c_fold]]
    dimensions_val = [x.shape[1] for x in val_data[c_fold]]
    dimensions_test = [x.shape[1] for x in test_data[c_fold]]

    assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/test'

    dimensions = dimensions_train

    # Transforms for PyCox
    train_surv = (train_duration[c_fold], train_event[c_fold])
    val_data_full = (val_data[c_fold], (val_duration[c_fold], val_event[c_fold]))


    # layers_u = copy.deepcopy(layers)
    # activation_functions_u = copy.deepcopy(activation_functions)
    # dropouts_u = copy.deepcopy(dropouts)
    # batchnorms_u = copy.deepcopy(batchnorms)
    if "cross" in model_types[0]:
        cross_setting = (1,0)

    else:
        cross_setting = None

    params={'l2_regularization_bool': False, 'learning_rate': 0.00139967793982214, 'l2_regularization_rate': 1.6487875319869825e-06, 'batch_size': 17, 'dropout_prob': 0.30000000000000004, 'dropout_bool': False, 'batchnorm_bool': False, 'prelu_rate': 0.35000000000000003, 'loss_surv': 0.231493808823674, 'layers_1_mRNA': 64, 'layers_2_mRNA': 32, 'layers_1_mRNA_activfunc': 'sigmoid', 'layers_2_mRNA_activfunc': 'prelu', 'layers_1_mRNA_dropout': 'no', 'layers_2_mRNA_dropout': 'yes', 'layers_1_mRNA_batchnorm': 'no', 'layers_2_mRNA_batchnorm': 'yes', 'layers_1_DNA': 64, 'layers_1_DNA_activfunc': 'sigmoid', 'layers_2_DNA_activfunc': 'relu', 'layers_1_DNA_dropout': 'yes', 'layers_2_DNA_dropout': 'no', 'layers_1_DNA_batchnorm': 'no', 'layers_2_DNA_batchnorm': 'no', 'layers_2_FCNN': 24, 'layers_1_FCNN_activfunc': 'relu', 'layers_2_FCNN_activfunc': 'prelu', 'FCNN_dropout_prob': 0.2, 'FCNN_dropout_bool': False, 'FCNN_batchnorm_bool': True, 'layers_1_FCNN_dropout': 'yes', 'layers_2_FCNN_dropout': 'no', 'layers_1_FCNN_batchnorm': 'yes', 'layers_2_FCNN_batchnorm': 'no'}


    layers = [[params['layers_1_mRNA'],params['layers_2_mRNA']],[params['layers_1_DNA'],params['layers_2_mRNA']]]
    out_sizes = []
    for c_layer in range(len(layers)):
        out_sizes.append(layers[c_layer][-1])

    in_feats_second_NN_concat = [sum(out_sizes)]

    # for element-wise avg/min/max input into FCNN (no hierarichcal AE)

    # Size is the one of the largest output dim of AE layers
    in_feats_second_NN_element_wise = [max([i[-1] for i in layers])]

    all_models.append(AE(views = view_names,
                         in_features= dimensions,
                         n_hidden_layers_dims= layers,
                         activ_funcs=[[params['layers_1_mRNA_activfunc'], params['layers_2_mRNA_activfunc']],
                                      [params['layers_1_DNA_activfunc'], params['layers_2_DNA_activfunc']]],
                         dropout_bool= params['dropout_bool'],
                         dropout_prob= params['dropout_prob'],
                         dropout_layers= [[params['layers_1_mRNA_dropout'],params['layers_2_mRNA_dropout']],
                                          [params['layers_1_DNA_dropout'],params['layers_2_DNA_dropout']]],
                         batch_norm_bool= params['batchnorm_bool'],
                         batch_norm= [[params['layers_1_mRNA_batchnorm'], params['layers_2_mRNA_batchnorm']],
                                      [params['layers_1_DNA_batchnorm'], params['layers_2_DNA_batchnorm']]],
                         type_ae=model_types[0],
                         cross_mutation=cross_setting,
                         print_bool=False,
                         prelu_init= params['prelu_rate'])).to(device)

    # for concatenated data input into FCNN (no hierarichcal AE)

    # Feature dimension are the input dimensions for the FCNN, feature dimensions h(ierachical) for the second AE
    if len(model_types) == 1:
        if "overall" in model_types[0]:
            feature_dimension_FCNN = [1]
        elif "elementwise" in model_types[0]:
            feature_dimension_FCNN = in_feats_second_NN_element_wise
        elif "concat" in model_types[0]:
            feature_dimension_FCNN = in_feats_second_NN_concat


    if len(model_types) == 1:
        if 'concat' in model_types[0]:
            size_of_fcnn_layer_1 = sum([params['layers_2_mRNA'],params['layers_2_DNA']])
        if 'elementwise' in model_types[0]:
            size_of_fcnn_layer_1 = max(params['layers_2_mRNA'],params['layers_2_mRNA'])
        if 'overall' in model_types[0]:
            size_of_fcnn_layer_1 = 1


    if len(model_types) == 2:
        if "elementwise" in model_types[0] and "overall" in model_types[1]:
            feature_dimension_h = in_feats_second_NN_element_wise
            feature_dimension_FCNN = [1]
        #  feature_dimension = in_feats_third_NN_elementwise_hierachical
        if "concat" in model_types[0] and "overall" in model_types[1]:
            feature_dimension_h = in_feats_second_NN_concat
            feature_dimension_FCNN = [1]
        #   feature_dimension = in_feats_third_NN_concat_hierachical


    if len(model_types) == 2 and model_types[0] != 'none':
        # In this case we already have done integration method and have single omic data structure left
        if 'concat' in model_types[0]:
            feats_in = sum(out_sizes)
        if 'elementwise' in model_types[0]:
            # can be the last layer of aribtrary view from the first AE, as they need to have the same size
            # for elementwise integration
            feats_in = params['layers_2_mRNA']

    if len(model_types) == 2:
        # Since we definetly had overall averaging/maximizing in second AE
        size_of_fcnn_layer_1 = 1


    #JUMPER1 (layer size)


    if len(model_types) == 2:
        all_models.append(AE(views = ['AE'],
                             in_features= feature_dimension_h,
                             n_hidden_layers_dims= [[feats_in, params['layers_2_hierarichcal_integrated']]],
                             activ_funcs=[[params['layers_1_activfunc_hierarichcal_integrated'],params['layers_2_activfunc_hierarichcal_integrated']]],
                             dropout_bool= params['dropout_bool_hierachical_integrated'],
                             dropout_prob= params['dropout_prob_hierachical_integrated'],
                             dropout_layers= [[params['layers_1_dropout_hierarichcal_integrated'],params['layers_2_hierarichcal_integrated']]],
                             batch_norm_bool= params['batchnorm_bool'],
                             batch_norm= [[params['layers_1_batchnorm_hierarichcal_integrated'],params['layers_2_batchnorm_hierarichcal_integrated']]],
                             type_ae=model_types[1],
                             cross_mutation=None,
                             print_bool=False,
                             prelu_init= params['prelu_rate'])).to(device)


    all_models.append(FCNN.NN_changeable(views = ['AE'],
                                         in_features = feature_dimension_FCNN,
                                         n_hidden_layers_dims= [[size_of_fcnn_layer_1]],
                                         activ_funcs = [[params['layers_1_FCNN_activfunc']],['none']],
                                         dropout_prob=params['FCNN_dropout_prob'],
                                         dropout_layers=[[params['layers_1_FCNN_dropout']]],
                                         batch_norm = [[params['layers_1_FCNN_batchnorm']]],
                                         dropout_bool=params['FCNN_dropout_bool'],
                                         batch_norm_bool=params['FCNN_batchnorm_bool'],
                                         print_bool=False,
                                         prelu_init = params['prelu_rate'])).to(device)



    if len(model_types) == 2:
        full_net = AE_Hierarichal(all_models, types=model_types)

    if len(model_types) == 1:
        full_net = AE_NN(all_models, type=model_types[0])


    # set optimizer
    if params['l2_regularization_bool'] == True:
        optimizer = Adam(full_net.parameters(), lr=params['learning_rate'], weight_decay=params['l2_regularization_rate'])
    else:
        optimizer = Adam(full_net.parameters(), lr=params['learning_rate'])

    callbacks = [tt.callbacks.EarlyStopping(patience=10)]
    #   cross_pos = model_types.index("cross") + 1
    #   model = models.CoxPH(full_net,optimizer, loss=LossHierarichcalAESingleCross(alpha=[0.4,0.4,0.2], decoding_bool=True, cross_position=cross_pos)) # Change Loss here

    if second_decoder_bool == True:
        loss_MSE = params['loss_MSE']
        loss_MSE_2 = params['loss_MSE_2']
        # losses need to sum up to 1
        summed_losses = loss_MSE + loss_MSE_2 + params['loss_surv']
        loss_3_values_hierarchical = [loss_MSE/summed_losses, loss_MSE_2/summed_losses, params['loss_surv']/summed_losses]


    if len(model_types) == 1:
        if "cross" in model_types[0]:
            loss_func = LossAECrossHazard(params['loss_surv'])
        else:
            loss_func = LossAEConcatHazard(params['loss_surv'])


    # if second decoder, we need to use loss_3 ; else normal 2 valued loss
    if len(model_types) == 2:
        if second_decoder_bool == False:
            loss_val  = [params['loss_surv'], 1-params['loss_surv']]
            if "cross" in model_types[0]:
                loss_func = LossHierarichcalAESingleCross(loss_val, cross_position=1,decoding_bool=second_decoder_bool)
            else:
                loss_func = LossHierarichcalAE(loss_val,decoding_bool=second_decoder_bool)
        else:
            loss_val = loss_3_values_hierarchical
            if "cross" in model_types[0]:
                loss_func = LossHierarichcalAESingleCross(loss_val, cross_position=1,decoding_bool=second_decoder_bool)
            else:
                loss_func = LossHierarichcalAE(loss_val,decoding_bool=second_decoder_bool)


        # decoding bool False : loss = [loss_surv, 1-loss_surv] ; else loss_3_values_hierarchical
    #   loss_hierachical_no_cross = LossHierarichcalAE(loss_3_values_hierarchical,decoding_bool=second_decoder_bool)
    # loss : alpha * surv_loss + (1-alpha) * ae_loss
    model = models.CoxPH(full_net,
                         optimizer,
                         loss=loss_func)
    model.set_device(torch.device(device))
    print_loss = False

    log = model.fit(train_data[c_fold],
                    train_surv,
                    params['batch_size'],
                    100,
                    verbose=print_loss,
                    val_data= val_data_full,
                    val_batch_size= params['batch_size'],
                    callbacks=callbacks)

    # Plot it
    #   _ = log.plot()

    # Change for EvalSurv-Function
    try:
        test_duration = test_duration.cpu().detach().numpy()
        test_event = test_event.cpu().detach().numpy()
    except AttributeError:
        pass


    for c,fold in enumerate(train_data):
        try:
            train_duration[c_fold] = train_duration[c_fold].cpu().detach().numpy()
            train_event[c_fold] = train_event[c_fold].cpu().detach().numpy()
            val_duration[c_fold] = val_duration[c_fold].cpu().detach().numpy()
            val_event[c_fold] = val_event[c_fold].cpu().detach().numpy()
        except AttributeError: # in this case already numpy arrays
            pass

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

    # Can also be optimized with brier score or binomial log-likelihood
    #brier score
    #  time_grid = np.linspace(test_duration.min(), test_duration.max(), 100)
    ##  _ = ev.brier_score(time_grid).plot
    #  brier_score = ev.integrated_brier_score(time_grid)

    #binomial log-likelihood
    #   binomial_score = ev.integrated_nbll(time_grid)

    print(concordance_index)
#  return concordance_index





def optuna_optimization(n_fold, t_preprocess,feature_selection_type,model_type,decoder_bool,cancer,mode):
    """
    Optuna Optimization for Hyperparameters.
    :param n_fold: number of fold to be optimized ; dtype : Int
    :param t_preprocess: type of preprocess ; dtype : String
    :param feature_selection_type : type of feature selection ; dtype : String
    :param model_type : types of AE model(s) ; dtype : List of String(s)
    :param decoder_bool : decide whether second decoder will be applied ; dtype : Boolean
    :param cancer : name of the cancer used to load the data ; dtype : String
    """


    # Set amount of different trials
    EPOCHS = 100

    func = lambda trial: objective(trial, n_fold, t_preprocess,feature_selection_type, model_type, decoder_bool,cancer,mode)


    study = optuna.create_study(directions=['maximize'],sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.MedianPruner())
    study.optimize(func, n_trials = EPOCHS)
    trial = study.best_trials

    # Save the best trial for each fold
    direc_set = 'SUMO'

    dir = os.path.expanduser(r'~/{}/Project/Trial/{}/{}/AE_{}_{}_{}_{}_{}.txt'.format(direc_set,model_type[0],t_preprocess,t_preprocess,cancer,feature_selection_type,model_type[0],n_fold))
    with open(dir, 'w') as fp:
        for item in trial:
            # write each item on a new line
            fp.write("%s\n" % item)



    # Show change of c-Index across folds
    fig = optuna.visualization.plot_optimization_history(study)
    dir = os.path.expanduser(r'~/{}/Project/Trial/{}/{}/AE_{}_{}_{}_{}_{}_C-INDICES.png'.format(direc_set,model_type[0],t_preprocess,t_preprocess,cancer,feature_selection_type,model_type[0],n_fold))
    # fig.show()
    fig.write_image(dir)
    # fig.show(renderer='browser')
    # Show hyperparameter importance
    fig = optuna.visualization.plot_param_importances(study)
    dir = os.path.expanduser(r'~/{}/Project/Trial/{}/{}/AE_{}_{}_{}_{}_{}_HPARAMOPTIMIZING.png'.format(direc_set,model_type[0],t_preprocess,t_preprocess,cancer,feature_selection_type,model_type[0],n_fold))
    fig.write_image(dir)



    # Save all trials in dataframe
#  df = study.trials_dataframe()
#    df = df.sort_values('value')
#  df.to_csv("~/SUMO/Project/Trial/FCNN_KIRC3_Standardize_PCA.csv")

# print("Best Concordance Sum", trial.value)
# print("Best Hyperparameters : {}".format(trial.params))


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
          batchnorm_layers_third,
          loss_rate):
    """

    :param train_data: Training Data for each fold for each view  ; dtype : List of Lists [for each view] of Tensors(n_samples,n_features)
    :param val_data: Validation Data for each fold for each view  ; dtype : List of Lists [for each view] of Tensors(n_samples,n_features)
    :param test_data: Test Data for each fold for each view  ; dtype : List of Lists [for each view] of Tensors(n_samples,n_features)
    :param train_duration: Training Duration for each fold  ; dtype : List of Tensors(n_samples,)
    :param val_duration: Validation Duration for each fold  ; dtype : List of Tensors(n_samples,)
    :param test_duration: Test Duration for each fold  ; dtype : List of Tensors(n_samples,)
    :param train_event: Training Event for each fold  ; dtype : List of Tensors(n_samples,)
    :param val_event: Validation Event for each fold  ; dtype : List of Tensors(n_samples,)
    :param test_event: Test Event for each fold  ; dtype : List of Tensors(n_samples,)
    :param n_epochs: Number of Epochs ; dtype : Int
    :param batch_size: Batch Size ; dtype : Int
    :param l2_regularization: Decide whether to apply L2 regularization ; dtype : Boolean
    :param l2_regularization_rate: L2 regularization rate ; dtype : Float
    :param learning_rate: Learning rate ; dtype : Float
    :param prelu_rate: Initial Value for PreLU activation ; dtype : Float [between 0 and 1]
    :param layers: Dimension of Layers for each view ; dtype : List of lists of Ints
    :param activation_layers: Activation Functions (for each view)
                              ; dtype : List of Lists of Strings ['relu', 'sigmoid', 'prelu']
    :param dropout: Decide whether Dropout is to be applied or not ; dtype : Boolean
    :param dropout_rate: Probability of Neuron Dropouts ; dtype : Int
    :param dropout_layers:  Layers in which to apply Dropout ; dtype : List of Lists of Strings ['yes','no']
    :param batchnorm: Decide whether Batch Normalization is to be applied or not ; dtype : Boolean
    :param batchnorm_layers: Layers in which to apply Batch Normalization ; dtype : List of Lists of Strings ['yes','no']
    :param view_names: Names of used views ; dtype : List of Strings
    :param cross_mutation: Choose additional decoder for view ; dtype : List of Int [Indices of decoders for view,
                            e.g [1,3,0,2] will decode features of view 1 with decoder of view 2 (and 1),
                            features of view 2 with decoder of view 4 (and 2) ...]
    :param model_types: Bottlneck/Decoder manipulation ; dtype : String ['concat','elementwisemax','elementwisemin',
    elementwiseavg','overallmax','overallmin','overallavg', 'cross' or 'cross_k' where k is on of the mentionted types]
    :param dropout_second: Decide whether Dropout is to be applied or not for the second AE ; dtype : Boolean
    :param dropout_rate_second: Probability of Neuron Dropouts for the second AE ; dtype : Int
    :param batchnorm_second: Decide whether Batch Normalization is to be applied or not for the second AE; dtype : Boolean
    :param layers_second: Dimension of Layers for each view for the second AE ; dtype : List of lists of Ints
    :param activation_layers_second: Activation Functions (for each view) for the second AE
                                     ; dtype : List of Lists of Strings ['relu', 'sigmoid', 'prelu']
    :param dropout_layers_second: Layers in which to apply Dropout for the second AE
                                  ; dtype : List of Lists of Strings ['yes','no']
    :param batchnorm_layers_second: Layers in which to apply Batch Normalization for the second AE
                                    ; dtype : List of Lists of Strings ['yes','no']
    :param dropout_third: Decide whether Dropout is to be applied or not for the FCNN ; dtype : Boolean
    :param dropout_rate_third: Probability of Neuron Dropouts fort the FCNN ; dtype : Int
    :param batchnorm_third: Decide whether Batch Normalization is to be applied or not for the FCNN; dtype : Boolean
    :param layers_third: Dimension of Layers for each view for the FCNN ; dtype : List of lists of Ints
    :param activation_layers_third: Activation Functions (for each view) for the FCNN
                                     ; dtype : List of Lists of Strings ['relu', 'sigmoid', 'prelu']
    :param dropout_layers_third: Layers in which to apply Dropout for the FCNN
                                  ; dtype : List of Lists of Strings ['yes','no']
    :param batchnorm_layers_third: Layers in which to apply Batch Normalization for the FCNN
                                    ; dtype : List of Lists of Strings ['yes','no']
    :param loss_rate : Loss ratio between MSE loss(es) and negative partial log likelihood loss ; dtype : List of Int [2 decoders] or Int [1 decoder]
    """




    #:param ae_hierarichcal_bool : Choose whether this AE is the second AE in the pipeline ; dtype : Boolean
    #:param ae_hierarichcal_decoding_bool : Choose whether the second AE has a decoder ; dtype : Boolean


    for c,fold in enumerate(train_data):
        # Need tuple structure for PyCox
        train_data[c] = tuple(train_data[c])
        val_data[c] = tuple(val_data[c])
        test_data[c] = tuple(test_data[c])



    ############################# FOLD X ###################################
    for c_fold,fold in enumerate(train_data):
        for c2,view in enumerate(fold):

            # For GPU acceleration, we need to have everything as tensors for the training loop, but pycox EvalSurv
            # Needs duration & event to be numpy arrays, thus at the start we set duration/event to tensors
            # and before EvalSurv to numpy
            try:
                test_duration = torch.from_numpy(test_duration).to(torch.float32)
                test_event = torch.from_numpy(test_event).to(torch.float32)
            except TypeError:
                pass


            for c,fold in enumerate(train_data):
                try:
                    train_duration[c] = torch.from_numpy(train_duration[c]).to(torch.float32)
                    train_event[c] = torch.from_numpy(train_event[c]).to(torch.float32)
                    val_duration[c] = torch.from_numpy(val_duration[c]).to(torch.float32)
                    val_event[c] = torch.from_numpy(val_event[c]).to(torch.float32)
                except TypeError:
                    pass

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



        #model_types = ['none', 'cross_elementwiseavg']
        print("MODEL TYPES : ", model_types)





        out_sizes = []
        for c_layer in range(len(layers)):
            out_sizes.append(layers[c_layer][-1])


        in_feats_second_NN_concat = [sum(out_sizes)]
        in_feats_second_NN_elementwise = [max([i[-1] for i in layers])]

        out_sizes_second = []
        for c_layer in range(len(layers_second)):
            out_sizes_second.append(layers_second[c_layer][-1])


        in_feats_third_NN_concat = [sum(out_sizes_second)]
        in_feats_third_NN_elementwise = [max([i[-1] for i in layers_second])]


        # if first model type is none, we need to take bottleneck dimensions for second AE
        in_feats_second_AE_hierachical = [x[-1] for x in layers]



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
                             cross_mutation=None,
                             print_bool=False,
                             prelu_init= prelu_rate))


        layers_u_2 = copy.deepcopy(layers_second)
        activation_layers_u_2 = copy.deepcopy(activation_layers_second)
        dropout_layers_u_2 = copy.deepcopy(dropout_layers_second)
        batchnorm_layers_u_2 = copy.deepcopy(batchnorm_layers_second)
        all_models.append(AE(views = view_names,
                             in_features=in_feats_second_AE_hierachical,
                             n_hidden_layers_dims= layers_u_2,
                             activ_funcs = activation_layers_u_2,
                             dropout_prob= dropout_rate_second,
                             dropout_bool= dropout_second,
                             dropout_layers = dropout_layers_u_2,
                             batch_norm = batchnorm_layers_u_2,
                             batch_norm_bool=batchnorm_second,
                             type_ae =model_types[1],
                             cross_mutation=cross_mutation,
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
                                             in_features = in_feats_third_NN_elementwise,
                                             n_hidden_layers_dims= layers_u_3,
                                             activ_funcs = activation_layers_u_3,
                                             dropout_prob=dropout_rate_third,
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

        #   full_net = AE_NN(all_models, type=model_types[0])


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
                             loss=LossHierarichcalAESingleCross(alpha=[loss_rate, 1-loss_rate], decoding_bool= False, cross_position=2))
        print_loss = False
        print("Split {} : ".format(c_fold + 1))
        log = model.fit(train_data[c_fold],
                        train_surv,
                        batch_size,
                        n_epochs,
                        verbose=print_loss,
                        val_data= val_data_full,
                        val_batch_size= 12,
                        callbacks=callbacks)

        # Plot it
        _ = log.plot()

        # Change for EvalSurv-Function
        try:
            test_duration = test_duration.cpu().detach().numpy()
            test_event = test_event.cpu().detach().numpy()
        except AttributeError:
            pass


        for c,fold in enumerate(train_data):
            try:
                train_duration[c_fold] = train_duration[c_fold].cpu().detach().numpy()
                train_event[c_fold] = train_event[c_fold].cpu().detach().numpy()
                val_duration[c_fold] = val_duration[c_fold].cpu().detach().numpy()
                val_event[c_fold] = val_event[c_fold].cpu().detach().numpy()
            except AttributeError: # in this case already numpy arrays
                pass

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




def load_data(data_dir):
    """
    Function to load data. Needed for Optuna Optimization.
    :param data_dir: Directory in which data is stored.
    :return: data and feature offsets (for feature values, duration and event)
    """

    trainset_0 = pd.read_csv(
        os.path.join(data_dir + "TrainData_0.csv"), index_col=0)
    trainset_1 = pd.read_csv(
        os.path.join(data_dir + "TrainData_1.csv"), index_col=0)
    trainset_2 = pd.read_csv(
        os.path.join(data_dir + "TrainData_2.csv"), index_col=0)
    trainset_3 = pd.read_csv(
        os.path.join(data_dir + "TrainData_3.csv"), index_col=0)
    trainset_4 = pd.read_csv(
        os.path.join(data_dir + "TrainData_4.csv"), index_col=0)

    trainset_feat_0 = pd.read_csv(
        os.path.join(data_dir +"TrainDataFeatOffs_0.csv"), index_col=0)

    trainset_feat_1 = pd.read_csv(
        os.path.join(data_dir +"TrainDataFeatOffs_1.csv"), index_col=0)

    trainset_feat_2 = pd.read_csv(
        os.path.join(data_dir +"TrainDataFeatOffs_2.csv"), index_col=0)

    trainset_feat_3 = pd.read_csv(
        os.path.join(data_dir +"TrainDataFeatOffs_3.csv"), index_col=0)

    trainset_feat_4 = pd.read_csv(
        os.path.join(data_dir +"TrainDataFeatOffs_4.csv"), index_col=0)


    valset_0 = pd.read_csv(
        os.path.join(data_dir + "ValData_0.csv"), index_col=0)
    valset_1 = pd.read_csv(
        os.path.join(data_dir + "ValData_1.csv"), index_col=0)
    valset_2 = pd.read_csv(
        os.path.join(data_dir + "ValData_2.csv"), index_col=0)
    valset_3 = pd.read_csv(
        os.path.join(data_dir + "ValData_3.csv"), index_col=0)
    valset_4 = pd.read_csv(
        os.path.join(data_dir + "ValData_4.csv"), index_col=0)

    testset_0 = pd.read_csv(
        os.path.join(data_dir +  "TestData_0.csv"), index_col=0)
    testset_1 = pd.read_csv(
        os.path.join(data_dir +  "TestData_1.csv"), index_col=0)
    testset_2 = pd.read_csv(
        os.path.join(data_dir +  "TestData_2.csv"), index_col=0)
    testset_3 = pd.read_csv(
        os.path.join(data_dir +  "TestData_3.csv"), index_col=0)
    testset_4 = pd.read_csv(
        os.path.join(data_dir +  "TestData_4.csv"), index_col=0)




    return trainset_0,trainset_1,trainset_2,trainset_3,trainset_4,valset_0,valset_1,valset_2,valset_3,valset_4, \
           testset_0,testset_1,testset_2,testset_3,testset_4, \
           trainset_feat_0,trainset_feat_1,trainset_feat_2,trainset_feat_3,trainset_feat_4