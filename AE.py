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






def train(module,
          device,
          feature_select_method = 'eigengenes',
          components = None,
          thresholds = None,
          feature_names = None,
          batch_size =25,
          n_epochs = 512,
          l2_regularization = False,
          val_batch_size=16,
          number_folds=5):
    """

    :param module: basically the dataset to be used
    :param batch_size: batch size for training
    :param n_epochs: number of epochs
    :param lr_scheduler_type: type of lr_scheduler : 'lambda','mulitplicative','step','multistep','exponential',...
    :param l2_regularization : bool whether should be applied or not
    """




    # Setup all the data
    n_train_samples, n_test_samples,n_val_samples, view_names = module.setup()



    #Select method for feature selection
    train_data, val_data, test_data, train_duration, train_event, val_duration, val_event, test_duration, test_event = module.feature_selection(method=feature_select_method, components= components,thresholds= thresholds,feature_names= feature_names)




    for c,fold in enumerate(train_data):
        train_duration[c] = train_duration[c].numpy()
        train_event[c] = train_event[c].numpy()
        val_duration[c] = val_duration[c].numpy()
        val_event[c] = val_event[c].numpy()
        for c2,view in enumerate(fold):
            train_data[c][c2] = (train_data[c][c2]).numpy()
            val_data[c][c2] = (val_data[c][c2]).numpy()
            test_data[c][c2] = (test_data[c][c2]).numpy()


        # Need tuple structure for PyCox
        train_data[c] = tuple(train_data[c])
        val_data[c] = tuple(val_data[c])
        test_data[c] = tuple(test_data[c])



    for c_fold,fold in enumerate(train_data):
        for c2,view in enumerate(fold):
            print("Split {} : ".format(c_fold))
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

        # TODO : weird problem with variables for models, bc models change variables outside model itself
        # TODO : could be that we create way more layers than we want --> debug!

        # AE's
        all_models.append(AE(view_names, dimensions, [[128,64] for i in range(len(view_names))],
                             [['relu'] for i in range(len(view_names))], 0.2,
                             [['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes']],
                             [['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes']],
                             dropout_bool=False, batch_norm_bool=False, type_ae=model_types[0], cross_mutation=[1,0], print_bool=False))


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

        all_models.append(NN.NN_changeable(views = ['AE'],in_features = [64],
                                           n_hidden_layers_dims= [[128,64]],
                                           activ_funcs = [['relu'],['none']],dropout_prob=0.2,dropout_layers=[['yes','yes']],
                                           batch_norm = [['yes','yes']],
                                           dropout_bool=False,batch_norm_bool=False,print_bool=False))




        #############
        # Note that for the Cross Method, the cross_mutation must be set in a way so that the input view x and
        # "crossed" view y have the same feature size : Otherwise, the MSELoss() cant be calculated
        # e.g. 4 views having feature sizes [15,15,5,5] --> cross mutations between first two and last two work,
        # but not e.g. between first and third/fourth





        #    full_net = AE_Hierarichal(all_models, types=model_types)

        full_net = AE_NN(all_models, type=model_types[0])


        # set optimizer
        if l2_regularization == True:
            optimizer = Adam(full_net.parameters(), lr=0.001, weight_decay=0.0001)
        else:
            optimizer = Adam(full_net.parameters(), lr=0.001)

        callbacks = [tt.callbacks.EarlyStopping()]
        #   cross_pos = model_types.index("cross") + 1
        #   model = models.CoxPH(full_net,optimizer, loss=LossHierarichcalAESingleCross(alpha=[0.4,0.4,0.2], decoding_bool=True, cross_position=cross_pos)) # Change Loss here


        # loss : alpha * surv_loss + (1-alpha) * ae_loss
        model = models.CoxPH(full_net,
                             optimizer,
                             loss=LossAEConcatHazard(0.7))


        log = model.fit(train_data[c_fold],
                        train_surv,
                        batch_size,
                        n_epochs,
                        verbose=True,
                        val_data= val_data_full,
                        val_batch_size= val_batch_size,
                        callbacks=callbacks)

        # Plot it
        _ = log.plot()

        # Since Cox semi parametric, we calculate a baseline hazard to introduce a time variable
        _ = model.compute_baseline_hazards()

        # Predict based on test data
        surv = model.predict_surv_df(test_data[c_fold])

        # Plot it
        surv.iloc[:, :5].plot()
        plt.ylabel('S(t | x)')
        _ = plt.xlabel('Time')


        # Evaluate with concordance, brier score and binomial log-likelihood
        ev = EvalSurv(surv, test_duration[c_fold], test_event[c_fold], censor_surv='km') # censor_surv : Kaplan-Meier

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

















    # Load Dataloaders
    trainloader = module.train_dataloader(batch_size=n_train_samples[0])

    #Train
    for train_data, train_duration, train_event in trainloader:
        for c,fold in enumerate(train_data):
            for c2,view in enumerate(fold):
                train_data[c][c2] = train_data[c][c2].to(device=device)

    trainloader = module.train_dataloader(batch_size=n_train_samples) # all training examples
    testloader =module.test_dataloader(batch_size=n_test_samples)

    # Load data and set device to cuda if possible

    #Train
    for train_data, train_duration, train_event in trainloader:
        for view in range(len(train_data)):
            train_data[view] = train_data[view].to(device=device)

        train_duration.to(device=device)
        train_event.to(device=device)

    for c,_ in enumerate(train_data):
        print("Train data shape after feature selection {}".format(train_data[c].shape))


    #Test
    for test_data, test_duration, test_event in testloader:
        for view in range(len(test_data)):
            test_data[view] = test_data[view].to(device=device)


        test_duration.to(device=device)
        test_event.to(device=device)

    for c,_ in enumerate(test_data):
        print("Test data shape after feature selection {}".format(test_data[c].shape))



    # For PyCox, we need to change the structure so that all the data of different views is wrapped in a tuple
    train_data = tuple(train_data)
    test_data = tuple(test_data)

    # Input dimensions (features for each view) for NN based on different data (train/validation/test)
    # Need to be the same for NN to work
  #  dimensions_train = [x.size(1) for x in train_data]
  #  dimensions_test = [x.size(1) for x in test_data]

    dimensions_train = [x.shape[1] for x in train_data]
    dimensions_test = [x.shape[1] for x in test_data]

    assert (dimensions_train == dimensions_test), 'Feature mismatch between train/test'

    dimensions = dimensions_train

    print("Views :", view_names)

    # Cross Validation:
    # We first need to concatenate all the data (train,test) together.
    # Since each view has a different feature size, we do that in the cross validation loop ;
    # duration & event can be done here already.

    full_data = []
    for view_idx in range(len(view_names)):
        data = torch.cat(tuple([train_data[view_idx],test_data[view_idx]]), dim=0)
        full_data.append(data)


    full_duration = torch.cat(tuple([train_duration,test_duration]),dim=0)
    full_event = torch.cat(tuple([train_event,test_event]),dim=0)

    # k is the number of folds
    k = number_folds

    splits = KFold(n_splits=k,shuffle=True,random_state=42)

    n_all_samples = n_train_samples + n_test_samples


    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(n_all_samples))):
        torch.manual_seed(0)

        print('Fold {}'.format(fold + 1))


        event_boolean = False

        while event_boolean == False: # TODO : random shuffle, wenn in e.g val_idx schon alle event 0 sind , wird nichts verändern und loop geht unendlich..
            np.random.seed(seed=0)
            train_sampler = SubsetRandomSampler(train_idx) # np.array e.g. [0 2 4 8 18 22 ..]
            test_sampler = SubsetRandomSampler(val_idx)






            # we take a subset of our train samples for the validation set
            # 20% of train is validation set
            # We first need to turn the train_sampler into a np.array (else the method won't work)
            train_sampler_array = np.empty(len(train_sampler))

            for c,idx in enumerate(train_sampler):
                np.put(train_sampler_array,c,idx)

            val_sampler = np.random.choice(train_sampler_array, int(0.2 * len(train_sampler)))

            #change dtype
            val_sampler = val_sampler.astype('int32')
            train_sampler_array = train_sampler_array.astype('int32')

            # get indices as list
            val_sampler_list = val_sampler.tolist()
            # remove these from train sampler
            train_sampler_array = np.setdiff1d(train_sampler_array, val_sampler_list)

            # Get duration & events
            train_duration = []
            train_event = []
            for idx in train_sampler_array:
                train_duration.append(full_duration[idx])
                train_event.append(full_event[idx])

            train_duration = torch.stack(tuple(train_duration))
            train_event = torch.stack(tuple(train_event))

            train_duration = train_duration.numpy()
            train_event = train_event.numpy()


            val_duration = []
            val_event = []
            for idx in val_sampler:
                val_duration.append(full_duration[idx])
                val_event.append(full_event[idx])

            val_duration = torch.stack(tuple(val_duration))
            val_event = torch.stack(tuple(val_event))

            val_duration = val_duration.numpy()
            val_event = val_event.numpy()

            # Check that each set contains atleast one event that is not censored




            test_duration = []
            test_event = []
            for idx in test_sampler:

                test_duration.append(full_duration[idx])
                test_event.append(full_event[idx])

            test_duration = torch.stack(tuple(test_duration))
            test_event = torch.stack(tuple(test_event))

            test_duration = test_duration.numpy()
            test_event = test_event.numpy()


            print("val non censored samples : ", np.count_nonzero(val_event))
            print("train non censored samples : ", np.count_nonzero(train_event))
            print("test non censored samples : ", np.count_nonzero(test_event))


            if np.count_nonzero(val_event) != 0 \
                    and np.count_nonzero(train_event) != 0 \
                    and np.count_nonzero(test_event) != 0:

                event_boolean = True


        # Numpy transforms for PyCox


        train_surv = (train_duration, train_event)




        # Get necessary data for each view
        train_data_full = []
        val_data_full = []
        test_data_full = []
        for view_idx in range(len(view_names)):

            # get training data
            train_data = []

            for idx in train_sampler_array:
                train_data.append(full_data[view_idx][idx])


            train_data = torch.stack(tuple(train_data))

            train_data_full.append(train_data.numpy())


            # get validation data
            val_data = []

            for idx in val_sampler:
                val_data.append(full_data[view_idx][idx])


            val_data = torch.stack(tuple(val_data))

            val_data_full.append(val_data.numpy())



            # get testing data
            test_data = []

            for idx in test_sampler:
                test_data.append(full_data[view_idx][idx])


            test_data = torch.stack(tuple(test_data))

            test_data_full.append(test_data.numpy())





        #Change into tuple structure
        train_data_full = tuple(train_data_full)
        val_data_full = tuple(val_data_full)
        test_data_full = tuple(test_data_full)


        # for PyCox
        val_data = (val_data_full, (val_duration, val_event))



        torch.manual_seed(0)

        # Create List of models to be used
        all_models = nn.ModuleList()


        model_types = ['cross','elementwisemax']

        print("MODEL TYPES : ", model_types)

        # TODO : weird problem with variables for models, bc models change variables outside model itself
        # TODO : could be that we create way more layers than we want --> debug!

        # AE's
        all_models.append(AE(view_names, dimensions, [[128,64] for i in range(len(view_names))],
                             [['relu'] for i in range(len(view_names))], 0.2,
                             [['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes']],
                             [['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes'],['yes','yes','yes']],
                             dropout_bool=False, batch_norm_bool=False, type_ae=model_types[0], cross_mutation=[1,0], print_bool=False))


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

        all_models.append(NN.NN_changeable(views = ['AE'],in_features = [64],
                                           n_hidden_layers_dims= [[128,64]],
                                           activ_funcs = [['relu'],['none']],dropout_prob=0.2,dropout_layers=[['yes','yes']],
                                           batch_norm = [['yes','yes']],
                                           dropout_bool=False,batch_norm_bool=False,print_bool=False))




            #############
            # Note that for the Cross Method, the cross_mutation must be set in a way so that the input view x and
            # "crossed" view y have the same feature size : Otherwise, the MSELoss() cant be calculated
            # e.g. 4 views having feature sizes [15,15,5,5] --> cross mutations between first two and last two work,
            # but not e.g. between first and third/fourth





    #    full_net = AE_Hierarichal(all_models, types=model_types)

        full_net = AE_NN(all_models, type=model_types[0])


        # set optimizer
        if l2_regularization == True:
            optimizer = Adam(full_net.parameters(), lr=0.001, weight_decay=0.0001)
        else:
            optimizer = Adam(full_net.parameters(), lr=0.001)

        callbacks = [tt.callbacks.EarlyStopping()]
     #   cross_pos = model_types.index("cross") + 1
     #   model = models.CoxPH(full_net,optimizer, loss=LossHierarichcalAESingleCross(alpha=[0.4,0.4,0.2], decoding_bool=True, cross_position=cross_pos)) # Change Loss here


        # loss : alpha * surv_loss + (1-alpha) * ae_loss
        model = models.CoxPH(full_net,
                             optimizer,
                             loss=LossAECrossHazard(0.7))


        log = model.fit(train_data_full,
                        train_surv,
                        batch_size,
                        n_epochs,
                        verbose=True,
                        val_data= val_data,
                        val_batch_size= val_batch_size,
                        callbacks=callbacks)

        # Plot it
        _ = log.plot()

        # Since Cox semi parametric, we calculate a baseline hazard to introduce a time variable
        _ = model.compute_baseline_hazards()

        # Predict based on test data
        surv = model.predict_surv_df(test_data_full)

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
    # Get feature offsets for train/validation/test
    # Need to be the same for NN to work
    feature_offsets_train = [0] + np.cumsum(dimensions_train).tolist()
    feature_offsets_test = [0] + np.cumsum(dimensions_test).tolist()

    feature_offsets = feature_offsets_train

    # Number of all features (summed up) for train/validation/test
    # These need to be the same, otherwise NN won't work
    feature_sum_train = feature_offsets_train[-1]
    feature_sum_test = feature_offsets_test[-1]

    feature_sum = feature_sum_train

    # Initialize empty tensors to store the data for train/validation/test
    train_data_pycox = torch.empty(n_train_samples, feature_sum_train).to(torch.float32)
    test_data_pycox = torch.empty(n_test_samples, feature_sum_test).to(torch.float32)

    # Train
    for idx_view,view in enumerate(train_data):
        for idx_sample, sample in enumerate(view):
            train_data_pycox[idx_sample][feature_offsets_train[idx_view]:
                                         feature_offsets_train[idx_view+1]] = sample


    # Test
    for idx_view,view in enumerate(test_data):
        for idx_sample, sample in enumerate(view):
            test_data_pycox[idx_sample][feature_offsets_test[idx_view]:
                                        feature_offsets_test[idx_view+1]] = sample

"""