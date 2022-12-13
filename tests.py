
# sk learn feature selection methods
# https://scikit-learn.org/stable/modules/feature_selection.html


from typing import Dict, List, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import DataInputNew
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from torch import nn
from torch.optim import Adam
import os
import time
import math
import subprocess



class F_PCA():

    input : List[torch.Tensor]
    components : int

    def __init__(self, train, test = None, components = None, keep_variance = 0.5):
        """

        :param train:
        :param test:
        :param components:
        :param keep_variance: choose minimal number of PC components such that x % variance is retained
        """
        self.components = components
        self.train = train #training input
        self.test = test
        self.keep_variance = keep_variance


    def apply_pca(self):
        self.pca = PCA(self.keep_variance)

        #As we already standardized & transformed the data with StandardScaler(), we can apply PCA directly
        self.pca.fit(self.train)
        pca_train_data = self.pca.transform(self.train)

        return pca_train_data





class F_VARIANCE():
    """Removing features with low variance
       https://scikit-learn.org/stable/modules/feature_selection.html
       https://towardsdatascience.com/how-to-use-variance-thresholding-for-robust-feature-selection-a4503f2b5c3f
       Unsupervised variance-based feature selection
    """

    def __init__(self, train, features = None, test = None, threshold = 0.5):                                           # TODO : Threshold for each diff data type, grid search ?
        self.train = train
        self.features = features
        self.test = test
        self.threshold = threshold



    def apply_variance(self):
        # Get rid of features with 0 variance
        vt = VarianceThreshold(self.threshold)
        data_selected = vt.fit_transform(self.train)


        # get mask showing the features that were selected/not selected
        # True : selected, False : not selected (variance 0)
        mask = vt.get_support()

        # Find features that were selected
        index_list = []

        index = 0
        for x in mask:
            if x == True:
                index_list.append(index)
            index += 1





        return data_selected, index_list


class F_mRMR():
    """For classification tasks"""
    pass


class F_eigengene_matrices():
    """Implemented with R :
       https://github.com/huangzhii/lmQCM
    """

    def __init__(self,train,mask,view_name,duration,event,stage = 'train'):
        self.train = train
        self.stage = stage # test or train stage
        self.mask = mask
        self.view_name = view_name
        self.duration = duration
        self.event = event


    def preprocess(self):
        """For the eigengene matrices computation, we need to remove rows (samples) which had all NaN values
        (missing sample) aswell as each column (feature), which had the same value for each patient (sample) (this feature
        is irrelevant either way, as it doesn't have any impact on any patient)"""


        train_df = pd.DataFrame(self.train.numpy())
        duration_df = pd.DataFrame(self.duration.numpy())
        event_df = pd.DataFrame(self.event.numpy())



        to_drop_columns = []

        for x in range(len(train_df.columns)):
            # if column (feature) has same values for each row (patient)
            if (train_df.iloc[:, x] == train_df.iloc[:, x][0]).all():

                to_drop_columns.append(x)

        train_df = train_df.drop(train_df.columns[to_drop_columns], axis = 1)

        to_drop_index = []

        for x in range(len(train_df.index)):

            # if row (sample) had all NaN values
            if torch.all(self.mask[x] == True):

                to_drop_index.append(x)


        train_df = train_df.drop(to_drop_index, axis = 0)
        duration_df = duration_df.drop(to_drop_index, axis = 0)
        event_df = event_df.drop(to_drop_index, axis = 0)

        train_df.reset_index()
        duration_df.reset_index()
        event_df.reset_index()


        if self.stage == 'train':
            train_df.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/{}_for_r.csv".format(self.view_name))
        else: # self.stage == 'test'
            train_df.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/{}_test_for_r.csv".format(self.view_name))


        # return these so we can add them to according samples
        return torch.tensor(duration_df.iloc[:, 0].values), torch.tensor(event_df.iloc[:, 0].values)


    def eigengene_multiplication(self):
        """We calculate the eigengene matrix for all views
        need to set rights first so Python can read it : chmod a+rwx 'PATHTOFILE'
        """

        path = r"/Users/marlon/DataSpellProjectsForSAMO/SAMO/eigengene_matrices.R"


        subprocess.call("Rscript /Users/marlon/DataSpellProjectsForSAMO/SAMO/eigengene_matrices.R", shell=True)



    def get_eigengene_matrices(self,views):
        """Loading the eigengene matrix for a sample as a panda Dataframe created from R."""
        eigengene_matrices = []
        eigengene_test_matrices =[]
        for view in views:
            #        while not os.path.exists(os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
            #                                              "{}_eigengene_matrix.csv".format(view))):
            #            time.sleep(1)


            eigengene_matrix = pd.read_csv(os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                                                        "{}_eigengene_matrix.csv".format(view)), index_col=0)
            eigengene_test_matrix = pd.read_csv(os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                                                             "{}_test_eigengene_matrix.csv".format(view)), index_col=0)


            eigengene_matrices.append(eigengene_matrix)
            eigengene_test_matrices.append(eigengene_test_matrix)
        return eigengene_matrices, eigengene_test_matrices




class F_AE(nn.Module):
    """Use the hidden layer between Encoder and Decoder.
    Based on : https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
    """
    # TODO : Modell konvergiert ?
    # TODO : test : variance-explained : wv varianz explained die approximation (encoder_output_layer) vs original daten
    # TODO : preprocessing AE centering & standardizing
    # normalize ? was genau gemeint ?
    def __init__(self, train, **kwargs):
        super().__init__()
        self.train = train # Need train for the dimension of features for current view

        self.encoder_hidden_1 = nn.Linear(in_features= train.size(dim=1), out_features= 512)
        self.encoder_hidden_2 = nn.Linear(in_features= 512, out_features= 256)
        self.encoder_hidden_3 = nn.Linear(in_features=256, out_features= 128)

        self.encoder_output_layer = nn.Linear(in_features=128, out_features= 128)

        self.decoder_hidden_1 = nn.Linear(in_features=128, out_features=128)
        self.decoder_hidden_2 = nn.Linear(in_features=128, out_features=256)
        self.decoder_hidden_3 = nn.Linear(in_features=256, out_features=512)
        self.decoder_hidden_output_layer = nn.Linear(in_features=512, out_features=train.size(dim=1))


    def forward(self,train, **kwargs):
        # Encoder hidden layer
        in_1 = self.encoder_hidden_1(train)
        out_1 = torch.relu(in_1)
        in_2 = self.encoder_hidden_2(out_1)
        out_2 = torch.relu(in_2)
        in_3 = self.encoder_hidden_3(out_2)
        out_3 = torch.relu(in_3)

        # select features from here
        in_layer_reduced = self.encoder_output_layer(out_3)
        out_layer_reduced = torch.relu(in_layer_reduced)

        # decode again (for training purposes)
        in_d_1 = self.decoder_hidden_1(out_layer_reduced)
        out_d_1 = torch.relu(in_d_1)
        in_d_2 = self.decoder_hidden_2(out_d_1)
        out_d_2 = torch.relu(in_d_2)
        in_d_3 = self.decoder_hidden_3(out_d_2)
        out_d_3 = torch.relu(in_d_3)
        in_d_final = self.decoder_hidden_output_layer(out_d_3)
        out_d_final = torch.relu(in_d_final)

        return out_d_final, out_layer_reduced



class F_PPI_NETWORK():
    def __init__(self,train):
        """

        :param train: the PPI-network does feature selection and integration at the same time. Thus the input is the
                      whole train set
        """
        self.train = train

    def setup(self):
        """
        We first load the PPI edges & score , the feature spaces of all views to be looked at aswell as the feature
        to protein mapping of all views. Based on these, we create the adjacency and feature (per sample) matrices
        :return: adjacency matrix and blueprint for feature matrices (to be created in data loader)
        """

        ppi_edges_score = pd.read_csv(
            os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                         "ppi_edges_score.csv"), index_col=0
        )

        features_mRNA = pd.read_csv(
            os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                         "mRNA_feat_for_transf.csv"), index_col=0
        )

        features_proteins_mapping_mRNA = pd.read_csv(
            os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                         "mRNA_feat_proteins.csv"), index_col=0
        )

        proteins = pd.read_csv(
            os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                         "proteins.csv"), index_col=0
        )

        # Converting data to right structures


        features_mRNA = features_mRNA.iloc[:,0].values.tolist()


        # mapping feature proteins mRNA
        features_proteins_mapping_mRNA = features_proteins_mapping_mRNA.to_dict(orient='list')


        # removing nan values
        for key in features_proteins_mapping_mRNA:
            for x in features_proteins_mapping_mRNA[key]:

                if type(x) is not str:
                    # since we only have strings in the list (proteins), nan values are the only non-strings

                    features_proteins_mapping_mRNA[key].remove(x)


        # ppi nodes & scores
        nodes1 = (ppi_edges_score.iloc[0].values.tolist())
        nodes2 = (ppi_edges_score.iloc[1].values.tolist())
        score = (ppi_edges_score.iloc[2].values.tolist())

        ppi_edges_score = torch.tensor([nodes1,nodes2,score])

        keys = proteins.columns.values.tolist()
        values = proteins.iloc[0].values.tolist()

        proteins_indexed = dict(zip(keys,values))







        #print("train ", data)
        #print("train 0", data[0])
        #print("sample 0", data[0][0])
        #print("sample0 first value", data[0][0][0])


        # Create a list of dictionaries, where each dictionary contains the mapping between proteins and feature
        # values of all views (TO IMPLEMENT : OTHER FEATURES THAN mRNA)
        # We use 389 as batch size, which are all training examples ; needs to be replaced with
        # something like training_elements_size later on

        train_loader = DataInputNew.multimodule.train_dataloader(batch_size = 389)
        # For each sample, make a dictionary with protein : feature value ; save list of dictionaries
        features_values_proteins_mapping_mRNA = []

        for data,mask, duration, event in train_loader:


            # Go through each mRNA sample
            for sample in data[0]:
                # for each sample dictionary
                dict_features_proteins_mRNA_values = {}
                # Through each feature (for mRNA)
                for feature_idx in range(len(sample)): # len = 6000 features
                    # If the feature has a mapping in feature to protein
                    # features_mRNA[feature_idx] : feature name at index
                    if features_mRNA[feature_idx] in features_proteins_mapping_mRNA:


                        # take the proteins at the current index which correspond to the feature we are looking at
                        protein_list = features_proteins_mapping_mRNA[features_mRNA[feature_idx]]

                        # create dictionary entries for it
                        for protein in protein_list:
                            #if key already exists, just append values to protein
                            if protein in dict_features_proteins_mRNA_values:
                                dict_features_proteins_mRNA_values[protein].append(sample[feature_idx])
                            #else create a key entry
                            else:

                                dict_features_proteins_mRNA_values[protein] = [sample[feature_idx]]


                # for each sample, append to list
                features_values_proteins_mapping_mRNA.append(dict_features_proteins_mRNA_values)


            # Create graph representation

        # feature matrix X |A| x |B| A : number of nodes (proteins), B : number of features per node (value features diff. views)
        # adjacency matrix Z : |A| x |A|

        #adjacency matrix for protein-protein-network
        adjacency_matrix_ppi = torch.zeros(len(proteins_indexed), len(proteins_indexed))


        for protein_idx in range(len(proteins_indexed)):
            # for each edge between two nodes, fill in a 1 in adjacency matrix
            adjacency_matrix_ppi[int((ppi_edges_score[0, protein_idx]).item()), int((ppi_edges_score[1, protein_idx]).item())] = torch.tensor(1)

        # todo : adjazenz mit gewichten ?


        # feature matrix for each sample
        #torch tensor with A x B : rows x columns


        n_samples = len(features_values_proteins_mapping_mRNA)
        n_proteins = len(proteins_indexed)

        # take protein (node) which has the most features (values from diff views) for matrix column size over all samples
        # so that we have the same structure for each sample
        n_features_matrix = 0
        for sample_idx in range(n_samples):
            for key, value in features_values_proteins_mapping_mRNA[sample_idx].items():
                if len(value) > n_features_matrix:
                    n_features_matrix = len(value)


        # Intialize empty tensor in the beginning :
        feature_matrices_mRNA = torch.empty(size=(n_samples,n_proteins,n_features_matrix))

        for sample_idx in range(n_samples):

            # As rows, we just use all the proteins ; matrix for one sample
            feature_matrix_mRNA = torch.zeros(len(proteins_indexed), n_features_matrix)

            # fill with data

            # go through each protein
            for protein in proteins_indexed:
                # check if protein in protein :feature values dict
                if protein in features_values_proteins_mapping_mRNA[sample_idx]:

                    # fill row with according data ; find right row by accesing dict protein data (find the right index)

                    # Find index by accessing dict which saves pairs of protein : index
                    index = proteins_indexed[protein]
                    feature_matrix_mRNA[index, 0:len(features_values_proteins_mapping_mRNA[sample_idx][protein])] = \
                        torch.tensor(features_values_proteins_mapping_mRNA[sample_idx][protein])

            #   TODO : normalization before adding into matrices (see DeepMOCCA)

            #else : we leave it as it is (filled with 0s)

            feature_matrices_mRNA[sample_idx] = (feature_matrix_mRNA)



        return adjacency_matrix_ppi, feature_matrices_mRNA





# TODO : allgemein je nachdem wv features eine view hat, relativ davon nur bestimmt viele für feature selection nehmen?


if __name__ == '__main__':

    # Set batch size to size of training set ; We are currently preparing data,
    # thus we can use all samples as one batch (train with single batches later)
    n_train_samples = DataInputNew.n_train_samples
    n_test_samples = DataInputNew.n_test_samples
    print(n_test_samples)
    train_loader = DataInputNew.multimodule.train_dataloader(batch_size = n_train_samples) # feature selection with all training

    test_loader = DataInputNew.multimodule.test_dataloader(batch_size= n_test_samples) # and all test
    # Load all the training and test data

    for train_data, train_mask, train_duration, train_event in train_loader:
        break

    for test_data, test_mask, test_duration, test_event in test_loader:
        break


    views_names = ['mRNA', 'DNA', 'microRNA', 'RPPA'] # replace with variable call from datainput
    views = len(train_data)


    print()
    print("Eigengene matrices based feature selection")

    counter = 0
    # durch einzelne views so mit loop gehen statt ganze df anzuschauen, da wir in eg_preprocess einzelne samples
    # für verschiedene views entfernen müssen --> im ganzen df strukturell schwieriger
    eigengene_duration_tensor = []
    eigengene_event_tensor = []
    # for test sets
    eigengene_duration_tensor_test = []
    eigengene_event_tensor_test = []
    for view in range(views):

        eg_view = F_eigengene_matrices(train=train_data[view], mask=train_mask[view], view_name=views_names[view], duration=train_duration, event=train_event, stage= 'train')
        eg_view_test = F_eigengene_matrices(train=test_data[view], mask=test_mask[view], view_name=views_names[view], duration=test_duration, event=test_event, stage='test')

        duration_tensor, event_tensor = eg_view.preprocess()
        duration_tensor_test, event_tensor_test =eg_view_test.preprocess()

        eigengene_duration_tensor.append(duration_tensor)
        eigengene_event_tensor.append(event_tensor)
        eigengene_duration_tensor_test.append(duration_tensor_test)
        eigengene_event_tensor_test.append(event_tensor_test)

        counter += 1
        if counter == views:
            eg_view.eigengene_multiplication()
            eigengene_matrices,eigengene_matrices_test = eg_view.get_eigengene_matrices(views_names)



    # as list as each eigengene matrix is of a different size
    eigengene_matrices_tensors = []
    eigengene_matrices_tensors_test = []
    for x in range(views):
        eigengene_matrices_tensors.append([])
        eigengene_matrices_tensors_test.append([])

    #Dataframe to tensor structure
    for c, view in enumerate(eigengene_matrices):
        for x in range(len(view.index)):
            temp = view.iloc[x, :].values.tolist()
            eigengene_matrices_tensors[c].append(temp)
        eigengene_matrices_tensors[c] = torch.tensor(eigengene_matrices_tensors[c])


    for c, view in enumerate(eigengene_matrices_test):
        for x in range(len(view.index)):
            temp = view.iloc[x, :].values.tolist()
            eigengene_matrices_tensors_test[c].append(temp)
        eigengene_matrices_tensors_test[c] = torch.tensor(eigengene_matrices_tensors_test[c])


    #eg_mRNA = F_eigengene_matrices(train= data[0], mask= mask[0], view ='mRNA')
    #eg_DNA = F_eigengene_matrices(train= data[1], mask= mask[1], view='DNA')
    #eg_microRNA = F_eigengene_matrices(train=data[2],mask=mask[2], view='microRNA')
    #eg_RPPA = F_eigengene_matrices(train=data[3],mask=mask[3], view='RPPA')

    #eg_mRNA.preprocess()
    #eg_DNA.preprocess()
    #eg_microRNA.preprocess()
    #eg_RPPA.preprocess()

    #eg_mRNA.eigengene_multiplication() # As of now, R file calculated eigengene for all types, thus one call is enough

    #mRNA_eigengene_matrix = (eg_mRNA.get_eigengene_matrix()).transpose()
    #DNA_eigengene_matrix = (eg_DNA.get_eigengene_matrix()).transpose()
    #microRNA_eigengene_matrix = (eg_microRNA.get_eigengene_matrix()).transpose()
    #RPPA_eigengene_matrix = (eg_RPPA.get_eigengene_matrix()).transpose()

    #mRNA_eigengene_tensor = []
    #DNA_eigengene_tensor = []
    #microRNA_eigengene_tensor = []
    # RPPA_eigengene_tensor = []

    # Dataframe to tensor structure
    #for x in range(len(mRNA_eigengene_matrix.index)):
    #    temp = mRNA_eigengene_matrix.iloc[x, :].values.tolist()
    #    mRNA_eigengene_tensor.append(temp)

    #mRNA_eigengene_tensor = torch.tensor(mRNA_eigengene_tensor)

    #for x in range(len(DNA_eigengene_matrix.index)):
    #    temp = DNA_eigengene_matrix.iloc[x, :].values.tolist()
    #    DNA_eigengene_tensor.append(temp)

    #DNA_eigengene_tensor = torch.tensor(DNA_eigengene_tensor)

    #for x in range(len(microRNA_eigengene_matrix.index)):
    #    temp = microRNA_eigengene_matrix.iloc[x, :].values.tolist()
    #    microRNA_eigengene_tensor.append(temp)

    #microRNA_eigengene_tensor = torch.tensor(microRNA_eigengene_tensor)

    #for x in range(len(RPPA_eigengene_matrix.index)):
    #    temp = RPPA_eigengene_matrix.iloc[x, :].values.tolist()
    #    RPPA_eigengene_tensor.append(temp)

    #RPPA_eigengene_tensor = torch.tensor(RPPA_eigengene_tensor)

    # only for train sets
    for c,view in enumerate(views_names):
        print("{} eigengene matrix : {} of size {}. Originally, we had {} features, now we have {}.".format
              (view,eigengene_matrices_tensors[c], eigengene_matrices_tensors[c].shape,len(DataInputNew.features[c]),
               eigengene_matrices_tensors[c].size(1)))

    #test set
    #  for c,view in enumerate(views_names):
    #      print("{} eigengene test matrix : {} of size {}. Originally, we had {} features, now we have {}.".format
    #            (view,eigengene_matrices_tensors_test[c], eigengene_matrices_tensors_test[c].shape,len(DataInputNew.features[c]),
    #             eigengene_matrices_tensors_test[c].size(1)))




    print("PCA based selection")
    PCA_tensors = []
    variance = [1,1,0.8,0.8] # TODO : Grid search
    duration_tensor_PCA = train_duration
    event_tensor_PCA = train_event
    for view in range(views):
        view_PCA = F_PCA(train_data[view], keep_variance=variance[view])
        pc_view = view_PCA.apply_pca()
        PCA_tensors.append(torch.tensor(pc_view))

    for c,view in enumerate(views_names):
        print("{} PCA feature selection : {} of size {} (samples, PC components).".format(view, PCA_tensors[c],
                                                                                          PCA_tensors[c].shape))

    #  mRNA_PCA = F_PCA(data[0], keep_variance= 0.9)
    #  principal_components_mRNA = mRNA_PCA.apply_pca()
    #  rows, columns = principal_components_mRNA.shape
    #  principal_components_mRNA_df = pd.DataFrame(principal_components_mRNA,
    #                                              columns= [["PC {}".format(x) for x in range(columns)]])

    # TODO : Feature selected Daten in csv laden oder direkt hier rüber
    # Tensors as basic data structure for all data
    #  principal_components_mRNA = torch.tensor(principal_components_mRNA)

    #   print("Reduction mRNA data with PCA :", 1 - (columns / len(DataInputNew.features[0])))
    # print("mRNA PCA feature selection : {} of size {} (samples, PC components).".format(principal_components_mRNA,
    #                                                             principal_components_mRNA.shape))


    #  DNA_PCA = F_PCA(data[1], keep_variance= 0.9)
    #  principal_components_DNA = DNA_PCA.apply_pca()
    #  rows, columns = principal_components_DNA.shape
    #  principal_components_DNA_df = pd.DataFrame(principal_components_DNA,
    #                                             columns= [["PC {}".format(x) for x in range(columns)]])

    #  principal_components_DNA = torch.tensor(principal_components_DNA)

    #  print("Reduction DNA data with PCA :", 1 - (columns / len(DataInputNew.features[1])))
    #  print("DNA PCA feature selection : {} of size {} (samples, PC components).".format(principal_components_DNA,
    #                                                           principal_components_DNA.shape))



    #  microRNA_PCA = F_PCA(data[2], keep_variance= 0.7)

    #  principal_components_microRNA = microRNA_PCA.apply_pca()
    #  rows, columns = principal_components_microRNA.shape

    #  principal_components_microRNA_df = pd.DataFrame(principal_components_microRNA,
    #                                                 columns= [["PC {}".format(x) for x in range(columns)]])


    #  principal_components_microRNA = torch.tensor(principal_components_microRNA)
    #  print("Reduction microRNA data with PCA :", 1 - (columns / len(DataInputNew.features[2])))
    #  print("microRNA PCA feature selection : {} of size {} (samples, PC components).".format(principal_components_microRNA,
    #                                                                 principal_components_microRNA.shape))


    #  RPPA_PCA = F_PCA(data[3], keep_variance= 0.7)
    #  principal_components_RPPA = RPPA_PCA.apply_pca()
    #  rows, columns = principal_components_RPPA.shape



    #  principal_components_RPPA_df = pd.DataFrame(principal_components_RPPA,
    #                                              columns= [["PC {}".format(x) for x in range(columns)]])


    #  principal_components_RPPA = torch.tensor(principal_components_RPPA)

    #  print("Reduction RPPA data with PCA :", 1 - (columns / len(DataInputNew.features[3])))
    #  print("RPPA PCA feature selection : {} of size {} (samples, PC-components).".format(principal_components_RPPA, principal_components_RPPA.shape))

    # ADD LABEL
    #principal_components_RPPA_df['duration'] = duration

    # TODO : in ConcatAE paper 1000 best features chosen, but for mRNA and DNA
    # TODO : we get about 1000-2000 even with a threshold of 1
    # need to save as list of tensors, bc each view has different subset of features selected
    # TODO : oder PCA mit best. vielen PC components nehmen, damit man dann gleiche Struktur hat ?
    # TODO : Listen könnten zeitintensiv werden? aber nur 4 views, deswegen okay ?
    #  data_PCA_selected_PRAD = [principal_components_mRNA, principal_components_DNA,
    #                                           principal_components_microRNA, principal_components_RPPA]
    print()
    print("Variance based selection")

    variance_tensors = []
    variance_selected_features = []
    duration_tensor_variance = train_duration
    event_tensor_variance = train_event
    thresholds = [0.8,0.8,0.8,0.6] # TODO : Grid Search

    for view in range(views):
        view_variance = F_VARIANCE(train_data[view], threshold= thresholds[view])
        data_variance, mask_variance = view_variance.apply_variance()
        variance_selected_features.append([DataInputNew.features[view][index] for index in mask_variance])
        variance_tensors.append(torch.tensor(data_variance))

    #  mRNA_variance = F_VARIANCE(data[0], threshold= 1)
    #  DNA_variance = F_VARIANCE(data[1], threshold= 1)
    #  microRNA_variance = F_VARIANCE(data[2], threshold = 0.8)
    #  RPPA_variance = F_VARIANCE(data[3], threshold = 0.6)

    #  data_mRNA_variance,mask_mRNA_variance = mRNA_variance.apply_variance()
    #  data_DNA_variance,mask_DNA_variance = DNA_variance.apply_variance()
    #  data_microRNA_variance,mask_microRNA_variance = microRNA_variance.apply_variance()
    #  data_RPPA_variance,mask_RPPA_variance = RPPA_variance.apply_variance()

    # mRNA_features_selected = [DataInputNew.features[0][index] for index in mask_mRNA_variance]
    # DNA_features_selected = [DataInputNew.features[1][index] for index in mask_DNA_variance]
    # microRNA_features_selected = [DataInputNew.features[2][index] for index in mask_microRNA_variance]
    # RPPA_features_selected = [DataInputNew.features[3][index] for index in mask_RPPA_variance]

    # data_mRNA_variance = torch.tensor(data_mRNA_variance)
    # data_DNA_variance = torch.tensor(data_DNA_variance)
    # data_microRNA_variance = torch.tensor(data_microRNA_variance)
    # data_RPPA_variance = torch.tensor(data_RPPA_variance)


    for c,view in enumerate(views_names):
        print("Reduction {} data with variance :".format(view), 1 - (len(variance_selected_features[c]) / len(DataInputNew.features[c])))
        print("{} variance feature selection : {} of size : {} (samples, latent features)".format(view, variance_tensors[c],
                                                                                                  variance_tensors[c].shape))

    print()
    print("Autoencoder based selection")

    AE_all_compressed_features = []
    duration_tensor_AE = train_duration
    event_tensor_AE = train_event

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for x in range(views):
        model = F_AE(train= train_data[x]).to(device)

        optimizer = Adam(model.parameters(), lr=1e-3)

        criterion = nn.MSELoss() # reconstrution loss
        # TODO : oben ein trainloader, hier anderen --> andere samples werden getestet ; später für benchmark schlecht ?
        train_loader_2 = DataInputNew.multimodule.train_dataloader(batch_size = 80)         # TODO: training samples gibt es 389 viele, Primzahl : kein batch außer 1 um alle durchzugehen ; für 80 werden nur 320 samples durchgegangen und die letzten samples nicht beachtet !
        epochs = 1
        temp = []
        temp2 = []
        for epoch in range(epochs):
            loss = 0
            for batch_data, train_mask, train_duration, train_event in train_loader_2:

                batch_data = batch_data[x].view(-1, batch_data[x].size(1)).to(device) #mRNA

                optimizer.zero_grad()

                # compressed features is what we are interested in
                reconstructed, compressed_features = model(batch_data)
                if epoch == epochs - 1: # save compressed_features of last epoch for each batch
                    temp.append(compressed_features) # list of tensors of compressed for each batch

                train_loss = criterion(reconstructed, batch_data)

                train_loss.backward()

                optimizer.step()

                loss += train_loss.item()



            loss = loss / len(train_loader_2)

            print("epoch : {}/{}, loss = {:.6f} for {} data".format(epoch + 1, epochs, loss, views_names[x]))


        compressed_features_view = torch.cat(temp, 0)
        AE_all_compressed_features.append(compressed_features_view)

    for x in range(len(AE_all_compressed_features)):
        AE_all_compressed_features[x] = torch.detach(AE_all_compressed_features[x]) #detach gradient as we only need
        # selected features

    for c, view in enumerate(views_names):
        print("{} AE feature selection : {} of size {} (samples,features)".format(view, AE_all_compressed_features[c],
                                                                                  AE_all_compressed_features[c].shape))


    # TODO : lots of values pressed to 0 ! --> less layers, other activation ?

    def tensor_helper(tensor_list):
        """Turns a list of size (x) of tensors with dimensions (y,z) into a tensor of dimension (x,y,z)"""
        x = len(tensor_list)
        z, y = tensor_list[0].shape

        tensor_new = torch.zeros((x, z, y))
        for i, tensor in enumerate(tensor_list):
            tensor_new[i, :, :] = tensor

        return tensor_new

    # Here we can create a tensor for all data, because we have the same feature size for each view
    # due to AE feature selection
    # (views, samples, features)
    data_AE_selected_PRAD = tensor_helper(AE_all_compressed_features)


    """
    print()
    print("Protein-Protein-Network")
    ppn_mRNA = F_PPI_NETWORK(data[0])
    duration_tensor_PPI = duration
    event_tensor_PPI = event
    adjacency_matrix, feature_matrices = ppn_mRNA.setup()
    print("Adjacency matrix : {}".format(adjacency_matrix))
    print("Feature matrix mRNA : {} of size {} (samples, proteins, features). "                                         
          "For each sample, we have {} proteins and {} possible features".format
          (feature_matrices, feature_matrices.shape, feature_matrices.size(1), feature_matrices.size(2)))
    
    """

