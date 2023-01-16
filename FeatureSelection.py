
# sk learn feature selection methods
# https://scikit-learn.org/stable/modules/feature_selection.html


from typing import Dict, List, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from torch import nn
from torch.optim import Adam
import os
import time
import math
import subprocess
import HelperFunctions as HF
from collections import defaultdict
import gzip



class F_PCA():

    input : List[torch.Tensor]
    components : int

    def __init__(self, train, components = 80, keep_variance = 0.3):
        """

        :param train:
        :param components: choose number of components one wants to use
        :param keep_variance: choose minimal number of PC components such that x % variance is retained
        """
        self.components = components
        self.train = train #training input
        self.keep_variance = keep_variance


    def apply_pca(self):
        self.pca = PCA(n_components= self.components)


        #As we already standardized & transformed the data with StandardScaler(), we can apply PCA directly
        self.pca.fit(self.train)
        pca_train_data = self.pca.transform(self.train)

        return pca_train_data




# TODO : Variance probelmatic : can't choose best 100 features for each view! only via threshold
class F_VARIANCE():
    """Removing features with low variance
       https://scikit-learn.org/stable/modules/feature_selection.html
       https://towardsdatascience.com/how-to-use-variance-thresholding-for-robust-feature-selection-a4503f2b5c3f
       Unsupervised variance-based feature selection
    """

    def __init__(self, train, features = None, test = None, threshold = 0.5):
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





class PPI():

    def __init__(self,train, feature_names, view_names):
        """

        :param train:
        :param feature_names: tuple : (view_name, list of lists containing feature names (original ones) for this view)
        :param view_names: names of views
        """
        self.train = train
        self.feature_names = feature_names

        self.view_names = [x.upper() for x in view_names]

        if 'DNA' not in self.view_names or 'MRNA' not in self.view_names:
            raise Exception("neither DNA nor mRNA data in input : no protein data.")



        # Check which view names we have and if needed, delete features from feature_names (bc view may have been
        # deleted in preprocessing)
        temp = []
        for x in self.feature_names:
            view_name = x[0].upper()
            if view_name in self.view_names:
                temp.append(x[1])

        self.feature_names = temp




    def get_matrices(self):
        # Read in protein data
        prot_to_feat = pd.read_csv(
            os.path.join("/Users", "marlon", "Desktop", "Project", "ProteinToFeature.csv"),index_col=0)

        # For each sample an adjacency matrix and a feature matrix is to be created

        # Feature matrix : size AxB, where A : proteins, B feature values

        samples = self.train[0].size(0)

        # Lists of all feature names with protein mapping and all according proteins
        all_features = list(prot_to_feat.iloc[:,1].values)
        all_proteins = list(prot_to_feat.iloc[:,0].values)




        # First, get feature names into the right structure
        fixed_features = []

        for idx,view in enumerate(self.feature_names):
            features_fix = []
            for idx2, feat in enumerate(view):

                index_1 = feat.find(':')
                index_2 = feat.find('|')


                if index_1 != -1:
                    if index_2 != -1:
                        features_fix.append(feat[index_1+1:index_2].upper())
                    else:
                        features_fix.append(feat[index_1+1:].upper())
                else:
                    if index_2 != -1:
                        features_fix.append(feat[0:index_2].upper())
                    else:
                        features_fix.append(feat.upper())

                features_fix[-1] = features_fix[-1].replace('-','')

            fixed_features.append(features_fix)


        # Find features for which we have protein mappings
        # This list will store all indices that have mappings so we can access feature names and proteins by it
        all_features_mapped_indices = [[] for _ in range(len(self.feature_names))]
        # also store indices just for the current cancer, as we need that to access the correct values in self.train
        cancer_features_mapped_indices = [[] for _ in range(len(self.feature_names))] # len is amount views

        # Store proteins with indices
        proteins_used = {}

        for c,view in enumerate(fixed_features):
            # miRNA/RPPA no protein data
            if self.view_names[c] == 'MIRNA' or self.view_names[c] == 'RPPA':
                continue
            for c2,_ in enumerate(view):
                if fixed_features[c][c2] in all_features:
                    # fixed features are the features of our cancer data
                    idx = all_features.index(fixed_features[c][c2])
                    if all_proteins[idx] not in proteins_used:
                        proteins_used[all_proteins[idx]] = len(proteins_used)
                    all_features_mapped_indices[c].append(idx) # this will help us to get the index for the right protein
                    cancer_features_mapped_indices[c].append(c2) # this will help us to get the index for the right feature values


        all_mappings = []
        all_mappings_indices = []


        for sample in range(samples):
            # Dictionary to store protein - feature value // for each view single list,
            # we'll need that to calculate the median more easily
            # since now it is clear, which tensor values accord to which view


            # Only mRNA/DNA has protein data
            co1 = self.view_names.count('MRNA')
            co2 = self.view_names.count('DNA')
            co_sum = co1+co2

            prot_to_feat_values = defaultdict(lambda: [[] for x in range(co_sum)])

            # track indices for median (so we know whether added element is from mRNA or DNA
            prot_to_feat_values_indices = defaultdict(list)


            for c,view in enumerate(cancer_features_mapped_indices):
                # miRNA/RPPA no protein-protein data
                if self.view_names[c] == 'MIRNA' or self.view_names[c] == 'RPPA':
                    continue
                for c2,_ in enumerate(view):
                    idx = all_features_mapped_indices[c][c2]
                    prot_to_feat_values[all_proteins[idx]][c].append(self.train[c][sample,_].item())
                    prot_to_feat_values_indices[all_proteins[idx]].append(c)
            all_mappings.append(prot_to_feat_values)
            all_mappings_indices.append(prot_to_feat_values_indices)


        # for missing values for certain feature values in protein mapping, we take the median of features for
        # this view
        all_medians = []
        # First, find protein which has the most feature values (so that we know for how many dimensions we have
        # to create medians)
        for c,mapping in enumerate(all_mappings):

            # now calculate the medians
            medians = [[] for i in range(co_sum)]
            for feature_values_listed in all_mappings[c].values():
                for c,feature_value in enumerate(feature_values_listed):
                    # if there is a feature value for that view
                    if len(feature_value) != 0:

                        medians[c].append(feature_value[0])

            for c,_ in enumerate(medians):

                medians[c] = (sum(medians[c])) / len(medians[c])
            all_medians.append(medians)




        # store feature values as themselves, as we can now access their respective protein via index of proteins_used

        features_used = [[] for _ in range(samples)]
        for c,mapping in enumerate(all_mappings):
            for c2,protein_idx in enumerate(mapping):
                for c3, feat_values in enumerate(mapping[protein_idx]):
                    if len(feat_values) == 0:
                        # set median in place
                        all_mappings[c][protein_idx][c3].append(all_medians[c][c3])

            for feat_values_listed in mapping.values():

                features_dlisted = HF.flatten(feat_values_listed)# dlisted : list of lists


                features_used[c].append(features_dlisted)


        # turn into tensor
        features_used = torch.tensor(features_used)


        interactions1 = []
        interactions2 = []
        with gzip.open('/Users/marlon/Desktop/Project/9606.protein.links.v11.5.txt.gz', 'rt') as f:
            next(f) # Ignore the header
            for line in f:
                protein1, protein2, score = line.strip().split()
                score = int(score)
                if score >= 700: # Filter interactions with more confidence
                    protein1 = protein1.split('.')[1]
                    protein2 = protein2.split('.')[1]
                    # First, check that both proteins in the interaction are in our mappings
                    if protein1 in proteins_used and protein2 in proteins_used:
                        interactions1.append(proteins_used[protein1])
                        interactions2.append(proteins_used[protein2])

        edge_index = [interactions1, interactions2]


        # features_used : all samples and used features as list of tensors, respective protein to be found via index
        #                 of proteins_used
        # proteins_used : proteins with indices
        # edge_index : interaction of proteins via indices



        return features_used,edge_index, proteins_used














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
        duration_df = pd.DataFrame(self.duration)#.numpy())
        event_df = pd.DataFrame(self.event) #.numpy())



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

       # train_df.reset_index()
       # duration_df.reset_index()
       # event_df.reset_index()
        print("Index dropped : ", to_drop_index)

        if self.stage == 'train':
            train_df.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/{}_for_r.csv".format(self.view_name))
        else: # self.stage == 'test'
            train_df.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/{}_test_for_r.csv".format(self.view_name))


        # return these so we can add them to according samples ; save dropped columns
      #  return torch.tensor(duration_df.iloc[:, 0].values), torch.tensor(event_df.iloc[:, 0].values)


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

            # reset index bc R saves starting at index 1
            eigengene_matrix.reset_index()
            eigengene_test_matrix.reset_index()

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


        # For each sample, make a dictionary with protein : feature value ; save list of dictionaries
        features_values_proteins_mapping_mRNA = []




        # Go through each mRNA sample
        for sample in self.train[0]:
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






