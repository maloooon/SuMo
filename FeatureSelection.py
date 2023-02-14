from typing import List
import torch
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from torch import nn
import os
import subprocess
import HelperFunctions as HF
from collections import defaultdict
import gzip



class F_PCA():

    """
    Principal Component based Feature Selection.
    Based on : https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """

    def __init__(self, data, components = 10):
        """
        :param data: Data input of one view ; dtype : Tensor TODO :check if numpy array or tensor
        :param components: Number of Principal Components ; dtype : Int
        """
        self.components = components
        self.data = data

    def apply_pca(self):
        """
        Apply PCA
        :return: PCA object
        """
        pca = PCA(n_components= self.components)

        return pca


    def fit_transform_pca(self,pca):
        """
        Fit and transform data with PCA ; This is used for training data
        :param pca: PCA object
        :return: data after fitting and transforming it using PCA ; dtype : Tensor TODO : check
        """
        train_data = pca.fit_transform(self.data)

        return train_data


    def transform_pca(self,pca):
        """
        Transform data with PCA ; This is used for validation and test data, as we transform the data based on the fitted
                                  train representation
        :param pca: PCA object
        :return: data after transforming it using PCA ; dtype : Tensor TODO : check
        """
        test_data = pca.transform(self.data)

        return test_data



class F_VARIANCE():
    """
    Variance based Feature Selection.
    Remove features based on a variance threshold
    Based on : https://scikit-learn.org/stable/modules/feature_selection.html
    """

    def __init__(self, data, threshold = 0.5):
        """

        :param data: data: Data input of one view ; dtype : Tensor TODO :check if numpy array or tensor
        :param threshold: Drop all features where 100 - threshold * 100 % of the values are similar; dtype : Float
        """
        self.data = data
        self.threshold = threshold



    def apply_variance(self):
        """
        Apply Variance Threshold
        :return: Variance threshold object
        """


        vt = VarianceThreshold(self.threshold)
        return vt

    def fit_transform_variance(self, vt):
        """
        Fit and transform data with Variance threshold ; This is used for training data
        :param vt: Variance threshold object
        :return: data after fitting and transforming it using Variance threshold ; dtype : Tensor TODO : check
        """

        train_data = vt.fit_transform(self.data)
        return train_data


    def transform_variance(self,vt):
        """
        Transform data with Variance threshold ; This is used for validation and test data,
                                                 as we transform the data based on the fitted
                                                 train representation
        :param vt: Variance threshold object
        :return: data after transforming it using Variance threshold ; dtype : Tensor TODO : check
        """

        test_data = vt.transform(self.data)
        return test_data


class PPI():
    """Protein-Protein-Interaction Feature Selection.
       Create matrices storing protein to feature value mappings sample wise.
       Protein data from String DB.
    """

    def __init__(self,data, feature_names, view_names):
        """
        :param data: Data input ; dtype : List of Tensors TODO: dtype check
        :param feature_names: Names of Features for all views ;
                              dtype : Tuple(Name of View, List of Lists containing feature names for this view)
        :param view_names: Names of Views ; dtype : List of Strings
        """
        self.data = data
        self.feature_names = feature_names
        self.view_names = [x.upper() for x in view_names]

        # Only DNA & mRNA data contains protein data
        if 'DNA' not in self.view_names and 'MRNA' not in self.view_names:
            raise Exception("neither DNA nor mRNA data in input : no protein data.")



        # Check which View names we have and if needed, delete features from feature_names (bc View may have been
        # deleted in preprocessing due to too many missing values)
        temp = []
        for x in self.feature_names:
            view_name = x[0].upper()
            if view_name in self.view_names:
                temp.append(x[1])

        self.feature_names = temp




    def get_matrices(self):
        """
        Create Feature Matrix A x B (A = proteins, B = Feature values) for each sample, Edge Indices mapping (store
        protein pairs which have an interaction)
        :return:
        """

        # Read in protein data
        print("Reading in protein data...")
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

        print("Mapping features to proteins ...")
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
        print("Mapping feature values to proteins...")
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

        checker = []
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
                    # Also check, that the current combination of proteins is not already in our interactions
                    if protein1 in proteins_used and protein2 in proteins_used:
                     #   curr_edge = [proteins_used[protein1],proteins_used[protein2]]
                     #   if curr_edge in checker or curr_edge.reverse() in checker:
                     #       pass
                     #   else:
                        interactions1.append(proteins_used[protein1])
                        interactions2.append(proteins_used[protein2])
                       #     checker.append([proteins_used[protein1],proteins_used[protein2]])

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

    def __init__(self,train,mask,view_name,duration,event,stage = 'train', cancer_name = None):
        self.train = train
        self.stage = stage # test or train stage
        self.mask = mask
        self.view_name = view_name
        self.duration = duration
        self.event = event
        self.cancer_name = cancer_name

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
            train_df.to_csv("/Users/marlon/Desktop/Project/TCGAData/{}/{}_for_r.csv".format(self.cancer_name,self.view_name))
        elif self.stage == 'val':
            train_df.to_csv("/Users/marlon/Desktop/Project/TCGAData/{}/{}_val_for_r.csv".format(self.cancer_name,self.view_name))

        else: # self.stage == 'test'
            train_df.to_csv("/Users/marlon/Desktop/Project/TCGAData/{}/{}_test_for_r.csv".format(self.cancer_name,self.view_name))


        # return these so we can add them to according samples ; save dropped columns
      #  return torch.tensor(duration_df.iloc[:, 0].values), torch.tensor(event_df.iloc[:, 0].values)


    def eigengene_multiplication(self):
        """We calculate the eigengene matrix for all views
        need to set rights first so Python can read it : chmod a+rwx 'PATHTOFILE'
        """

        path = ["Rscript /Users/marlon/DataSpellProjectsForSAMO/SAMO/eigengene_matrices.R"]

        rscript = ["/usr/local/bin/Rscript"]

        commands = path + rscript
        cancer_R = [self.cancer_name]
        subprocess.call(commands + cancer_R, shell=True)



    def get_eigengene_matrices(self,views):
        """Loading the eigengene matrix for a sample as a panda Dataframe created from R."""
        eigengene_matrices = []
        eigengene_val_matrices = []
        eigengene_test_matrices =[]
        for view in views:
    #        while not os.path.exists(os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
    #                                              "{}_eigengene_matrix.csv".format(view))):
    #            time.sleep(1)


            eigengene_matrix = pd.read_csv(os.path.join("/Users", "marlon", "Desktop", "Project","TCGAData", "{}/"
                                                         "{}_eigengene_matrix.csv".format(self.cancer_name,view)),
                                           index_col=0)

            eigengene_val_matrix = pd.read_csv(os.path.join("/Users", "marlon", "Desktop", "Project","TCGAData", "{}/"
                                                            "{}_val_eigengene_matrix.csv".format(self.cancer_name,view)),
                                               index_col=0)
    # /Users/marlon/Desktop/Project/TCGAData/READ/microRNA_eigengene_matrix.csv
            eigengene_test_matrix = pd.read_csv(os.path.join("/Users", "marlon", "Desktop", "Project", "TCGAData", "{}/"
                                                             "{}_test_eigengene_matrix.csv".format(self.cancer_name,view)),
                                                index_col=0)

            # reset index bc R saves starting at index 1
            eigengene_matrix.reset_index()
            eigengene_val_matrix.reset_index()
            eigengene_test_matrix.reset_index()

            eigengene_matrices.append(eigengene_matrix)
            eigengene_val_matrices.append(eigengene_val_matrix)
            eigengene_test_matrices.append(eigengene_test_matrix)
        return eigengene_matrices, eigengene_val_matrices, eigengene_test_matrices




class F_AE(nn.Module):
    """Use the hidden layer between Encoder and Decoder.
    Based on : https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
    """
# TODO : Modell konvergiert ?
# TODO : test : variance-explained : wv varianz explained die approximation (encoder_output_layer) vs original daten
# TODO : preprocessing AE centering & standardizing


#TODO : AE implementation ersetzen mit der aus AE Modul (nur einfache Variante, aber einstellbar wv. Layer etc...)
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









