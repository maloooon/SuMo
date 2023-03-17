import torch
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import os
import subprocess
import HelperFunctions as HF
from collections import defaultdict
import gzip
import copy



class F_PCA():

    """
    Principal Component based Feature Selection.
    Based on : https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """

    def __init__(self, data, components = 10):
        """
        :param data: Data input of one view ; dtype : Tensor
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
        :return: data after fitting and transforming it using PCA ; dtype : ndarray of shape (n_samples, n_components)
        """
        train_data = pca.fit_transform(self.data)

        return train_data


    def transform_pca(self,pca):
        """
        Transform data with PCA ; This is used for validation and test data, as we transform the data based on the fitted
                                  train representation
        :param pca: PCA object
        :return: data after transforming it using PCA ; dtype : ndarray of shape (n_samples, n_components)
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

        :param data: data: Data input of one view ; dtype : Tensor
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
        :return: data after fitting and transforming it using Variance threshold ; dtype : ndarray array of shape (n_samples, n_features_new)
        """

        train_data = vt.fit_transform(self.data)
        return train_data


    def transform_variance(self,vt):
        """
        Transform data with Variance threshold ; This is used for validation and test data,
                                                 as we transform the data based on the fitted
                                                 train representation
        :param vt: Variance threshold object
        :return: data after transforming it using Variance threshold ; dtype : ndarray array of shape (n_samples, n_features_new)
        """

        test_data = vt.transform(self.data)
        return test_data


class PPI():
    """Protein-Protein-Interaction Feature Selection.
       Create matrices storing protein to feature value mappings sample wise.
       Protein data from String DB.
    """

    def __init__(self,data, feature_names, view_names,columns_removed):
        """
        :param data: Data input ; dtype : List of Tensors(n_samples, n_features) [Tensor for each view]
        :param feature_names: Names of Features for all views ;
                              dtype : Tuple(Name of View (String), List containing feature names (Strings) for this view)
        :param columns_removed : Removed features due to preprocessing ; List of strings
        :param view_names: Names of Views ; dtype : List of Strings
        """
        self.data = data
        self.feature_names = feature_names
        self.view_names = [x.upper() for x in view_names]
        self.columns_removed = columns_removed
        self.direc_set = 'SUMO'

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

        temp = [[] for i in range(len(self.view_names))]
        for c, view in enumerate(self.feature_names):
            for feat_name in view:
                if feat_name not in columns_removed:
                    temp[c].append(feat_name)


        # Delete features which have been deleted due to preprocessing
        self.feature_names = temp

        # temp is a list of lists of each views remaining feature names




    def get_matrices(self):
        """
        Create Feature Matrix A x B (A = proteins, B = Feature values) for each sample and Edge Indices mapping
        (protein pairs which have an interaction)
        :return: features_used : feature values with protein mappings for each sample
                               ; dtype : Tensor(n_samples,n_features,n_feature_values)
                 edge_index : Pairs of Protein (Indices) which have an interaction
                              ; dtype : List of Lists
                 proteins_used : Protein - Index mapping ; dtype : Dictionary(Protein - Index)
        """

        print("Reading in protein data...")

        prot_to_feat = pd.read_csv(
            os.path.join("~", self.direc_set, "Project", "ProteinToFeature.csv"),index_col=0)


        samples = self.data[0].size(0)

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
        # Also store indices just for the current cancer, as we need that to access the correct values in data
        cancer_features_mapped_indices = [[] for _ in range(len(self.feature_names))]

        print("Mapping features to proteins ...")
        # Store proteins with indices
        proteins_used = {}

        for c,view in enumerate(fixed_features):
            # microRNA/RPPA no protein data
            if self.view_names[c] == 'MIRNA' or self.view_names[c] == 'RPPA':
                continue
            for c2,_ in enumerate(view):
                # If there is a mapping ...
                if fixed_features[c][c2] in all_features:
                    # Fixed features are the features of the views of our current cancer data
                    idx = all_features.index(fixed_features[c][c2])
                    # Then we can add the protein to used proteins ...
                    if all_proteins[idx] not in proteins_used:
                        proteins_used[all_proteins[idx]] = len(proteins_used)
                    all_features_mapped_indices[c].append(idx) # This will help us to get the index for the right protein
                    cancer_features_mapped_indices[c].append(c2) # This will help us to get the index for the right feature values


        all_mappings = []
        all_mappings_indices = []


        for sample in range(samples):
            # Dictionary to store protein - feature value // for each view single list,
            # We will use that to calculate the median

            # Only mRNA/DNA has protein data
            co1 = self.view_names.count('MRNA')
            co2 = self.view_names.count('DNA')
            co_sum = co1+co2

            prot_to_feat_values = defaultdict(lambda: [[] for x in range(co_sum)])

            # Track indices for median (so we know whether added element is from mRNA or DNA)
            prot_to_feat_values_indices = defaultdict(list)


            for c,view in enumerate(cancer_features_mapped_indices):
                # microRNA/RPPA no protein-protein data
                if self.view_names[c] == 'MIRNA' or self.view_names[c] == 'RPPA':
                    continue
                for c2,_ in enumerate(view):
                    idx = all_features_mapped_indices[c][c2]
                    # Protein - Feature value mapping
                    prot_to_feat_values[all_proteins[idx]][c].append(self.data[c][sample,_].item())
                    # Protein - Feature value view type mapping
                    prot_to_feat_values_indices[all_proteins[idx]].append(c)
            # All mappings : for all samples
            all_mappings.append(prot_to_feat_values)
            all_mappings_indices.append(prot_to_feat_values_indices)


        # For missing values for certain feature values in protein mapping, we take the median of features for
        # this view
        all_medians = []

        # For each sample ...
        for c,mapping in enumerate(all_mappings):

            # Calculate the medians
            medians = [[] for i in range(co_sum)]
            for feature_values_listed in all_mappings[c].values():
                # For each view ...
                for c,feature_value in enumerate(feature_values_listed):
                    # If there is a feature value for that view
                    if len(feature_value) != 0:

                        medians[c].append(feature_value[0])

            for c,_ in enumerate(medians):

                medians[c] = (sum(medians[c])) / len(medians[c])
            all_medians.append(medians)





        protein_to_delete = [[] for _ in range(samples)]
        temp = copy.deepcopy(all_mappings)
        # Add Median values if needed
        print("Mapping feature values to proteins...")
        features_used = [[] for _ in range(samples)]
        # For all samples ...
        for c,mapping in enumerate(temp):       # all_mappings
            # For certain sample ...
            for c2,protein_idx in enumerate(mapping):
                # For certain protein in sample ...
                for c3, feat_values in enumerate(mapping[protein_idx]):
                    # If we have no feature value for this view, we add the median #

                    # Proteins which don't have mappings to each view that could possibly have one will be deleted
                    if len(feat_values) == 0:
                        protein_to_delete[c].append(protein_idx)
                        del all_mappings[c][protein_idx]
                        # Also delete from used proteins
                        try:
                            del proteins_used[protein_idx]
                        except KeyError:
                            pass

                        # set median in place
                    #   all_mappings[c][protein_idx][c3].append(all_medians[c][c3])
                    # Try setting 0 instead of median and check if results get better
                    #   all_mappings[c][protein_idx][c3].append(0)







            #for feat_values_listed in mapping.values():
            for feat_values_listed in all_mappings[c].values():

                features_dlisted = HF.flatten(feat_values_listed)

                # Store feature values as themselves, as we can now access their respective protein via index of proteins_used
                features_used[c].append(features_dlisted)


        # also delete from used proteins


        # Turn into tensor
        features_used = torch.tensor(features_used)

        # Find Protein-Protein Interactions
        interactions1 = []
        interactions2 = []
        dir = os.path.expanduser('~/{}/Project/9606.protein.links.v11.5.txt.gz'.format(self.direc_set))
        with gzip.open(dir, 'rt') as f:
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
                        interactions1.append(proteins_used[protein1])
                        interactions2.append(proteins_used[protein2])


        edge_index = [interactions1, interactions2]



        return features_used,edge_index, proteins_used



class F_eigengene_matrices():
    """
    Eigengene Matrices Feature Selection.
    Calculate the gene co-expression module, then the expression values and summarize these into the
    first principal component using singular value decomposition (SVD) , which results in an eigengene matrix.
    Based on : https://github.com/huangzhii/lmQCM
    """

    def __init__(self,data,mask,view_name,duration,event,stage = 'train', cancer_name = None):
        """
        :param data: Data ; dtype : Tensor(n_samples,n_features)
        :param mask: Mask for NaN values ; dtype : Tensor(n_samples,n_features)
        :param view_name: Name of current view ; dtype : String
        :param duration: Duration value ; dtype : Tensor(n_samples,)
        :param event: Event value ; dtype : Tensor(n_samples,)
        :param stage: Current stage ; dtype : String ['train'/'val'/'test']
        :param cancer_name: Current cancer name ; dtype : String
        """
        self.data = data
        self.stage = stage
        self.mask = mask
        self.view_name = view_name
        self.duration = duration
        self.event = event
        self.cancer_name = cancer_name
        self.direc_set = 'SUMO'

    def preprocess(self):
        """
        For the eigengene matrices computation, we need to remove rows (samples) which had all NaN values
        (missing sample) aswell as each column (feature), which had the same value for each patient (sample) (this feature
        is irrelevant either way, as it doesn't have any impact on any patient)
        """


        data_df = pd.DataFrame(self.data.numpy())


        to_drop_columns = []

        for x in range(len(data_df.columns)):
            # If column (feature) has same values for each row (patient)
            if (data_df.iloc[:, x] == data_df.iloc[:, x][0]).all():

                to_drop_columns.append(x)

        data_df = data_df.drop(data_df.columns[to_drop_columns], axis = 1)

        to_drop_index = []

        for x in range(len(data_df.index)):

            # If row (sample) had all NaN values
            if torch.all(self.mask[x] == True):

                to_drop_index.append(x)


        data_df = data_df.drop(to_drop_index, axis = 0)



        if self.stage == 'train':
            dir = os.path.expanduser('~/{}/Project/TCGAData/{}/{}_for_r.csv'.format(self.direc_set,self.cancer_name,self.view_name.upper()))

            data_df.to_csv(dir)
        elif self.stage == 'val':
            dir = os.path.expanduser('~/{}/Project/TCGAData/{}/{}_val_for_r.csv'.format(self.direc_set,self.cancer_name,self.view_name.upper()))
            data_df.to_csv(dir)


        else:
            dir = os.path.expanduser('~/{}/Project/TCGAData/{}/{}_test_for_r.csv'.format(self.direc_set,self.cancer_name,self.view_name.upper()))
            data_df.to_csv(dir)



    def eigengene_multiplication(self):
        """
        We calculate the eigengene matrices for all views.
        Need to set rights first so Python can read it : chmod a+rwx 'PATHTOFILE'
        """

        path = ["Rscript /Users/marlon/Desktop/SuMO/eigengene_matrices.R"]

        rscript = ["/usr/local/bin/Rscript"]

        commands = path + rscript
        cancer_R = [self.cancer_name]
        subprocess.call(commands + cancer_R, shell=True)



    def get_eigengene_matrices(self,views):
        """
        Loading the eigengene matrix for a sample as a panda Dataframe created from R.
        :param views: All used views ; dtype : List [of Strings]
        :return: Eigengene Matrices for train, validation, test data ; dtype : DataFrame(n_samples, n_PC of expression values)
        """

        eigengene_matrices = []
        eigengene_val_matrices = []
        eigengene_test_matrices =[]
        for view in views:


            eigengene_matrix = pd.read_csv(os.path.join("~",self.direc_set, "Project","TCGAData", "{}/"
                                                                                                  "{}_eigengene_matrix.csv".format(self.cancer_name,view)),
                                           index_col=0)

            eigengene_val_matrix = pd.read_csv(os.path.join("~",self.direc_set, "Project","TCGAData", "{}/"
                                                                                                      "{}_val_eigengene_matrix.csv".format(self.cancer_name,view)),
                                               index_col=0)

            eigengene_test_matrix = pd.read_csv(os.path.join("~",self.direc_set, "Project", "TCGAData", "{}/"
                                                                                                        "{}_test_eigengene_matrix.csv".format(self.cancer_name,view)),
                                                index_col=0)

            # Reset index bc R saves starting at index 1
            eigengene_matrix.reset_index()
            eigengene_val_matrix.reset_index()
            eigengene_test_matrix.reset_index()

            eigengene_matrices.append(eigengene_matrix)
            eigengene_val_matrices.append(eigengene_val_matrix)
            eigengene_test_matrices.append(eigengene_test_matrix)

        return eigengene_matrices, eigengene_val_matrices, eigengene_test_matrices




