from typing import Dict, List, Tuple
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pycox.datasets import metabric
from pycox.models import logistic_hazard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from torch.utils.data import DataLoader, Dataset
import math
import copy
from itertools import chain


class MultiOmicsDataset(Dataset):

    # [tensor von allen samples view 1,  tensor von allen samples view 2, ... ]
    # in dem tensor jeweils features sample1, features sample 2, ...

    def __init__(self, X, duration, event):
        self.X = [torch.nan_to_num(x_view) for x_view in X]
        self.n_samples = X[0].size(0) # number samples
        self.n_views = len(X) # number views (mRNA, DNA, microRNA, RPPA)
        self.mask = [torch.isnan(x_view) for x_view in X] #List of booleans for each numeric value in samples ;True : NaN values
        self.duration = duration
        self.event = event


        # Check if for each view (each tensor containing all samples for one view) the amount of samples
        # is the same as the sample size, otherwise print mismatch
        assert all(
            view.size(0) == self.n_samples for view in X
        ), "Size mismatch between tensors"



    def __len__(self):
        """Return the amount of samples"""
        return self.n_samples


    def __getitem__(self, index):
        """return the whole sample (all views)"""
        #index, : slicing for multidimensional array (multidim tensor)
        return [self.X[m][index, :] for m in range(self.n_views)], \
               [self.mask[m][index, :] for m in range(self.n_views)], \
               self.duration[index], self.event[index]



def preprocess_features(
        df_train: pd.DataFrame, # train dataset
        df_test: pd.DataFrame, # test dataset
        cols_std: List[str], # feature names, numeric variables
        cols_leave: List[str], # feature names, binary variables
        feature_offset : List[int]
) -> Tuple[torch.Tensor]:
    """Preprocess different data
    #Numeric variables: Standardize
    #Binary variables: No preprocessing necessary
    (#Categorical variables: Create embeddings)
    see pycox tutorial  : https://nbviewer.org/github/havakv/pycox/blob/master/examples/01_introduction.ipynb
    or https://towardsdatascience.com/how-to-implement-deep-neural-networks-for-time-to-event-analyses-9aa0aeac4717
    """
    if cols_std is not None:
        standardize = [([col], StandardScaler()) for col in cols_std]
        leave = [(col, None) for col in cols_leave]
        # map together so we have all features present again
        mapper = DataFrameMapper(standardize + leave)
        x_train = mapper.fit_transform(df_train).astype(np.float32)
        x_test = mapper.transform(df_test).astype(np.float32)
        x_train_df = pd.DataFrame(x_train)
        x_test_df = pd.DataFrame(x_test)

        # Order by view so it works with Dataset class
        x_train_ordered_by_view = []
        x_test_ordered_by_view = []

        for x in range(len(feature_offset) - 3): # -3 bc we don't have duration/event in training tensor
            x_train_ordered_by_view.append(torch.tensor((x_train_df.iloc[:, feature_offsets[x]: feature_offsets[x + 1]]).values))
            x_test_ordered_by_view.append(torch.tensor((x_train_df.iloc[:, feature_offsets[x]: feature_offsets[x + 1]]).values))




    else:
        # Order by view so it works with Dataset class
        x_train_ordered_by_view = []
        x_test_ordered_by_view = []
        for x in range(len(feature_offset) - 3): # -3 bc we don't have duration/event in training tensor
            x_train_ordered_by_view.append(torch.tensor((df_train.iloc[:, feature_offsets[x]: feature_offsets[x + 1]]).values))
            x_test_ordered_by_view.append(torch.tensor((df_train.iloc[:, feature_offsets[x]: feature_offsets[x + 1]]).values))



    return x_train_ordered_by_view, x_test_ordered_by_view


class SurvMultiOmicsDataModule(pl.LightningDataModule):
    """Input is the whole dataframe : We merge all data types together, with the feature offsets we can access
       certain data types"""
    def __init__(
            self, df, feature_offsets, n_durations = 10, batch_size = 100): # 389 all training samples
        super().__init__()
        self.df = df
        self.feature_offsets = feature_offsets # cumulative sum of features in list of features [6000, 6000, 336, 148]
        self.n_views = len(feature_offsets) - 1
        self.n_durations = n_durations
        self.batch_size = batch_size

    def setup(
            self,
            test_size=0.2,
            cols_std=None,   #numeric feature names
            cols_leave=None, #binary feature names
            col_duration="duration",
            col_event="event",
            stage=None, #current stage of program (fit,test)
    ):

        """Split data into test and training set, preprocess this data with preprocess_features"""
        df_train, df_test = train_test_split(self.df, test_size=test_size)

        duration_train, duration_test = (
            df_train[col_duration].values,
            df_test[col_duration].values,
        )
        event_train, event_test = df_train[col_event].values, df_test[col_event].values

        cols_survival = [col_duration, col_event]
        cols_drop = cols_survival



        if cols_leave is None:
            cols_leave = []

        # features with numeric values
        if cols_std is None:
            cols_std = [
                col for col in self.df.columns if col not in cols_leave + cols_drop
            ]



        # Preprocess train and test data with programmed function
        self.x_train, self.x_test = preprocess_features(
            df_train=df_train.drop(cols_drop, axis = 1), # drop duration/event from df, as we don't want these
            df_test=df_test.drop(cols_drop, axis = 1),   # for training and testing sets
            cols_std=cols_std,
            cols_leave=cols_leave,
            feature_offset= self.feature_offsets
        )



        if stage == "fit" or stage is None:

            self.train_set = MultiOmicsDataset(
                self.x_train, duration_train, event_train)



    #    if stage == "test" or stage is None:
    #          self.test_set = MultiOmicsDataset(
    #              self.x_test, duration_test, event_test)



    def train_dataloader(self, batch_size):
        """
        Build training dataloader
        num_workers set to 0 by default because of some thread issue
        """
        train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)
        return train_loader




data_transform = pd.read_csv(
os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                 "1.tsv"), index_col=0, sep='\t'
)

names_mRNA = (data_transform["Entry Name"])
#names_mRNA.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/mRNA_transf.csv")

#names_mRNA_tensor = torch.tensor(names_mRNA.values)




#if __name__ == '__main__':

data_mRNA = pd.read_csv(
os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                 "TCGA_PRAD_1_mrna.csv"), index_col=0
)
data_DNA = pd.read_csv(
os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
             "TCGA_PRAD_2_dna.csv"), index_col=0
)
data_microRNA = pd.read_csv(
os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
             "TCGA_PRAD_3_microrna.csv"), index_col=0
)
data_RPPA = pd.read_csv(
os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
             "TCGA_PRAD_4_rppa.csv"), index_col=0
)
data_survival = pd.read_csv(
os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
             "TCGA_PRAD_meta.csv"), index_col=0
)


data_survival.rename(
{"vital_status": "event", "duration": "duration"}, axis="columns", inplace=True
)
data_survival.drop(labels= 'cancer_type', axis='columns', inplace=True)

data_event = data_survival.iloc[:, 0]
data_duration = data_survival.iloc[:, 1]


n_samples = data_survival.shape[0]
n_features = [len(data_mRNA.columns),
              len(data_DNA.columns),
              len(data_microRNA.columns),
              len(data_RPPA.columns),
              1,                    # event (no feature)
              1]                    # duration (no feature)
#print(n_features) # 6000, 6000, 336, 148

# cumulative sum of features in list
feature_offsets = [0] + np.cumsum(n_features).tolist()
#print("foff", feature_offsets) # 0,6000,12000,12336,12484,12485,12486


# TODO : feature selection in module

df_all = pd.concat([data_mRNA, data_DNA, data_microRNA, data_RPPA, data_survival], axis=1)


# drop unnecessary samples
df_all.drop(df_all[df_all["duration"] <= 0].index, axis = 0, inplace= True)


#Get names of features
features = []
for a in range(len(n_features)):
    features.append(list(df_all.columns.values[feature_offsets[a]:feature_offsets[a+1]]))


# Get features without numbers (needed for conversion) ; also remove | and - from strings

features_mRNA_no_numbers = []
for x in features[0]:
    index = x.find('|') # delete characters starting at | as the online tool won't work with these
    features_mRNA_no_numbers.append(x[:index])

features_DNA_no_numbers = []
for x in features[1]:
    index = x.find('|')
    features_DNA_no_numbers.append(x[:index])



# Index features (all features, no matter if mapping or not) so we can later on combine feature values of samples to proteins
feature_mRNA_indexed = {}
for index in range(len(features_mRNA_no_numbers)):
    feature_mRNA_indexed[index] = features_mRNA_no_numbers[index]



# features to csv file
df_mRNA_features_no_numbers = pd.DataFrame(features_mRNA_no_numbers)
df_DNA_features_no_number = pd.DataFrame(features_DNA_no_numbers)

df_mRNA_features_no_numbers.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/mRNA_feat_for_transf.csv",
                                   index=False)
df_DNA_features_no_number.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/DNA_feat_for_transf.csv",
                                 index=False)




# Load proteins and interactions

ppi_data = pd.read_csv(os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                                    "pp.txt"), sep=" ", header=0)





# drop scores below 700 (like in DeepMOCCA)
ppi_data.drop(ppi_data[(ppi_data["combined_score"]) < 700].index, axis = 0, inplace= True)

#reset indices
ppi_data = ppi_data.reset_index(drop=True)

#set so we can remove duplicates
proteins_data = list(set(ppi_data["protein1"].values))   # list of all proteins we look at

#check if protein 2 has proteins that are not in protein1 and add them if needed
for x in set(ppi_data["protein2"].values):
    if x not in proteins_data:
        proteins_data.append(x)




#create dictionary so we can use the index more easily

dict_proteins_data = {}
for index in range(len(proteins_data)):
    dict_proteins_data[index] = proteins_data[index]

 #switch key and values (to get same data structure as in DeepMOCCA) (protein name: index)
dict_proteins_data = {y: x for x, y in dict_proteins_data.items()}

# Edges in PPI network and according score

node1 = []
node2 = []
score = []



for index, row in ppi_data.iterrows():
    node1.append(dict_proteins_data[row["protein1"]])
    node2.append(dict_proteins_data[row["protein2"]])
    score.append(row["combined_score"])

temp = [node1,node2,score]

# tensor with rows : node 1, node 2, score
ppi_edges_score = torch.tensor(temp)






# Mapping HGNC to uniprot and uniprot to proteins : https://www.uniprot.org/id-mapping
# Mapping feature names to HGNC : https://www.syngoportal.org/convert



# feature to HGNC mapping
mRNA_HGNC_full = pd.read_excel(
    os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                 "mRNA_to_HGNC.xlsx"), index_col=0
)

mRNA_HGNC_full["features"] = mRNA_HGNC_full.index

mRNA_features_only = mRNA_HGNC_full["features"]

#Needed for conversion online
mRNA_HGNC_only = mRNA_HGNC_full["HGNC"]

mRNA_HGNC_only.to_csv(
    os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                 "HGNC_only"),index= False
)


mRNA_HGNC_dict = {}

# Dictionary ordering features to HGNC
for feature in mRNA_features_only:
    mRNA_HGNC_dict[feature] = mRNA_HGNC_full.loc[mRNA_HGNC_full["features"] == feature]["HGNC"].values

# turn arrays into lists in dict
for key in mRNA_HGNC_dict:
    mRNA_HGNC_dict[key] = mRNA_HGNC_dict[key].tolist()

counter = 0

# Remove features with no HGNC value
for key in list(mRNA_HGNC_dict.keys()):
    if type(mRNA_HGNC_dict[key][0]) is not str: # no HGNC value : nan as value to key, so not a string
        counter += 1
        del mRNA_HGNC_dict[key]


print("We have {} successful mappings from features to HGNC. {} could not be mapped.".format(len(mRNA_HGNC_dict), counter))









#HGNC to uniprot mapping

uniprot_HGNC_mRNA_full = pd.read_csv(
    os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                 "HGNC_mRNA_to_uniprot.tsv"), index_col=0, sep='\t'
)

uniprot_HGNC_mRNA_HGNC_only = uniprot_HGNC_mRNA_full.index
uniprot_HGNC_mRNA_HGNC_only = list(uniprot_HGNC_mRNA_HGNC_only)

uniprot_HGNC_mRNA_dict = {}

for HGNC in uniprot_HGNC_mRNA_HGNC_only:
    uniprot_HGNC_mRNA_dict[HGNC] = uniprot_HGNC_mRNA_full.loc[uniprot_HGNC_mRNA_full.index == HGNC]["Entry"].values


# turn arrays into lists in dict
for key in uniprot_HGNC_mRNA_dict:
    uniprot_HGNC_mRNA_dict[key] = uniprot_HGNC_mRNA_dict[key].tolist()

# Needed for further conversion
uniprot_HGNC_mRNA_full_entries = uniprot_HGNC_mRNA_full["Entry"] # Entry columns has uniprot values

uniprot_HGNC_mRNA_full_entries.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/uniprot_mRNA_transf.csv", index=False)

#counter = 0
# Remove HGNC-values which couldn't be mapped to uniprot
#for key in list(uniprot_HGNC_mRNA_dict.keys()):
#    if len(uniprot_HGNC_mRNA_dict[key]) == 0:
#        del uniprot_HGNC_mRNA_dict[key]
#        counter += 1

print("We have {} successful mappings from HGNC to uniprots.".format(len(uniprot_HGNC_mRNA_dict)))













#Finally, uniprot values to proteins
uniprot_to_proteins_mRNA_full = pd.read_csv(
    os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                 "uniprot_to_protein_mRNA.tsv"), index_col=0, sep="\t"
)

uniprot_to_proteins_mRNA_dict = {}


uniprot_to_proteins_mRNA_uniprot_only = uniprot_to_proteins_mRNA_full.index
uniprot_to_proteins_mRNA_uniprot_only = list(uniprot_to_proteins_mRNA_uniprot_only)

proteins = list(uniprot_to_proteins_mRNA_full["To"].values)
#save proteins and give them indices
dict_index_proteins = {}

for x in range(len(proteins)):
    dict_index_proteins[x] = proteins[x]



for uniprot in uniprot_to_proteins_mRNA_uniprot_only:
    uniprot_to_proteins_mRNA_dict[uniprot] = uniprot_to_proteins_mRNA_full.loc[uniprot_to_proteins_mRNA_full.index == uniprot]["To"].values

# turn arrays into lists in dict
for key in uniprot_to_proteins_mRNA_dict:
    uniprot_to_proteins_mRNA_dict[key] = uniprot_to_proteins_mRNA_dict[key].tolist()

#counter = 0
# Remove HGNC-values which couldn't be mapped to uniprot        #TODO : diese funktionen könnten key errors bei dicts zusammenfügen später erzeugen!
#for key in list(uniprot_to_proteins_mRNA_dict.keys()):
#    if len(uniprot_to_proteins_mRNA_dict[key]) == 0:
#        del uniprot_to_proteins_mRNA_dict[key]
#        counter += 1


print("We have {} successful mappings from uniprots to proteins.".format(len(uniprot_to_proteins_mRNA_dict)))






# Map our features from the beginning to the proteins (the ones that could be mapped : only these will be a part of
# the GCN

#first from dict(HGNC: uniprot) and dict(uniprot : proteins) to dict(HGNC : proteins)

dict_HGNC_proteins = {}

# Intialize empty dictionary for each HGNC value as key

for HGNC in uniprot_HGNC_mRNA_HGNC_only:
    dict_HGNC_proteins[HGNC] = []

# fill dictionary

for key in uniprot_to_proteins_mRNA_dict:
    for key2 in uniprot_HGNC_mRNA_dict:
        if key in uniprot_HGNC_mRNA_dict[key2]:
            dict_HGNC_proteins[key2].append(uniprot_to_proteins_mRNA_dict[key])


print("We have {} mappings from HGNC to proteins".format(len(dict_HGNC_proteins)))


# now from dict(HGNC:proteins) and dict(feature:HGNC) to dict(feature:proteins) (by applying the same logic)

dict_features_proteins_mRNA = {}

for features in mRNA_features_only:
    dict_features_proteins_mRNA[features] = []


def flatten(l):
    return [item for sublist in l for item in sublist]



for key in dict_HGNC_proteins:
    for key2 in mRNA_HGNC_dict:
        if key in mRNA_HGNC_dict[key2]:

            dict_features_proteins_mRNA[key2] = (flatten(dict_HGNC_proteins[key]))






counter = 0
# Remove features with no protein connection
for key in list(dict_features_proteins_mRNA.keys()):
    if len(dict_features_proteins_mRNA[key]) == 0:
        counter += 1
        del dict_features_proteins_mRNA[key]




# Note that some features are connected to multiple proteins !
print("We have {} mappings from features to proteins. {} could not be mapped".format(len(dict_features_proteins_mRNA),counter))

#TODO : load in DNA data, assign indexes to proteins and tensor for interaction based on STRING PPI csv file














tensor_data_order_by_sample = torch.tensor(df_all.values)
tensor_data_order_by_view = []

for x in range(len(n_features)):
    tensor_data_order_by_view.append(torch.tensor((df_all.iloc[:, feature_offsets[x]: feature_offsets[x + 1]]).values))





multimodule = SurvMultiOmicsDataModule(df_all, feature_offsets)
multimodule.setup()



# set feature values to proteins, sample wise
train_loader = multimodule.train_dataloader(batch_size = 20)

#train_data = []
#mask_data = []
sample_to_protein = []
# For each sample, make a dictionary with protein : feature value ; save list of dictionaries
features_proteins_mRNA_values_all = []
#print("train ", data)
#print("train 0", data[0])
#print("sample 0", data[0][0])
#print("sample0 first value", data[0][0][0])
#mRNA


for data,mask, duration, event in train_loader:


    # Go through each mRNA sample
    for sample in data[0]:
        # for each sample dictionary
        dict_features_proteins_mRNA_values = {}
        # Through each feature (for mRNA)
        for feature_idx in range(len(sample)): # len = 6000 features
            temp = []
            # If the feature has a mapping in feature to protein
            # features_mRNA_no_numbers[feature_idx] : feature name at index
            if features_mRNA_no_numbers[feature_idx] in dict_features_proteins_mRNA:


                # take the proteins at the current index which correspond to the feature we are looking at
                protein_list = dict_features_proteins_mRNA[features_mRNA_no_numbers[feature_idx]]

                # create dictionary entries for it
                for protein in protein_list:
                    #if key already exists, just append values to protein
                    if protein in dict_features_proteins_mRNA_values:
                        dict_features_proteins_mRNA_values[protein].append(sample[feature_idx])
                    #else create a key entry
                    else:

                        dict_features_proteins_mRNA_values[protein] = [sample[feature_idx]]


        # for each sample, append to list
        features_proteins_mRNA_values_all.append(dict_features_proteins_mRNA_values)


#features_proteins_mRNA_values_all : list containing all samples, where for each sample we have a
# dictionary with the key being the protein and the values being the values of the features that correspond
# to the protein




# Create graph representation

# feature matrix X |A| x |B| A : number of nodes (proteins), B : number of features per node (value features diff. views)
# adjacency matrix Z : |A| x |A|

#adjacency matrix for protein-protein-network
adjacency_matrix_ppi = torch.zeros(len(proteins_data), len(proteins_data))


for protein_idx in range(len(proteins_data)):
    # for each edge between two nodes, fill in a 1 in adjacency matrix
    adjacency_matrix_ppi[int((ppi_edges_score[0, protein_idx]).item()), int((ppi_edges_score[1, protein_idx]).item())] = torch.tensor(1)




# feature matrix for each sample
#torch tensor with A x B : rows x columns
# Initialize
feature_matrices_mRNA = []
counter = 0
#for each sample (we could also use smth diff then len(features...) to access all samples
for sample_idx in range(len(features_proteins_mRNA_values_all)):

    # As column size for the tensor, we use the protein (node) which has the most features (values from diff views)
    # We use the largest as the torch tensor is basically a matrix and we need a fixed size
    # to test, just implemented for mRNA
    most_feat_for_protein = len(max(features_proteins_mRNA_values_all[sample_idx].values(), key=len))

    # As rows, we just use all the proteins ; matrix for one sample
    feature_matrix_mRNA = torch.zeros(len(proteins_data), most_feat_for_protein)

    # fill with data

    # go through each protein
    for protein in proteins_data:
        # check if protein in protein :feature values dict
        if protein in features_proteins_mRNA_values_all[sample_idx]:

            # fill row with according data ; find right row by accesing dict protein data (find the right index)

            # Find index by accessing dict which saves pairs of protein : index
            index = dict_proteins_data[protein]
            feature_matrix_mRNA[index, 0:len(features_proteins_mRNA_values_all[sample_idx][protein])] = \
                torch.tensor(features_proteins_mRNA_values_all[sample_idx][protein])



        #else : we leave it as it is (filled with 0s)

    feature_matrices_mRNA.append(feature_matrix_mRNA)
#   Todo : normalization before adding into matrices (see DeepMOCCA)



#print(feature_matrix_mRNA[0])
#print(feature_matrix_mRNA.shape) # 16814,3
print(feature_matrices_mRNA[2])








# each protein node has features from different views (features of that node, in deepmocca sind das die 1-8 ?
# bei mir dann mRNA und DNA ?
# input gcn ist dann das PPI : jeder Node hat best. features und wir kennen die edges











 #       print(data)
 #       print(mask)
 #      print(duration)
 #       print(event)
 #       break





#%%
