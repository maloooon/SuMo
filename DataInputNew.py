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


# Get features without numbers (needed for conversion)
features_mRNA_no_numbers = []
for x in features[0]:
    features_mRNA_no_numbers.append(''.join([i for i in x if not i.isdigit()]))



# features to csv file
df_mRNA_features_no_numbers = pd.DataFrame(features_mRNA_no_numbers)

df_mRNA_features_no_numbers.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/mRNA_feat_for_transf.csv", index=False)

#print(features[1])

# feature to HGNC mapping
mRNA_HGNC_full = pd.read_excel(
    os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                 "mRNA_to_HGNC.xlsx"), index_col=0
)

mRNA_HGNC_full["features"] = mRNA_HGNC_full.index

mRNA_features_only = mRNA_HGNC_full["features"]

mRNA_features_only = list(mRNA_features_only)
mRNA_HGNC_dict = {}

# Dictionary ordering HGNC to features
for feature in mRNA_features_only:
    mRNA_HGNC_dict[feature] = mRNA_HGNC_full.loc[mRNA_HGNC_full["features"] == feature]["HGNC"].values






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

#print(uniprot_HGNC_mRNA_dict)

# Needed for further conversion
uniprot_HGNC_mRNA_full_entries = uniprot_HGNC_mRNA_full["Entry"]

uniprot_HGNC_mRNA_full_entries.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/uniprot_mRNA_transf.csv", index=False)


#Finally, uniprot values to proteins
uniprot_to_proteins_mRNA_full = pd.read_csv(
    os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                 "uniprot_to_protein_mRNA.tsv"), index_col=0, sep="\t"
)

uniprot_to_proteins_mRNA_dict = {}


uniprot_to_proteins_mRNA_uniprot_only = uniprot_to_proteins_mRNA_full.index
uniprot_to_proteins_mRNA_uniprot_only = list(uniprot_to_proteins_mRNA_uniprot_only)

for uniprot in uniprot_to_proteins_mRNA_uniprot_only:
    uniprot_to_proteins_mRNA_dict[uniprot] = uniprot_to_proteins_mRNA_full.loc[uniprot_to_proteins_mRNA_full.index == uniprot]["To"].values


# Map our features from the beginning to the proteins (the ones that could be mapped : only these will be a part of
# the GCN

#first from dict(HGNC: uniprot) and dict(uniprot : proteins) to dict(HGNC : proteins)

dict_HGNC_proteins = {}

# Intialize empty dictionary for each HGNC value as key
print(len(uniprot_HGNC_mRNA_HGNC_only))
for HGNC in uniprot_HGNC_mRNA_HGNC_only:
    dict_HGNC_proteins[HGNC] = []

# fill dictionary

for key in uniprot_to_proteins_mRNA_dict:
    for key2 in uniprot_HGNC_mRNA_dict:
        if key in uniprot_HGNC_mRNA_dict[key2]:
            dict_HGNC_proteins[key2].append(uniprot_to_proteins_mRNA_dict[key])

# Wv proteine in HGNC : protein ?
counter = 0
for key in dict_HGNC_proteins:
    if len(dict_HGNC_proteins[key]) != 0:
        counter += 1

print("counter: {}".format(counter)) #correct

print(dict_HGNC_proteins)


# now from dict(HGNC:proteins) and dict(feature:HGNC) to dict(feature:proteins) (by applying the same logic)

dict_features_proteins_mRNA = {}

for features in mRNA_features_only:
    dict_features_proteins_mRNA[features] = []



for key in dict_HGNC_proteins:
    for key2 in mRNA_HGNC_dict:
        if key in mRNA_HGNC_dict[key2]:

            dict_features_proteins_mRNA[key2].append(dict_HGNC_proteins[key])


#print(len(dict_features_proteins_mRNA)) 3020

# How many of our 6000 mRNA features actually have proteins we can look at ?
proteins = 0
more_conn = 0

for key in dict_features_proteins_mRNA:
    if len(dict_features_proteins_mRNA[key]) != 0:
        proteins += 1
    if len(dict_features_proteins_mRNA[key]) > 1:
        more_conn +=1

print("We have {} many feature-protein connections".format(proteins))
print("We have {} many feature-protein connections where the feature has multiple proteins".format(more_conn))          # TODO : kann das sein ?










tensor_data_order_by_sample = torch.tensor(df_all.values)
tensor_data_order_by_view = []

for x in range(len(n_features)):
    tensor_data_order_by_view.append(torch.tensor((df_all.iloc[:, feature_offsets[x]: feature_offsets[x + 1]]).values))





multimodule = SurvMultiOmicsDataModule(df_all, feature_offsets)
multimodule.setup()












 #       print(data)
 #       print(mask)
 #      print(duration)
 #       print(event)
 #       break








#%%

#%%
