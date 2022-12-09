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
import CancerDataLoading
from PPINetworkDataLoading import flatten


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
            x_train_ordered_by_view.append(torch.tensor((x_train_df.iloc[:, feat_offsets_PRAD[x]:
                                                                            feat_offsets_PRAD[x + 1]]).values))
            x_test_ordered_by_view.append(torch.tensor((x_train_df.iloc[:, feat_offsets_PRAD[x]:
                                                                           feat_offsets_PRAD[x + 1]]).values))




    else:
        # Order by view so it works with Dataset class
        x_train_ordered_by_view = []
        x_test_ordered_by_view = []
        for x in range(len(feature_offset) - 3): # -3 bc we don't have duration/event in training tensor
            x_train_ordered_by_view.append(torch.tensor((df_train.iloc[:, feat_offsets_PRAD[x]:
                                                                          feat_offsets_PRAD[x + 1]]).values))
            x_test_ordered_by_view.append(torch.tensor((df_train.iloc[:, feat_offsets_PRAD[x]:
                                                                         feat_offsets_PRAD[x + 1]]).values))



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




# Read PRAD data
data_PRAD = pd.read_csv(
    os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                 "PRADData.csv"), index_col=0)

# Read feature offsets of PRAD data
feat_offsets_PRAD = pd.read_csv(
    os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                 "PRADDataFeatOffsets.csv"), index_col=0)

# convert to list and flatten list (since it has each element in a list itself)
feat_offsets_PRAD = flatten(feat_offsets_PRAD.values.tolist())


#Get names of features
features = []
for a in range(len(feat_offsets_PRAD) - 1):
    features.append(list(data_PRAD.columns.values[feat_offsets_PRAD[a]:
                                                  feat_offsets_PRAD[a+1]]))


multimodule = SurvMultiOmicsDataModule(data_PRAD, feat_offsets_PRAD)
multimodule.setup()

#tensor_data_order_by_sample = torch.tensor(data_PRAD.values)
#tensor_data_order_by_view = []

#for x in range(len(feat_offsets_PRAD) - 1):
#    tensor_data_order_by_view.append(torch.tensor((data_PRAD.iloc[:, feat_offsets_PRAD[x]:
#                                                                     feat_offsets_PRAD[x + 1]]).values))

# Setting up data (preprocessing, standardizing, preparing so we can use it with dataloader)

