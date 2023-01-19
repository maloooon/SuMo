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
import FeatureSelection
from torch.optim import Adam
from torch import nn
from pycox import models
import ReadInData



class MultiOmicsDataset(Dataset):

    # [tensor von allen samples view 1,  tensor von allen samples view 2, ... ]
    # in dem tensor jeweils features sample1, features sample 2, ...

    def __init__(self, X, duration, event, type= 'new'):
        self.type = type

        # Change dtypes of tensors for usage of NN later on
        self.duration = duration
        self.event = event
        self.duration = torch.from_numpy(self.duration).to(torch.float32)
        self.event = torch.from_numpy(self.event).to(torch.int32)



        if self.type == 'new':
            self.n_views = len(X) # number views (mRNA, DNA, microRNA, RPPA)
            self.X = [torch.nan_to_num(x_view) for x_view in X]
            self.mask = [torch.isnan(x_view) for x_view in X] #List of booleans for each numeric value in samples ;True : NaN values
            self.n_samples = X[0].size(0) # number samples # TODO: versch- sample Anzahlen bei eigengenes
            # Check if for each view (each tensor containing all samples for one view) the amount of samples
            # is the same as the sample size, otherwise print mismatch

            assert all(
                view.size(0) == self.n_samples for view in X
        ), "Size mismatch between tensors"


        # TODO : center data,  , mit dne null values arbeiten // mean mutation
        elif self.type == 'processed':
            self.n_views = len(X) # number views (mRNA, DNA, microRNA, RPPA)
            self.X = X
            self.n_samples = X[0].size(0)
            for view in range(len(self.X)):
                self.X[view] = self.X[view].to(torch.float32)


        elif self.type == 'ppi':
            self.X = X
            self.n_samples = X.size(0)



    def __len__(self):
        """Return the amount of samples"""
        return self.n_samples


    def __getitem__(self, index):
        """return the whole sample (all views)"""
        if self.type == 'new':
            return [self.X[m][index, :] for m in range(self.n_views)], \
                   [self.mask[m][index, :] for m in range(self.n_views)], \
                   self.duration[index], self.event[index]
        elif self.type == 'processed':
            return [self.X[m][index, :] for m in range(self.n_views)], \
                   self.duration[index], self.event[index]
        elif self.type == 'ppi':
            return self.X[index], self.duration[index], self.event[index]










def preprocess_features(
        df_train: pd.DataFrame, # train dataset
        df_test: pd.DataFrame, # test dataset
        df_all: pd.DataFrame, # all data
      #  df_val: pd.DataFrame, # validation dataset
        cols_std: List[str], # feature names, numeric variables
        cols_leave: List[str], # feature names, binary variables
        feature_offset : List[int],
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
        x_all = mapper.fit_transform(df_all).astype(np.float32) # TODO : we standardize over all data ! --> test data leakage into train data
     #   x_val = mapper.transform(df_val).astype(np.float32)
        x_train_df = pd.DataFrame(x_train)
        x_test_df = pd.DataFrame(x_test)
        x_all_df = pd.DataFrame(x_all)
     #   x_val_df = pd.DataFrame(x_val)

        # Order by view so it works with Dataset class
        x_train_ordered_by_view = []
        x_test_ordered_by_view = []
        x_all_ordered_by_view = []
     #   x_val_ordered_by_view = []

        for x in range(len(feature_offset) - 3): # -3 bc we don't have duration/event in training tensor
            x_train_ordered_by_view.append(torch.tensor((x_train_df.iloc[:, feature_offset[x]:
                                                                            feature_offset[x + 1]]).values))
            x_test_ordered_by_view.append(torch.tensor((x_test_df.iloc[:, feature_offset[x]:
                                                                           feature_offset[x + 1]]).values))

            x_all_ordered_by_view.append(torch.tensor((x_all_df.iloc[:, feature_offset[x]:
                                                                          feature_offset[x + 1]]).values))
       #     x_val_ordered_by_view.append(torch.tensor((x_val_df.iloc[:, feature_offset[x]:
       #                                                                  feature_offset[x + 1]]).values))




    else:
        # Order by view so it works with Dataset class
        x_train_ordered_by_view = []
        x_test_ordered_by_view = []
        x_all_ordered_by_view = []
   #     x_val_ordered_by_view = []
        for x in range(len(feature_offset) - 3): # -3 bc we don't have duration/event in training tensor
            x_train_ordered_by_view.append(torch.tensor((df_train.iloc[:, feature_offset[x]:
                                                                          feature_offset[x + 1]]).values))
            x_test_ordered_by_view.append(torch.tensor((df_test.iloc[:, feature_offset[x]:
                                                                         feature_offset[x + 1]]).values))
            x_all_ordered_by_view.append(torch.tensor((df_all.iloc[:, feature_offset[x]:
                                                                        feature_offset[x + 1]]).values))
     #       x_val_ordered_by_view.append(torch.tensor((df_val.iloc[:, feature_offset[x]:
     #                                                                   feature_offset[x + 1]]).values))




    return x_train_ordered_by_view, x_test_ordered_by_view, x_all_ordered_by_view #, x_val_ordered_by_view


class SurvMultiOmicsDataModule(pl.LightningDataModule):
    """Input is the whole dataframe : We merge all data types together, with the feature offsets we can access
       certain data types ; dataframe also contains duration and event !"""
    def __init__(
            self, df, feature_offsets, view_names, n_durations = 10, onezeronorm_bool = False, cancer_name=None):
        super().__init__()
        self.df = df
        self.feature_offsets = feature_offsets # cumulative sum of features in list of features
        self.n_durations = n_durations
        self.view_names = view_names
        self.n_views = len(view_names)
        self.onezeronorm_bool = onezeronorm_bool
        self.cancer_name = cancer_name

    def setup(
            self,
            test_size=0.2,
            cols_std=None,   #numeric feature names
           # cols_leave=None, #binary feature names
            col_duration="duration",
            col_event="event",
            stage=None, #current stage of program (fit,test)
    ):


        if self.onezeronorm_bool == True:
            # 01-Normalization if wanted
            event_and_duration = self.df.iloc[:,-2:].values # get event & duration values, as they don't need to be normalized to 01
            x = self.df.iloc[:,0:-2].values #returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            x_scaled = np.append(x_scaled, event_and_duration, axis=1)
            self.df = pd.DataFrame(x_scaled, columns=self.df.columns)

        """Split data into test and training set and training into training and validation set
           , preprocess this data with preprocess_features."""
        df_train_temp, df_test = train_test_split(self.df, test_size=test_size) # all --> train  , test

        # Also use a non splitted version to see how the performance increases/decreases (so we can do feature selection
        # on the dataset as a whole)

        n_samples = self.df.shape[0]
        self.duration = self.df[col_duration].values
        self.event = self.df[col_event].values


   #     n_train_samples = df_train.shape[0]
        n_train_samples = df_train_temp.shape[0]
        n_test_samples = df_test.shape[0]
     #   n_val_samples = df_val.shape[0]
        self.duration_train, self.duration_test =(df_train_temp[col_duration].values,                                   # ,self.duration_val
                                                df_test[col_duration].values
                                                )                                                                           # df_val[col_duration].values
        self.event_train, self.event_test = df_train_temp[col_event].values, df_test[col_event].values                                              # , self.event_val
                                                                                                                        # df_val[col_event].values


        cols_leave = [col for col in self.df
                     if np.isin(self.df[col].dropna().unique(), [0, 1]).all()]
        # remove event from binary column values (we are only interested in feature values from views)
        if 'event' in cols_leave:
            cols_leave.remove('event')



        cols_survival = [col_duration, col_event] # columns we don't want to standardize
        cols_drop = cols_survival



        if cols_leave is None:
            cols_leave = []

        # features with numeric values
        if cols_std is None:
            cols_std = [
                col for col in self.df.columns if col not in cols_leave + cols_drop
            ]



        # Preprocess train and test data with programmed function #  ,self.x_val, df_val=df_val.drop(cols_drop, axis = 1)
        self.x_train, self.x_test, self.x_all = preprocess_features(
            df_train=df_train_temp.drop(cols_drop, axis = 1), # drop duration/event from df, as we don't want these
            df_test=df_test.drop(cols_drop, axis = 1),
            df_all= self.df.drop(cols_drop, axis = 1),# for training and testing sets
            cols_std=cols_std,
            cols_leave=cols_leave,
            feature_offset= self.feature_offsets
        )

        self.x_train = [torch.nan_to_num(x_view) for x_view in self.x_train]
        self.x_train_mask = [torch.isnan(x_view) for x_view in self.x_train] #List of booleans for each numeric value in samples ;True : NaN values


        self.x_test = [torch.nan_to_num(x_view) for x_view in self.x_test]
        self.x_test_mask = [torch.isnan(x_view) for x_view in self.x_test]

        self.x_all = [torch.nan_to_num(x_view) for x_view in self.x_all]
        self.x_all_mask = [torch.isnan(x_view) for x_view in self.x_all]

    #    self.x_val = [torch.nan_to_num(x_view) for x_view in self.x_val]
    #    self.x_val_mask = [torch.isnan(x_view) for x_view in self.x_val]


        # We check for each view how many 0s this view contains
        train_zeros = []
        test_zeros = []
      #  all_zeros = []
    #    val_zeros = []
        for c,view in enumerate(self.x_train):
            curr_view_count_train = torch.count_nonzero(view)
            train_zeros.append(curr_view_count_train)
     #       curr_view_count_val = torch.count_nonzero(self.x_val[c])
     #       val_zeros.append(curr_view_count_val)
            curr_view_count_test = torch.count_nonzero(self.x_test[c])
            test_zeros.append(curr_view_count_test)
       #     curr_view_count_all = torch.count_nonzero(self.x_all[c])
       #     all_zeros.append(curr_view_count_all)

        removed_views_index = []
        for x,count in enumerate(train_zeros):
            # If there arent atleast 10 % values greater than 0 for this view for all samples, remove this view
            # from consideration
            if train_zeros[x] < int(0.1 * ((self.feature_offsets[x + 1] - self.feature_offsets[x]) * n_train_samples))\
                    or \
                test_zeros[x] < int(0.1 * ((self.feature_offsets[x + 1] - self.feature_offsets[x]) * n_test_samples)):
        #            or \
        #        val_zeros[x] < int(0.1 * ((self.feature_offsets[x + 1] - self.feature_offsets[x]) * n_val_samples)):
                print("{} has nearly only 0 values. We don't take this data into consideration.".format(self.view_names[x]))
                removed_views_index.append(x)

        for index in sorted(removed_views_index, reverse=True):
            del self.x_train[index]
            del self.x_test[index]
    #        del self.x_val[index]
            del self.x_all[index]
            del self.view_names[index]
            diff = self.feature_offsets[index + 1] - self.feature_offsets[index]
            for c,_ in enumerate(self.feature_offsets):
                if c > (index + 1):
                    self.feature_offsets[c] = self.feature_offsets[c] - diff
            del self.feature_offsets[index+1]

        self.n_views = self.n_views - len(removed_views_index)


        return n_train_samples, n_test_samples, n_samples, self.view_names   #  n_val_samples,


    def label_transform(self, num_durations):
        """label transform as needed for pycox :
        https://github.com/havakv/pycox/blob/master/pycox/preprocessing/label_transforms.py
        num_durations : size of equidistant discretization grid (if used, network will have num_durations output nodes)
        """
        labtrans_LH = models.LogisticHazard.label_transform(num_durations)
       # labtrans_PMF = models.PMF.label_transform(num_durations)
       # labtrans_DeepHit = models.DeepHit.label_transform(num_durations)

        y_train = labtrans_LH.fit_transform(self.duration_train, self.event_train)


        return labtrans_LH



    def feature_selection(self, method = None,
                          feature_names = None, # for PPI network
                          components = None, # for PCA
                          thresholds = None,
                          select_setting = 'all'): # decide to feature select on all data or on train/test split

        """Feature selection is done on the whole dataset for the purpose of not throwing any data away due to feature
           size mismatches between the test and train set."""

        if method.lower() == 'eigengenes':


            for view in range(self.n_views):

                eg_view = FeatureSelection.F_eigengene_matrices(train=self.x_train[view],
                                                                mask=self.x_train_mask[view],
                                                                view_name=self.view_names[view],
                                                                duration=self.duration_train,
                                                                event=self.event_train,
                                                                stage= 'train',
                                                                cancer_name= self.cancer_name)

                eg_view_test = FeatureSelection.F_eigengene_matrices(train=self.x_test[view],
                                                                     mask=self.x_test_mask[view],
                                                                     view_name=self.view_names[view],
                                                                     duration=self.duration_test,
                                                                     event=self.event_test,
                                                                     stage='test',
                                                                     cancer_name= self.cancer_name)

                eg_view.preprocess()
                eg_view_test.preprocess()



            eg_view.eigengene_multiplication()
            eigengene_matrices,eigengene_matrices_test = eg_view.get_eigengene_matrices(self.view_names)

            # as list as each eigengene matrix is of a different size
            eigengene_matrices_tensors = []
            eigengene_matrices_tensors_test = []
            for x in range(self.n_views):
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



            # only for train sets
            for c,view in enumerate(self.view_names):
                print("{} eigengene matrix : {} of size {}. Originally, we had {} features, now we have {}.".format
                      (view,eigengene_matrices_tensors[c], eigengene_matrices_tensors[c].shape,
                       self.feature_offsets[c+1] - self.feature_offsets[c],
                       eigengene_matrices_tensors[c].size(1)))


            # Train and test feature sizes need to be the same for each view separately for neural network
            for c in range(self.n_views):
                if eigengene_matrices_tensors[c].size(1) != eigengene_matrices_tensors_test[c].size(1):
                    minimum = min(eigengene_matrices_tensors[c].size(1), eigengene_matrices_tensors_test[c].size(1))

                    # change size of test set
                    if minimum == eigengene_matrices_tensors[c].size(1):
                        eigengene_matrices_tensors_test[c] = eigengene_matrices_tensors_test[c][:,0:minimum]
                    # change size of train set
                    else:
                        eigengene_matrices_tensors[c] = eigengene_matrices_tensors[c][:,0:minimum]



            self.train_set = MultiOmicsDataset(eigengene_matrices_tensors,
                                               self.duration_train,
                                               self.event_train,
                                               type = 'processed')

            self.test_set = MultiOmicsDataset(eigengene_matrices_tensors_test,
                                              self.duration_test,
                                              self.event_test,
                                              type = 'processed')






        if method.lower() == 'pca':
            PCA_train_tensors = []
            PCA_test_tensors = []


            for view in range(self.n_views):
                # Initialize PCA objects for both train and test with same components
                view_train_PCA = FeatureSelection.F_PCA(self.x_train[view], components=components[view])
                view_test_PCA = FeatureSelection.F_PCA(self.x_test[view], components=components[view])
                # Apply PCA just to the train set
                obj_PCA = view_train_PCA.apply_pca()
                # Fit & Transform the train set
                train_data = view_train_PCA.fit_transform_pca(obj_PCA)
                # Only transform the test set with the given PCA object of train
                test_data = view_test_PCA.transform_pca(obj_PCA)

                PCA_train_tensors.append(torch.tensor(train_data))
                PCA_test_tensors.append(torch.tensor(test_data))




            """
            for view in range(self.n_views):
                view_train_PCA = FeatureSelection.F_PCA(self.x_train[view], components=components[view])
                view_test_PCA = FeatureSelection.F_PCA(self.x_test[view], components=components[view])
             #   view_val_PCA = FeatureSelection.F_PCA(self.x_val[view], components=components)
                pc_train_view = view_train_PCA.apply_pca()
                pc_test_view = view_test_PCA.apply_pca()
             #   pc_val_view = view_val_PCA.apply_pca()
                PCA_train_tensors.append(torch.tensor(pc_train_view))
                PCA_test_tensors.append(torch.tensor(pc_test_view))
           #     PCA_validation_tensors.append(torch.tensor(pc_val_view))



            # Train and test feature sizes need to be the same for each view separately for neural network

            # For PCA, we have components as an input for both train/test sets simultaneously, thus this
            # isn't actually needed.
            for c in range(self.n_views):
                if PCA_train_tensors[c].size(1) != PCA_test_tensors[c].size(1):
                    minimum = min(PCA_train_tensors[c].size(1), PCA_test_tensors[c].size(1))

                    # change size of test set
                    if minimum == PCA_train_tensors[c].size(1):
                        PCA_test_tensors[c] = PCA_test_tensors[c][:,0:minimum]
                    # change size of train set
                    else:
                        PCA_train_tensors[c] = PCA_train_tensors[c][:,0:minimum]
            """




            self.train_set = MultiOmicsDataset(PCA_train_tensors,
                                                   self.duration_train,
                                                   self.event_train,
                                                   type = 'processed')

            self.test_set = MultiOmicsDataset(PCA_test_tensors,
                                              self.duration_test,
                                              self.event_test,
                                              type = 'processed')






        #    self.val_set = MultiOmicsDataset(PCA_validation_tensors,
        #                                     self.duration_val,
        #                                     self.event_val,
        #                                      type = 'processed')




        if method.lower() == 'variance':
            variance_train_tensors = []
            variance_test_tensors = []
            variance_all_tensors = []


         #   thresholds = [0.8,0.8,0.8,0] # TODO : Grid Search


            if select_setting == 'split':

                for view in range(self.n_views):

                    view_train_variance = FeatureSelection.F_VARIANCE(self.x_train[view], threshold= thresholds[view])
                    view_test_variance = FeatureSelection.F_VARIANCE(self.x_test[view], threshold = thresholds[view])
                    data_train_variance, mask_train_variance = view_train_variance.apply_variance()
                    data_test_variance, mask_test_variance = view_test_variance.apply_variance()
                    variance_train_tensors.append(torch.tensor(data_train_variance))
                    variance_test_tensors.append(torch.tensor(data_test_variance))

                for c in range(self.n_views):
                    if variance_train_tensors[c].size(1) != variance_test_tensors[c].size(1):
                        minimum = min(variance_train_tensors[c].size(1), variance_test_tensors[c].size(1))

                        # change size of test set
                        if minimum == variance_train_tensors[c].size(1):
                            variance_test_tensors[c] = variance_test_tensors[c][:,0:minimum]
                        # change size of train set
                        else:
                            variance_train_tensors[c] = variance_train_tensors[c][:,0:minimum]

                self.train_set = MultiOmicsDataset(variance_train_tensors,
                                                   self.duration_train,
                                                    self.event_train,
                                                   type = 'processed')

                self.test_set = MultiOmicsDataset(variance_test_tensors,
                                                  self.duration_test,
                                                  self.event_test,
                                                  type = 'processed')
            if select_setting ==  'all':
                for view in range(self.n_views):
                    view_variance = FeatureSelection.F_VARIANCE(self.x_all[view], threshold= thresholds[view])
                    data_variance, mask_variance = view_variance.apply_variance()
                    variance_all_tensors.append(torch.tensor(data_variance))


            self.all_set = MultiOmicsDataset(variance_all_tensors,
                                             self.duration,
                                             self.event,
                                             type = 'processed')


        if method.lower() == 'ae':


            AE_all_compressed_train_features = []
            AE_all_compressed_validation_features = []
            AE_all_compressed_test_features = []


            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for view in range(self.n_views):
                model = FeatureSelection.F_AE(train= self.x_train[view]).to(device)

                optimizer = Adam(model.parameters(), lr=1e-3)

                criterion = nn.MSELoss() # reconstrution loss

                self.train_set = MultiOmicsDataset(
                           self.x_train, self.duration_train, self.event_train, type = 'new')

        #        self.val_set = MultiOmicsDataset(self.x_val, self.duration_val, self.event_val, type = 'new')

                self.test_set = MultiOmicsDataset(self.x_test, self.duration_test, self.event_test, type = 'new')

                ae_trainloader = DataLoader(self.train_set,batch_size=self.x_train[0].size(0),shuffle=True,drop_last=True)

         #       ae_validationloader = DataLoader(self.val_set, batch_size= self.x_val[0].size(0), shuffle=True, drop_last=True)

                ae_testloader = DataLoader(self.test_set, batch_size=self.x_test[0].size(0), shuffle=True, drop_last=True)
                epochs = 30
                temp = []
                temp2 = []
                temp3 = []
                for epoch in range(epochs):
                    loss = 0
                    for batch_data, train_mask, train_duration, train_event in ae_trainloader:

                        batch_data = batch_data[view].view(-1, batch_data[view].size(1)).to(device)

                        optimizer.zero_grad()

                        # compressed features is what we are interested in
                        reconstructed, compressed_train_features = model(batch_data)
                        if epoch == epochs - 1: # save compressed_features of last epoch for each batch
                            temp.append(compressed_train_features) # list of tensors of compressed for each batch

                        train_loss = criterion(reconstructed, batch_data)

                        train_loss.backward()

                        optimizer.step()

                        loss += train_loss.item()



                    loss = loss / len(ae_trainloader)

                    print("epoch : {}/{}, loss = {:.6f} for {} data".format(epoch + 1, epochs, loss, self.view_names[view]))




                    for batch_data, test_mask, test_duration, test_event in ae_testloader:
                        batch_data = batch_data[view].view(-1, batch_data[view].size(1)).to(device)

                        optimizer.zero_grad()

                        # compressed features is what we are interested in
                        reconstructed, compressed_test_features = model(batch_data)
                        if epoch == epochs - 1: # save compressed_features of last epoch for each batch
                            temp2.append(compressed_test_features) # list of tensors of compressed for each batch

                        test_loss = criterion(reconstructed, batch_data)

                        test_loss.backward()

                        optimizer.step()

                        loss += test_loss.item()



                    loss = loss / len(ae_testloader)

                    print("epoch : {}/{}, loss = {:.6f} for {} test data".format(epoch + 1, epochs, loss, self.view_names[view]))



           #         for batch_data, val_mask, val_duration, val_event in ae_validationloader:
           #             batch_data = batch_data[view].view(-1, batch_data[view].size(1)).to(device)

           #             optimizer.zero_grad()

                        # compressed features is what we are interested in
           #             reconstructed, compressed_test_features = model(batch_data)
           #             if epoch == epochs - 1: # save compressed_features of last epoch for each batch
           #                 temp3.append(compressed_test_features) # list of tensors of compressed for each batch

           #             val_loss = criterion(reconstructed, batch_data)

           #             val_loss.backward()

           #             optimizer.step()

           #             loss += val_loss.item()



            #        loss = loss / len(ae_testloader)

           #         print("epoch : {}/{}, loss = {:.6f} for {} val data".format(epoch + 1, epochs, loss, self.view_names[view]))



                compressed_train_features_view = torch.cat(temp, 0)
                AE_all_compressed_train_features.append(compressed_train_features_view)

          #      compressed_val_features_view = torch.cat(temp3, 0)
          #      AE_all_compressed_validation_features.append(compressed_val_features_view)

                compressed_test_features_view = torch.cat(temp2, 0)
                AE_all_compressed_test_features.append(compressed_test_features_view)



            for x in range(len(AE_all_compressed_train_features)):
                AE_all_compressed_train_features[x] = torch.detach(AE_all_compressed_train_features[x]) #detach gradient as we only need
                # selected features

    #        for x in range(len(AE_all_compressed_validation_features)):
    #            AE_all_compressed_validation_features[x] = torch.detach(AE_all_compressed_validation_features[x])

            for x in range(len(AE_all_compressed_test_features)):
                AE_all_compressed_test_features[x] = torch.detach(AE_all_compressed_test_features[x]) #detach gradient as we only need
                # selected features

            for c, view in enumerate(self.view_names):
                print("{} AE feature selection : {} of size {} (samples,features)".format(view, AE_all_compressed_train_features[c],
                                                                                          AE_all_compressed_train_features[c].shape))




            # Here we can create a tensor for all data, because we have the same feature size for each view
            # due to AE feature selection
            # (views, samples, features)
            data_AE_selected_train_PRAD = tensor_helper(AE_all_compressed_train_features)
     #       data_AE_selected_val_PRAD = tensor_helper(AE_all_compressed_validation_features)
            data_AE_selected_test_PRAD = tensor_helper(AE_all_compressed_test_features)

            self.train_set = MultiOmicsDataset(data_AE_selected_train_PRAD,
                                              self.duration_train,
                                              self.event_train,
                                              type = 'processed')

        #    self.val_set = MultiOmicsDataset(data_AE_selected_val_PRAD,
        #                                     self.duration_val,
        #                                     self.event_val,
        #                                     type = 'processed')

            self.test_set = MultiOmicsDataset(data_AE_selected_test_PRAD,
                                              self.duration_test,
                                              self.event_test,
                                              type = 'processed')



        if method.lower() == 'ppi':
            ppi_train = FeatureSelection.PPI(self.x_train, feature_names, self.view_names)
       #     ppi_val = FeatureSelection.PPI(self.x_val, feature_names, self.view_names)
            ppi_test = FeatureSelection.PPI(self.x_test, feature_names, self.view_names)

            data_train,edge_index_train, proteins_used_train = ppi_train.get_matrices()
      #      data_val, edge_index_val, proteins_used_val = ppi_val.get_matrices()
            data_test, edge_index_test, proteins_used_test = ppi_test.get_matrices()


            # edge indexes and used proteins are to be the same amongst train/val/test
            edge_index = edge_index_train
            proteins_used = proteins_used_train

            self.train_set = MultiOmicsDataset(data_train,
                                               self.duration_train,
                                               self.event_train,
                                               type = 'ppi')

      #      self.val_set = MultiOmicsDataset(data_val,
      #                                       self.duration_val,
      #                                       self.event_val,
      #                                       type = 'ppi')

            self.test_set = MultiOmicsDataset(data_test,
                                              self.duration_test,
                                              self.event_test,
                                              type = 'ppi')




            return edge_index, proteins_used




    def train_dataloader(self, batch_size):
        """
        Build training dataloader
        num_workers set to 0 by default because of some thread issue
        """
        train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=10)
        return train_loader

    def test_dataloader(self,batch_size):
        """
        Build test dataloader
        num_workers set to 0 by default because of some thread issue
        """
        test_loader = DataLoader(
            dataset=self.test_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=10)
        return test_loader



    def all_dataloader(self,batch_size):

        all_loader = DataLoader(dataset= self.all_set,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=10)
        return all_loader




def tensor_helper(tensor_list):
    #Turns a list of size (x) of tensors with dimensions (y,z) into a tensor of dimension (x,y,z)
    x = len(tensor_list)
    z, y = tensor_list[0].shape

    tensor_new = torch.zeros((x, z, y))
    for i, tensor in enumerate(tensor_list):
        tensor_new[i, :, :] = tensor

    return tensor_new

def flatten(l):
    """
    :param l: list input
    :return: flattened list (removal of one inner list layer)
    """
    return [item for sublist in l for item in sublist]

