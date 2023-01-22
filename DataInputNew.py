from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from torch.utils.data import DataLoader, Dataset
import FeatureSelection
from torch.optim import Adam
from torch import nn
from pycox import models
import AE as featAE
from sklearn.model_selection import StratifiedKFold



class MultiOmicsDataset(Dataset):

    # [tensor von allen samples view 1,  tensor von allen samples view 2, ... ]
    # in dem tensor jeweils features sample1, features sample 2, ...

    def __init__(self, X, duration, event, type= 'new'):
        self.type = type

        self.duration = duration
        self.event = event

        if self.type == 'temp':
            self.n_views = len(X) # number views (mRNA, DNA, microRNA, RPPA)
            self.X = X
            self.n_samples = X[0].size(0)


        elif self.type == 'processed2':
            self.n_folds = len(X)
            self.n_views = len(X[0])
            self.X = X
            self.n_samples = X[0][0].size(0)

            for c,fold in enumerate(self.X):
                for view in range(len(fold)):
                    self.X[c][view] = self.X[c][view].to(torch.float32)

            for fold in range(self.n_folds):
                self.duration[fold] = torch.from_numpy(self.duration[fold]).to(torch.float32)
                self.event[fold] = torch.from_numpy(self.event[fold]).to(torch.float32)

        else:

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


        elif self.type == 'temp':
            return [self.X[m][index, :] for m in range(self.n_views)], \
                   self.duration[index], self.event[index]



        elif self.type == 'processed2':

            for fold in range(self.n_folds):
                for m in range(self.n_views):
                    return self.X[fold][m][index, :], self.duration[fold][index], self.event[fold][index]

          #  return [self.X[fold][m][index, :] for fold in range(self.n_folds) for m in range(self.n_views)], \
          #         [self.duration[fold][index] for fold in range(self.n_folds)], \
          #         [self.event[fold][index] for fold in range(self.n_folds)]

        elif self.type == 'processed':
            return [self.X[m][index, :] for m in range(self.n_views)], \
                   self.duration[index], self.event[index]
        elif self.type == 'ppi':
            return self.X[index], self.duration[index], self.event[index]










def preprocess_features(
        df_train: pd.DataFrame, # train dataset
        df_test: pd.DataFrame, # test dataset
        df_val: pd.DataFrame, # validation dataset
        cols_std: List[str], # feature names, numeric variables
        cols_leave: List[str], # feature names, binary variables
        feature_offset : List[int],
        mode # set mode to 0 for first fold, then 1 so we don't preprocess test values each time
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
        x_val = mapper.transform(df_val).astype(np.float32)
        x_train_df = pd.DataFrame(x_train)
        x_val_df = pd.DataFrame(x_val)


        # Order by view so it works with Dataset class
        x_train_ordered_by_view = []
        x_val_ordered_by_view = []
        x_test_ordered_by_view = []

        if mode == 0:
            x_test = mapper.transform(df_test).astype(np.float32)
            x_test_df = pd.DataFrame(x_test)


        for x in range(len(feature_offset) - 3): # -3 bc we don't have duration/event in training tensor
            x_train_ordered_by_view.append(torch.tensor((x_train_df.iloc[:, feature_offset[x]:
                                                                            feature_offset[x + 1]]).values))

            x_val_ordered_by_view.append(torch.tensor((x_val_df.iloc[:, feature_offset[x]:
                                                                         feature_offset[x + 1]]).values))
            if mode == 0:
                x_test_ordered_by_view.append(torch.tensor((x_test_df.iloc[:, feature_offset[x]:
                                                                              feature_offset[x + 1]]).values))




    else:
        # Order by view so it works with Dataset class
        x_train_ordered_by_view = []
        x_val_ordered_by_view = []
        x_test_ordered_by_view = []

        for x in range(len(feature_offset) - 3): # -3 bc we don't have duration/event in training tensor
            x_train_ordered_by_view.append(torch.tensor((df_train.iloc[:, feature_offset[x]:
                                                                          feature_offset[x + 1]]).values))
            x_val_ordered_by_view.append(torch.tensor((df_val.iloc[:, feature_offset[x]:
                                                                        feature_offset[x + 1]]).values))

            if mode == 0:
                x_test_ordered_by_view.append(torch.tensor((df_test.iloc[:, feature_offset[x]:
                                                                            feature_offset[x + 1]]).values))





    return x_train_ordered_by_view, x_test_ordered_by_view,x_val_ordered_by_view


class SurvMultiOmicsDataModule(pl.LightningDataModule):
    """Input is the whole dataframe : We merge all data types together, with the feature offsets we can access
       certain data types ; dataframe also contains duration and event !"""
    def __init__(
            self, df, feature_offsets, view_names, n_durations = 10, onezeronorm_bool = False, cancer_name=None, which_views = [], n_folds = 2):
        super().__init__()
        self.df = df
        self.feature_offsets = feature_offsets # cumulative sum of features in list of features
        self.n_durations = n_durations
        self.view_names = [x.upper() for x in view_names]
        self.n_views = len(view_names)
        self.onezeronorm_bool = onezeronorm_bool
        self.cancer_name = cancer_name
        self.which_views = [x.upper() for x in which_views] # Decide which views to use for survival analysis
        self.n_folds = n_folds

    def setup(
            self,
            test_size=0.2,
            cols_std=None,   #numeric feature names
           # cols_leave=None, #binary feature names
            col_duration="duration",
            col_event="event",
            stage=None, #current stage of program (fit,test)
    ):

        # Decide which views to look at :
        # First, check if the user views input are actually in the current loaded cancer
        if len(self.which_views) != 0:
            missing_views = []
            for i in self.which_views:
                if i not in self.view_names:
                    missing_views.append(i)

            if len(missing_views) != 0:
                raise Exception("Sorry, the currently loaded cancer doesn't have the following view(s) : {}".format(missing_views))

            else:
                dropped_views = []
                # Drop views
                for c,view in enumerate(self.view_names):
                    # If there is a view we don't want..
                    if view not in self.which_views:
                        dropped_views.append(view)
                        self.df.drop(self.df.iloc[:, self.feature_offsets[c]:self.feature_offsets[c+1]], inplace=True, axis=1)
                        idx = self.view_names.index(view) + 1
                        for c2,i in enumerate(self.feature_offsets):
                            if c2 > idx:
                                self.feature_offsets[c2] -= (self.feature_offsets[idx] - self.feature_offsets[idx -1])
                        # Set to the same value because we don't have this view anymore, but want to keep the index
                        # structure of feature offsets
                        self.feature_offsets[idx] = self.feature_offsets[idx - 1]



                # Reset indices in dataframe and delete necessary elements in self.view_names and self.feature_offsets
                # and n_views
                for view in dropped_views:
                    self.n_views -=1
                    self.view_names.remove(view)
                # remove duplicates
                self.feature_offsets = set(self.feature_offsets)
                # return to right structure
                self.feature_offsets = sorted(list(self.feature_offsets))

















        if self.onezeronorm_bool == True: # Scale ALL (!) views (complete data) between 0&1
            # 01-Normalization if wanted
            event_and_duration = self.df.iloc[:,-2:].values # get event & duration values, as they don't need to be normalized to 01
            x = self.df.iloc[:,0:-2].values #returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            x_scaled = np.append(x_scaled, event_and_duration, axis=1)
            self.df = pd.DataFrame(x_scaled, columns=self.df.columns)

        """Split data into test and training set and training into training and validation set
           , preprocess this data with preprocess_features."""
        event_values = list(self.df[col_event].values)
        df_train_temp, df_test = train_test_split(self.df, test_size=test_size, stratify= event_values) # all --> train  , test



        n_train_samples = df_train_temp.shape[0]
        n_test_samples = df_test.shape[0]

        print("Split in train and test")
        print("non censored events in train : {} with {} samples in total".format(int(sum(list(df_train_temp[col_event].values))),n_train_samples))
        print("non censored events in test : {} with {} samples in total".format(int(sum(list(df_test[col_event].values))), n_test_samples))
     #   n_val_samples = df_val.shape[0]
        self.duration_train, self.duration_test =(df_train_temp[col_duration].values,                                   # ,self.duration_val
                                                df_test[col_duration].values
                                                )                                                                           # df_val[col_duration].values
        self.event_train, self.event_test = df_train_temp[col_event].values, df_test[col_event].values                                              # , self.event_val

        # Needed for cross validation
        self.event_train_df= df_train_temp[col_event]
        self.duration_train_df = df_train_temp[col_duration]
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


        data_folds, data_folds_targets, data_folds_durations, k_folds = self.cross_validation(df_train_temp, self.event_train_df, self.duration_train_df, self.n_folds)


        n_train_fold_samples = []
        n_val_fold_samples = []

        print("Cross validation : {} splits".format(k_folds))
        for fold in range(k_folds):
            print("Split {} : ".format(fold + 1))
            print("non censored events in train : {} with {} samples in total".format(int(np.sum(data_folds_targets[fold][0])), data_folds_targets[fold][0].size))
            print("non censored events in validation : {} with {} samples in total".format(int(np.sum(data_folds_targets[fold][1])), data_folds_targets[fold][1].size))
            n_train_fold_samples.append(data_folds_targets[fold][0].size)
            n_val_fold_samples.append(data_folds_targets[fold][1].size)





        self.train_folds_events = [x[0] for x in data_folds_targets]
        self.val_folds_events = [x[1] for x in data_folds_targets]
        self.train_folds_durations = [x[0] for x in data_folds_durations]
        self.val_folds_durations = [x[1] for x in data_folds_durations]
        self.train_folds = []
        self.val_folds = []
        # Preprocess train and test data with programmed function
        for fold in range(k_folds):
            self.x_train, self.x_test_actual, self.x_val = preprocess_features(
                df_train=data_folds[fold][0].drop(cols_drop, axis = 1), # drop duration/event from df, as we don't want these
                df_test=df_test.drop(cols_drop, axis = 1),
                df_val= data_folds[fold][1].drop(cols_drop, axis = 1),# for training and testing sets
                cols_std=cols_std,
                cols_leave=cols_leave,
                feature_offset= self.feature_offsets,
                mode= fold
            )
            self.train_folds.append(self.x_train)
            self.val_folds.append(self.x_val)

            if fold == 0: # if mode is set to 0
                self.x_test = self.x_test_actual

        self.train_mask_folds = []
        self.val_mask_folds = []

        for c,fold in enumerate(self.train_folds):
            self.train_folds[c] = [torch.nan_to_num(x_view) for x_view in self.train_folds[c]]
            self.train_mask = [torch.isnan(x_view) for x_view in self.train_folds[c]]
            self.train_mask_folds.append(self.train_mask)

            self.val_folds[c] = [torch.nan_to_num(x_view) for x_view in self.val_folds[c]]
            self.val_mask = [torch.isnan(x_view) for x_view in self.val_folds[c]]
            self.val_mask_folds.append(self.val_mask)

     #       self.x_train = [torch.nan_to_num(x_view) for x_view in self.x_train]
     #       self.x_train_mask = [torch.isnan(x_view) for x_view in self.x_train] #List of booleans for each numeric value in samples ;True : NaN values


        self.x_test = [torch.nan_to_num(x_view) for x_view in self.x_test]
        self.x_test_mask = [torch.isnan(x_view) for x_view in self.x_test]




        # For the purpose of finding views which have only 0 data, we don't go through each fold, but rather look at
        # uncrossvalidated full train/test data


        self.x_train_complete, temp, temp2 = preprocess_features(
            df_train=df_train_temp.drop(cols_drop, axis = 1), # drop duration/event from df, as we don't want these
            df_test=df_test.drop(cols_drop, axis = 1), # this also doesnt matter
            df_val= data_folds[0][1].drop(cols_drop, axis = 1),# this doesnt matter here, just a placeholder
            cols_std=cols_std,
            cols_leave=cols_leave,
            feature_offset= self.feature_offsets,
            mode= 1
        )

        self.x_train_complete = [torch.nan_to_num(x_view) for x_view in self.x_train_complete]




        # We check for each view how many 0s this view contains
        train_zeros = []
        test_zeros = []
        for c,view in enumerate(self.x_train_complete):
            curr_view_count_train = torch.count_nonzero(view)
            train_zeros.append(curr_view_count_train)
            curr_view_count_test = torch.count_nonzero(self.x_test[c])
            test_zeros.append(curr_view_count_test)


        removed_views_index = []

        for x,count in enumerate(train_zeros):
            # If there arent atleast 10 % values greater than 0 for this view for all samples, remove this view
            # from consideration
            if train_zeros[x] < int(0.1 * ((self.feature_offsets[x + 1] - self.feature_offsets[x]) * n_train_samples))\
                    or \
                test_zeros[x] < int(0.1 * ((self.feature_offsets[x + 1] - self.feature_offsets[x]) * n_test_samples)):

                print("{} has nearly only 0 values. We don't take this data into consideration.".format(self.view_names[x]))
                removed_views_index.append(x)

        for index in sorted(removed_views_index, reverse=True):
            for c,fold in enumerate(self.train_folds):
                del self.train_folds[c][index]
                del self.val_folds[c][index]


            del self.x_test[index]

            del self.view_names[index]
            diff = self.feature_offsets[index + 1] - self.feature_offsets[index]
            for c,_ in enumerate(self.feature_offsets):
                if c > (index + 1):
                    self.feature_offsets[c] = self.feature_offsets[c] - diff
            del self.feature_offsets[index+1]

        self.n_views = self.n_views - len(removed_views_index)

        ############################ CASTING NEEDED FOR NEURAL NETS ##########################################
        for c in range(self.n_folds):
            # Cast all elements to torch.float32
            self.train_folds_durations[c] = torch.from_numpy(self.train_folds_durations[c]).to(torch.float32)
            self.train_folds_events[c] = torch.from_numpy(self.train_folds_events[c]).to(torch.float32)
            self.val_folds_durations[c] = torch.from_numpy(self.val_folds_durations[c]).to(torch.float32)
            self.val_folds_events[c] = torch.from_numpy(self.val_folds_events[c]).to(torch.float32)

        # Also cast train duration & events
        self.duration_test = torch.from_numpy(self.duration_test).to(torch.float32)
        self.event_test = torch.from_numpy(self.event_test).to(torch.float32)





        return n_train_fold_samples, n_val_fold_samples, n_test_samples, self.view_names


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



    def cross_validation(self,data_df, event_df, duration_df, k_folds):
        """
        Cross Validation, stratified based on event indicators
        :param data_df : data as pandas dataframe
        :param k_folds: Amount of folds to use
        :param event_df : event indicators for data in dataframe (targets for stratification) ; as pandas dataframe
        :param duration_df : duration indicators for data in dataframe (targets for stratification) ; as pandas dataframe
        :return: list of k_folds many train/val splits
        """

        skfold = StratifiedKFold(n_splits=k_folds,shuffle=True,random_state=42)

        # save all folds in here, each sublist of type [train_fold_df, val_fold_df]
        folds = [[] for i in range(k_folds)]
        # save all targets (event indicators) here, each sublist of type [train_fold_targets_df, val_fold_targets_df]
        folds_targets = [[] for i in range(k_folds)]
        # Save folds durations here
        folds_durations = [[] for i in range(k_folds)]

        c = 0
        for train_idx, val_idx in skfold.split(data_df, event_df):
            folds[c].append(data_df.iloc[train_idx])
            folds[c].append(data_df.iloc[val_idx])

            folds_targets[c].append(event_df.iloc[train_idx].values)
            folds_targets[c].append(event_df.iloc[val_idx].values)

            folds_durations[c].append(duration_df.iloc[train_idx].values)
            folds_durations[c].append(duration_df.iloc[val_idx].values)

            c += 1


        return folds, folds_targets, folds_durations, k_folds




    def feature_selection(self, method = None,
                          feature_names = None, # for PPI network
                          components = None, # for PCA
                          thresholds = None):





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
            PCA_train_tensors = [[] for i in range(len(self.train_folds))]
            PCA_val_tensors = [[] for i in range(len(self.train_folds))]
            PCA_test_tensors = [[] for i in range(len(self.train_folds))]




            for c,fold in enumerate(self.train_folds):
                for view in range(self.n_views):
                    # Initialize PCA objects for both train and test with same components
                    view_train_PCA = FeatureSelection.F_PCA(self.train_folds[c][view], components=components[view])
                    view_val_PCA = FeatureSelection.F_PCA(self.val_folds[c][view], components=components[view])
                    view_test_PCA = FeatureSelection.F_PCA(self.x_test[view], components=components[view])
                    # Apply PCA just to the train set
                    obj_PCA = view_train_PCA.apply_pca()
                    # Fit & Transform the train set
                    train_data = view_train_PCA.fit_transform_pca(obj_PCA)
                    # Only transform the test and val set with the given PCA object of train
                    test_data = view_test_PCA.transform_pca(obj_PCA)
                    val_data = view_val_PCA.transform_pca(obj_PCA)


                    PCA_train_tensors[c].append(torch.tensor(train_data))
                    PCA_val_tensors[c].append(torch.tensor(val_data))
                    PCA_test_tensors[c].append(torch.tensor(test_data))




            # Before returning, we cast all elements to torch.float32
            for c,fold in enumerate(PCA_train_tensors):
                for c2, view in enumerate(fold):
                    PCA_train_tensors[c][c2] = PCA_train_tensors[c][c2].to(torch.float32)
                    PCA_val_tensors[c][c2] =PCA_val_tensors[c][c2].to(torch.float32)
                    PCA_test_tensors[c][c2] =PCA_test_tensors[c][c2].to(torch.float32)












            return PCA_train_tensors,PCA_val_tensors,PCA_test_tensors,\
                   self.train_folds_durations,self.train_folds_events,\
                   self.val_folds_durations,self.val_folds_events,\
                   self.duration_test,self.event_test









        """
        #   PCA_train_tensors = []
        #   PCA_test_tensors = []
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
            
            
            
        self.train_set = MultiOmicsDataset(PCA_train_tensors,
                                               self.duration_train,
                                               self.event_train,
                                               type = 'processed')

        self.test_set = MultiOmicsDataset(PCA_test_tensors,
                                          self.duration_test,
                                          self.event_test,
                                              type = 'processed')
        """




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







        if method.lower() == 'variance':
            variance_train_tensors = [[] for i in range(len(self.train_folds))]
            variance_val_tensors = [[] for i in range(len(self.train_folds))]
            variance_test_tensors = [[] for i in range(len(self.train_folds))]




            for c,fold in enumerate(self.train_folds):
                for view in range(self.n_views):
                    # Initialize Variance objects for both train and test with same components
                    view_train_variance = FeatureSelection.F_VARIANCE(self.train_folds[c][view], threshold=thresholds[view])
                    view_val_variance = FeatureSelection.F_VARIANCE(self.val_folds[c][view], threshold=thresholds[view])
                    view_test_variance = FeatureSelection.F_VARIANCE(self.x_test[view], threshold=thresholds[view])
                    # Apply Variance just to the train set
                    obj_variance = view_train_variance.apply_variance()
                    # Fit & Transform the train set
                    train_data = view_train_variance.fit_transform_variance(obj_variance)
                    # Only transform the test and val set with the given Variance object of train
                    test_data = view_test_variance.transform_variance(obj_variance)
                    val_data = view_val_variance.transform_variance(obj_variance)


                    variance_train_tensors[c].append(torch.tensor(train_data))
                    variance_val_tensors[c].append(torch.tensor(val_data))
                    variance_test_tensors[c].append(torch.tensor(test_data))


            # Before returning, we cast all elements to torch.float32
            for c,fold in enumerate(variance_train_tensors):
                for c2, view in enumerate(fold):
                    variance_train_tensors[c][c2] = variance_train_tensors[c][c2].to(torch.float32)
                    variance_val_tensors[c][c2] =variance_val_tensors[c][c2].to(torch.float32)
                    variance_test_tensors[c][c2] =variance_test_tensors[c][c2].to(torch.float32)


            return variance_train_tensors,variance_val_tensors,variance_test_tensors, \
                   self.train_folds_durations,self.train_folds_events, \
                   self.val_folds_durations,self.val_folds_events, \
                   self.duration_test,self.event_test




            """

            variance_train_tensors = []
            variance_test_tensors = []
            variance_all_tensors = []

            thresholds = [0.8,0.8,0.8,0.8]

            for view in range(self.n_views):
                # Initialize variance objects for both train and test with same components
                view_train_variance = FeatureSelection.F_VARIANCE(self.x_train[view], threshold=thresholds[view])
                view_test_variance = FeatureSelection.F_VARIANCE(self.x_test[view], threshold=thresholds[view])
                # Apply variance threshold just to the train set
                obj_variance = view_train_variance.apply_variance()
                # Fit & Transform the train set
                train_data = view_train_variance.fit_transform_variance(obj_variance)
                # Only transform the test set with the given PCA object of train
                test_data = view_test_variance.transform_variance(obj_variance)

                variance_train_tensors.append(torch.tensor(train_data))
                variance_test_tensors.append(torch.tensor(test_data))


            self.train_set = MultiOmicsDataset(variance_train_tensors,
                                               self.duration_train,
                                               self.event_train,
                                               type = 'processed')

            self.test_set = MultiOmicsDataset(variance_test_tensors,
                                              self.duration_test,
                                              self.event_test,
                                              type = 'processed')


            """
            """
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
            """


        if method.lower() == 'ae':


            # Define learning rate
            learning_rate_train = 0.0001
            learning_rate_val = 0.0001
            learning_rate_test = 0.0001

            # Define Loss
            criterion = nn.MSELoss()

            # Define device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            #Define number of epochs
            n_epochs = 5


            # Get amount samples

            batch_size_train = 64
            batch_size_val = 64
            batch_size_test = 16


            # List of lists to save output for each fold
            selected_train_features = [[] for _ in range(self.n_folds)]
            selected_val_features = [[] for _ in range(self.n_folds)]
            selected_test_features = [[] for _ in range(self.n_folds)]



            for c_fold in range(self.n_folds):
                dimensions_train = [x.shape[1] for x in self.train_folds[c_fold]]
                dimensions_val = [x.shape[1] for x in self.val_folds[c_fold]]
                dimensions_test = [x.shape[1] for x in self.x_test]


                # TODO : need to be resetted for each fold bc class is somehow overwriting them after each fold ??
                layers_train = [[512,256,128] for x in range(self.n_views)]
                layers_val = [[512,256,128] for x in range(self.n_views)]
                layers_test = [[256,128] for x in range(self.n_views)]

                activation_functions_train = [['relu'] for i in range(self.n_views)]
                activation_functions_val = [['relu'] for i in range(self.n_views)]
                activation_functions_test = [['relu'] for i in range(self.n_views)]

                batch_normalization_train = [[] for i in range(self.n_views)]
                batch_normalization_val = [[] for i in range(self.n_views)]
                batch_normalization_test = [[] for i in range(self.n_views)]

                dropout_layers_train = [[] for i in range(self.n_views)]
                dropout_layers_val = [[] for i in range(self.n_views)]
                dropout_layers_test = [[] for i in range(self.n_views)]

                for c,dim in enumerate(layers_train):
                    for i in range(len(dim)):
                        batch_normalization_train[c].append('no')
                        batch_normalization_val[c].append('no')
                        batch_normalization_test[c].append('no')
                        dropout_layers_train[c].append('no')
                        dropout_layers_val[c].append('no')
                        dropout_layers_test[c].append('no')


                # Call train net
                net_train = featAE.AE(self.view_names,
                                      dimensions_train, layers_train,
                                      activ_funcs= activation_functions_train,
                                      batch_norm= batch_normalization_train,
                                      dropout_layers= dropout_layers_train,
                                      dropout_prob= 0.1,
                                      dropout_bool=False,
                                      batch_norm_bool=False,
                                      type_ae='none',
                                      print_bool=False)

                # Call val net (same structure)
                net_val = featAE.AE(self.view_names,
                                    dimensions_val, layers_val,
                                    activ_funcs= activation_functions_val,
                                    batch_norm= batch_normalization_val,
                                    dropout_layers= dropout_layers_val,
                                    dropout_prob= 0.1,
                                    dropout_bool=False,
                                    batch_norm_bool=False,
                                    type_ae='none',
                                    print_bool=False)


                # Call test net (same structure)
                net_test = featAE.AE(self.view_names,
                                     dimensions_test, layers_test,
                                     activ_funcs= activation_functions_test,
                                     batch_norm= batch_normalization_test,
                                     dropout_layers= dropout_layers_test,
                                     dropout_prob= 0.1,
                                     dropout_bool=False,
                                     batch_norm_bool=False,
                                     type_ae='none',
                                     print_bool=False)

                # optimizers
                optimizer_train = Adam(net_train.parameters(), lr= learning_rate_train)
                optimizer_val = Adam(net_val.parameters(), lr= learning_rate_val)
                optimizer_test = Adam(net_test.parameters(), lr= learning_rate_test)



                # Load Data for current fold with Dataloaders for batching structure
                self.train_set = MultiOmicsDataset(self.train_folds[c_fold], self.train_folds_durations[c_fold], self.train_folds_events[c_fold], type = 'temp')
                self.val_set = MultiOmicsDataset(self.val_folds[c_fold], self.val_folds_durations[c_fold], self.val_folds_events[c_fold], type = 'temp')
                self.test_set = MultiOmicsDataset(self.x_test, self.duration_test, self.event_test, type = 'temp')

                # drop last false since we are in feature selection and don't want to lose data here
                ae_trainloader = DataLoader(self.train_set,batch_size=batch_size_train,shuffle=True,drop_last=False)
                ae_valloader = DataLoader(self.val_set, batch_size=batch_size_val, shuffle=True,drop_last=False)
                ae_testloader = DataLoader(self.test_set,batch_size=batch_size_test,shuffle=True,drop_last=False)


                # Train for current epoch
                for epoch in range(n_epochs):
                    loss_train = 0
                    loss_val = 0
                    loss_test = 0
                    ############################# TRAIN SET ##################################
                    for train_batch, train_duration, train_event in ae_trainloader:
                        # Send data to device if possible
                        for view in range(len(train_batch)):
                            train_batch[view] = train_batch[view].to(device=device)

                        # Structure must be a tuple
                        train_batch = tuple(train_batch)

                        optimizer_train.zero_grad()

                        data_middle, final_out, input_data_raw = net_train(*train_batch)

                        # If we're in the last epoch, we save our data in the middle (our selected features)
                        if epoch == n_epochs - 1:
                            selected_train_features[c_fold].append(data_middle)

                        train_loss = 0
                        for view in range(len(train_batch)):
                            train_loss += criterion(input_data_raw[view], final_out[view])

                        train_loss.backward()

                        optimizer_train.step()

                        loss_train += train_loss.item()


                    loss_train = loss_train / len(ae_trainloader)

                    print("epoch : {}/{}, loss = {:.6f} for training data".format(epoch + 1, n_epochs, loss_train))


                    ###################### VALIDATION SET ##########################
                    for val_batch, val_duration, val_event in ae_valloader:

                        # Send data to device if possible
                        for view in range(len(train_batch)):
                            val_batch[view] = val_batch[view].to(device=device)

                        # Structure must be a tuple
                        val_batch = tuple(val_batch)

                        optimizer_val.zero_grad()

                        data_middle, final_out, input_data_raw = net_val(*val_batch)

                        # If we're in the last epoch, we save our data in the middle (our selected features)
                        if epoch == n_epochs - 1:
                            selected_val_features[c_fold].append(data_middle)

                        val_loss = 0
                        for view in range(len(val_batch)):
                            val_loss += criterion(input_data_raw[view], final_out[view])

                        val_loss.backward()

                        optimizer_train.step()

                        loss_val += val_loss.item()


                    loss_val = loss_val / len(ae_valloader)

                    print("epoch : {}/{}, loss = {:.6f} for validation data".format(epoch + 1, n_epochs, loss_val))



                    ######################### TEST SET ##########################
                    for test_batch, test_duration, test_event in ae_testloader:
                        # Send data to device if possible
                        for view in range(len(test_batch)):
                            test_batch[view] = test_batch[view].to(device=device)

                        # Structure must be a tuple
                        test_batch = tuple(test_batch)

                        optimizer_test.zero_grad()

                        data_middle, final_out, input_data_raw = net_test(*test_batch)

                        if epoch == n_epochs - 1:
                            selected_test_features[c_fold].append(data_middle)

                        test_loss = 0
                        for view in range(len(test_batch)):
                            test_loss += criterion(input_data_raw[view], final_out[view])

                        test_loss.backward()

                        optimizer_test.step()

                        loss_test = test_loss.item()


                    loss_test = loss_test / len(ae_testloader)

                    print("epoch : {}/{}, loss = {:.6f} for test data".format(epoch + 1, n_epochs, loss_test))



            # We need to concatenate the output data of each batch for each fold for each view respectively
            # so we can pass all of it to our neural survival nets

            ae_train_tensors = [[] for  _ in range(self.n_folds)]
            ae_val_tensors = [[] for _ in range(self.n_folds)]
            ae_test_tensors = [[] for _ in range(self.n_folds)]

            for c_fold, fold in enumerate(selected_train_features):
                view_counter = 0
                while view_counter < self.n_views:
                    curr_view_values_train = []
                    for c_batch,batch in enumerate(fold):
                        curr_view_values_train.append(selected_train_features[c_fold][c_batch][view_counter].detach())
                    view_counter += 1
                    view_values_cat_train = torch.cat(tuple(curr_view_values_train), dim=0)
                    ae_train_tensors[c_fold].append(view_values_cat_train)



            for c_fold, fold in enumerate(selected_val_features):
                view_counter = 0
                while view_counter < self.n_views:
                    curr_view_values_val = []
                    for c_batch,batch in enumerate(fold):
                        curr_view_values_val.append(selected_val_features[c_fold][c_batch][view_counter].detach())
                    view_counter += 1
                    view_values_cat_val = torch.cat(tuple(curr_view_values_val), dim=0)
                    ae_val_tensors[c_fold].append(view_values_cat_val)

            for c_fold, fold in enumerate(selected_test_features):
                view_counter = 0
                while view_counter < self.n_views:
                    curr_view_values_test = []
                    for c_batch,batch in enumerate(fold):
                        curr_view_values_test.append(selected_test_features[c_fold][c_batch][view_counter].detach())
                    view_counter += 1
                    view_values_cat_test = torch.cat(tuple(curr_view_values_test), dim=0)
                    ae_test_tensors[c_fold].append(view_values_cat_test)




            return ae_train_tensors,ae_val_tensors,ae_test_tensors, \
                   self.train_folds_durations,self.train_folds_events, \
                   self.val_folds_durations,self.val_folds_events, \
                   self.duration_test,self.event_test






            """
            # We get the feature size dimensions of each view (input dims for AE)
            # As they are the same in training and test set, we only iterate over the train set
            dimensions_train = []
            dimensions_test = []
            for view in self.x_train:
                dim = view.size(1)
                dimensions_train.append(dim)
                dimensions_test.append(dim)

            # layers in AE for each view
           # layers = [[] for x in range(self.n_views)]
            layers_train = [[512,256,128] for x in range(self.n_views)]
            layers_test = [[128,64] for x in range(self.n_views)]
         #   for c,d in enumerate(dimensions):
         #       dim_reduct = d
         #       # Reduce input feature dim to 1/10 of feature dim
         #       while int(d/10) < dim_reduct:
         #           dim_reduct //= 2
         #           layers[c].append(dim_reduct)

            # Activation functions in AE for each layer of each view

            # Need two since structure is changed in AE call to actual activ. functions # TODO : fix, auch in AE Klasse --> dann kann man das auch über application direkt einstellen
            activation_functions_train = [['relu'] for i in range(self.n_views)]
            activation_functions_test = [['relu'] for i in range(self.n_views)]

            batch_normalization_train = [[] for i in range(self.n_views)]
            batch_normalization_test = [[] for i in range(self.n_views)]

            dropout_layers_train = [[] for i in range(self.n_views)]
            dropout_layers_test = [[] for i in range(self.n_views)]

            for c,dim in enumerate(layers_train):
                for i in range(len(dim)):
                    batch_normalization_train[c].append('no')
                    batch_normalization_test[c].append('no')
                    dropout_layers_train[c].append('no')
                    dropout_layers_test[c].append('no')



            # Define learning rate
            learning_rate_train = 0.0001
            learning_rate_test = 0.0001

            # Define Loss
            criterion = nn.MSELoss()

            # Define device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            #Define number of epochs
            n_epochs = 50


            # Get amount samples
            n_train_samples  = self.x_train[0].size(0)
            n_test_samples = self.x_test[0].size(0)

            batch_size_train = 32
            batch_size_test = 8


            # Lists to save selected features
            selected_train_features = []
            selected_test_features = []



            # Call train net
            net_train = featAE.AE(self.view_names,
                                  dimensions_train, layers_train,
                                  activ_funcs= activation_functions_train,
                                  batch_norm= batch_normalization_train,
                                  dropout_layers= dropout_layers_train,
                                  dropout_prob= 0.1,
                                  dropout_bool=False,
                                  batch_norm_bool=False,
                                  type_ae='none',
                                  print_bool=False)

            # Call test net (same structure)
            net_test = featAE.AE(self.view_names,
                                 dimensions_test, layers_test,
                                 activ_funcs= activation_functions_test,
                                 batch_norm= batch_normalization_test,
                                 dropout_layers= dropout_layers_test,
                                 dropout_prob= 0.1,
                                 dropout_bool=False,
                                 batch_norm_bool=False,
                                 type_ae='none',
                                 print_bool=False)

            optimizer_train = Adam(net_train.parameters(), lr= learning_rate_train)
            optimizer_test = Adam(net_test.parameters(), lr= learning_rate_test)


            # Load Data with Dataloaders for batching structure
            self.train_set = MultiOmicsDataset(self.x_train, self.duration_train, self.event_train, type = 'new')

            self.test_set = MultiOmicsDataset(self.x_test, self.duration_test, self.event_test, type = 'new')

            ae_trainloader = DataLoader(self.train_set,batch_size=batch_size_train,shuffle=True,drop_last=True)

            ae_testloader = DataLoader(self.test_set,batch_size=batch_size_test,shuffle=True,drop_last=True)



            for epoch in range(n_epochs):
                loss_train = 0
                loss_test = 0
                ############################# TRAIN SET ##################################
                for train_batch, train_mask, train_duration, train_event in ae_trainloader:
                    # Send data to device if possible
                    for view in range(len(train_batch)):
                        train_batch[view] = train_batch[view].to(device=device)

                    # Structure must be a tuple
                    train_batch = tuple(train_batch)

                    optimizer_train.zero_grad()

                    data_middle, final_out, input_data_raw = net_train(*train_batch)

                    # If we're in the last epoch, we save our data in the middle (our selected features)
                    if epoch == n_epochs - 1:
                        selected_train_features.append(data_middle)

                    train_loss = 0
                    for view in range(len(train_batch)):
                        train_loss += criterion(input_data_raw[view], final_out[view])

                    train_loss.backward()

                    optimizer_train.step()

                    loss_train += train_loss.item()


                loss_train = loss_train / len(ae_trainloader)

                print("epoch : {}/{}, loss = {:.6f} for training data".format(epoch + 1, n_epochs, loss_train))

                ######################### TEST SET ##########################
                for test_batch, test_mask, test_duration, test_event in ae_testloader:
                    # Send data to device if possible
                    for view in range(len(test_batch)):
                        test_batch[view] = test_batch[view].to(device=device)

                    # Structure must be a tuple
                    test_batch = tuple(test_batch)

                    optimizer_test.zero_grad()

                    data_middle, final_out, input_data_raw = net_test(*test_batch)

                    if epoch == n_epochs - 1:
                        selected_test_features.append(data_middle)

                    test_loss = 0
                    for view in range(len(test_batch)):
                        test_loss += criterion(input_data_raw[view], final_out[view])

                    test_loss.backward()

                    optimizer_test.step()

                    loss_test = test_loss.item()


                loss_test = loss_test / len(ae_testloader)

                print("epoch : {}/{}, loss = {:.6f} for test data".format(epoch + 1, n_epochs, loss_test))
                """


            """



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

            """


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



  #  def all_dataloader(self,batch_size):

  #      all_loader = DataLoader(dataset= self.all_set,
  #                              batch_size=batch_size,
  #                              shuffle=True,
  #                              drop_last=True,
  #                              num_workers=10)
  #      return all_loader




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


#%%
