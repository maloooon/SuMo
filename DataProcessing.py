import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from torch.utils.data import DataLoader, Dataset
import FeatureSelection
from torch.optim import Adam
from torch import nn
import AE as featAE
from sklearn.model_selection import StratifiedKFold
import optuna
import copy
import os


class MultiOmicsDataset(Dataset):

    def __init__(self, X, duration, event, type= 'tensor'):
        """
        :param X: Data input ; dtype : List of Tensors or Numpy Arraya of Floats (n_samples,n_features) , one for each view
        :param duration: duration (time-to-event or time-to-censorship) ; dtype : Tensor of Int (n_samples,1 [duration value])
        :param event: event (1 : not censored, 0 : censored) ; dtype : Tensor of Int (n_samples, 1 [event value])
        :param type: Type of the Data Input X (Tensors or Numpy Arrays) ; dtype : String ['tensor', 'np']
        """

        self.type = type
        self.duration = duration
        self.event = event

        if self.type == 'tensor':
            self.n_views = len(X) # number views (mRNA, DNA, microRNA, RPPA)
            self.X = X
            self.n_samples = X[0].size(0)

        if self.type == 'np':
            # Numpy arrays
            self.n_views = len(X)
            self.X = X
            self.n_samples = X[0].shape[0]


    def __len__(self):
        """
        :return: the amount of samples ; dtype : Int
        """
        return self.n_samples


    def __getitem__(self, index):
        """
        :param index: Index for data for each view, duration and event value
        :return: data for each view ; dtype : List of Tensors/Numpy Arrays
                 duration value ; dtype : Tensor/Numpy Array
                 event value ; dtype : Tensor/Numpy Array
        """

        return [self.X[m][index, :] for m in range(self.n_views)], \
               self.duration[index], self.event[index]


def preprocess_features(
        df_train,
        df_test,
        df_val,
        cols_std,
        cols_leave,
        feature_offset,
        mode,
        preprocess_type):
    """
    Preprocessing data

    :param df_train: Current train data ; dtype : (pandas) Dataframe (rows : samples, columns : features)
    :param df_test: Current test data ; dtype : (pandas) Dataframe (rows : samples, columns : features)
    :param df_val: Current validation data ; dtype : (pandas) Dataframe (rows : samples, columns : features)
    :param cols_std: Feature names with numeric feature values (across all views) ; dtype : List of Strings
    :param cols_leave: Feature names with binary feature values (across all views) ; dtype : List of Strings
    :param feature_offset: Feature Offsets for different views
                          [0, n_feats_view_1, n_feats_view_1 + n_feats_view_2,..]
                          ; dtype : List of Int [Cumulative Sum]
    :param mode: Mode deciding whether we preprocess test data [Needed for Cross-Validation preprocessing, since we
                 only want to preprocess our test data once, but need to preprocess each train/val fold]
                 ; dtype : Int [0 for preprocessing test set, anything else for no preprocessing]
    :param preprocess_type : Type of Preprocessing (Standardization/Normalization/None) ; dtype : String
    :return: Feature values ordered by views for train/validation/test ;
             dtype : List of Tensors (n_samples, n_features) [for each view]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if cols_std is not None and preprocess_type.lower() != 'none':

        if preprocess_type.lower() == 'standardize':
            standardize = [([col], StandardScaler()) for col in cols_std]
        if preprocess_type.lower() == 'normalize':
            standardize = [([col], MinMaxScaler()) for col in cols_std]
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

    def __init__(
            self,
            df,
            feature_offsets,
            view_names,
            n_durations = 10,
            cancer_name=None,
            which_views = [],
            n_folds = 2,
            type_preprocess = 'standardize'):
        """
        :param df: Complete data (feature values each view, duration, event)
                   ; dtype : (pandas) Dataframe (n_samples, (n_features,1 [duration], 1 [event]))
        :param feature_offsets: Feature Offsets for different views aswell as duration & event [List ends with [..,x,x+1,x+2]
                               [0, n_feats_view_1, n_feats_view_1 + n_feats_view_2,..]
                          ; dtype : List of Int [Cumulative Sum]
        :param view_names: Names of all views ; dtype : List of Strings
        :param n_durations: TODO: not needed I think
        :param cancer_name: Name of current looked at cancer ; dtype : String
        :param which_views: Name of views we currently want to analyze ; if empty, all possible views are taken
                           ; dtype : List of Strings
        :param n_folds: Number of Folds for Cross-Validation ; dtype : Int
        :param type_preprocess: Type of Preprocessing (Standardization/Normalization/None) ; dtype : String
        """
        super().__init__()
        self.df = df
        self.feature_offsets = feature_offsets # cumulative sum of features in list of features
        self.n_durations = n_durations
        self.view_names = [x.upper() for x in view_names]
        self.n_views = len(view_names)
        self.cancer_name = cancer_name
        self.which_views = [x.upper() for x in which_views] # Decide which views to use for survival analysis
        self.n_folds = n_folds
        self.type_preprocess = type_preprocess

    def setup(
            self,
            test_size=0.2,
            cols_std=None,
            cols_leave=None,
            col_duration="duration",
            col_event="event"
    ):
        """
        :param test_size: Size of test set in train/test split ; dtype : Flaot
        :param cols_std: Feature names with numeric feature values (across all views) ; dtype : List of Strings [of feature names]
        :param cols_leave : Feature names with binary feature values (across all views) ; dtype : List of Strings [of feature names]
        :param col_duration: Column name for duration values in datas Dataframe ; dtype : String
        :param col_event: Column name for event values in datas Dataframe ; dtype : String
        :return: n_train_samples : List of Int [Samples]
                 n_val_samples : List of Int [Samples]
                 n_test_samples : Int [Samples]
                 view_names : List of Strings
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


        # Split data into test and training set and training
        # into training and validation set, preprocess this data with preprocess_features
        event_values = list(self.df[col_event].values)
        df_train_temp, df_test = train_test_split(self.df, test_size=test_size, stratify= event_values)

        n_train_samples = df_train_temp.shape[0]
        n_test_samples = df_test.shape[0]

        print("Split in train and test")
        print("non censored events in train : {} with {} samples in total".
              format(int(sum(list(df_train_temp[col_event].values))),n_train_samples))
        print("non censored events in test : {} with {} samples in total".
              format(int(sum(list(df_test[col_event].values))), n_test_samples))

        self.duration_train, self.duration_test =(df_train_temp[col_duration].values,
                                                df_test[col_duration].values
                                                )
        self.event_train, self.event_test = df_train_temp[col_event].values, df_test[col_event].values

        # Needed for cross validation
        self.event_train_df= df_train_temp[col_event]
        self.duration_train_df = df_train_temp[col_duration]


        # Drop all binary values, aswell as NaN values ; This might lead to problems, as numeric values 0/1 will
        # be seen as binary values ; we leave that out and just remove the columns (features)
        # which have only NaN values, see cols_remove below
      #  cols_leave = [col for col in self.df
      #               if np.isin(self.df[col].dropna().unique(), [0, 1]).all()]
        # Remove event from binary column values (we are only interested in feature values from views)


        if cols_leave is None:
            cols_leave = []

        if 'event' in cols_leave:
            cols_leave.remove('event')


        # Columns we don't want to standardize
        cols_survival = [col_duration, col_event]
        cols_drop = cols_survival





        # Columns (features) which have only NaN values
        cols_remove = self.df.columns[self.df.isna().all()].tolist()

        cols_leave = cols_remove

        # features with numeric values
        if cols_std is None:
            cols_std = [
                col for col in self.df.columns if col not in cols_leave + cols_drop + cols_remove
            ]


        data_folds, data_folds_targets, data_folds_durations, k_folds = self.cross_validation(df_train_temp,
                                                                                              self.event_train_df,
                                                                                              self.duration_train_df,
                                                                                              self.n_folds)


        n_train_fold_samples = []
        n_val_fold_samples = []

        print("Cross validation : {} splits".format(k_folds))
        for fold in range(k_folds):
            print("Split {} : ".format(fold + 1))
            print("non censored events in train : {} with {} samples in total".
                  format(int(np.sum(data_folds_targets[fold][0])), data_folds_targets[fold][0].size))
            print("non censored events in validation : {} with {} samples in total".
                  format(int(np.sum(data_folds_targets[fold][1])), data_folds_targets[fold][1].size))
            n_train_fold_samples.append(data_folds_targets[fold][0].size)
            n_val_fold_samples.append(data_folds_targets[fold][1].size)


        self.train_folds_events = [x[0] for x in data_folds_targets]
        self.val_folds_events = [x[1] for x in data_folds_targets]
        self.train_folds_durations = [x[0] for x in data_folds_durations]
        self.val_folds_durations = [x[1] for x in data_folds_durations]
        self.train_folds = []
        self.val_folds = []

        # Preprocess train and test data with programmed function
        if self.type_preprocess.lower() != 'none':
            print("Preprocessing data....")
        for fold in range(k_folds):
            self.x_train, self.x_test_actual, self.x_val = preprocess_features(
                df_train=data_folds[fold][0].drop(cols_drop, axis = 1), # drop duration/event from df, as we don't want these
                df_test=df_test.drop(cols_drop, axis = 1),
                df_val= data_folds[fold][1].drop(cols_drop, axis = 1),# for training and testing sets
                cols_std=cols_std,
                cols_leave=cols_leave,
                feature_offset= self.feature_offsets,
                mode= fold,
                preprocess_type= self.type_preprocess
            )
            self.train_folds.append(self.x_train)
            self.val_folds.append(self.x_val)

            # We preprocess the test set only for the first fold (bc it stays the same across all folds)
            if fold == 0: # if mode is set to 0
                self.x_test = self.x_test_actual

        # Features mapped to True/False, where True means the feature value is NaN
        self.train_mask_folds = []
        self.val_mask_folds = []

        for c,fold in enumerate(self.train_folds):
            self.train_folds[c] = [torch.nan_to_num(x_view) for x_view in self.train_folds[c]]
            self.train_mask = [torch.isnan(x_view) for x_view in self.train_folds[c]]
            self.train_mask_folds.append(self.train_mask)

            self.val_folds[c] = [torch.nan_to_num(x_view) for x_view in self.val_folds[c]]
            self.val_mask = [torch.isnan(x_view) for x_view in self.val_folds[c]]
            self.val_mask_folds.append(self.val_mask)


        self.x_test = [torch.nan_to_num(x_view) for x_view in self.x_test]
        self.x_test_mask = [torch.isnan(x_view) for x_view in self.x_test]

        # For the purpose of finding views which nearly have only NaN data,
        # we don't go through each fold, but rather look at uncrossvalidated full train/test data
        # temp/temp2 not needed
        self.x_train_complete, temp, temp2 = preprocess_features(
            df_train=df_train_temp.drop(cols_drop, axis = 1), # drop duration/event from df, as we don't want these
            df_test=df_test.drop(cols_drop, axis = 1), # this also doesnt matter
            df_val= data_folds[0][1].drop(cols_drop, axis = 1),# this doesnt matter here, just a placeholder
            cols_std=cols_std,
            cols_leave=cols_leave,
            feature_offset= self.feature_offsets,
            mode= 1,
            preprocess_type= 'standardize')

        # Conversion NaN to 0
        self.x_train_complete = [torch.nan_to_num(x_view) for x_view in self.x_train_complete]


        # We check for each view how many 0s this view contains (which previously were NaN values)
        train_zeros = []
        test_zeros = []


        for c,view in enumerate(self.x_train_complete):
            curr_view_count_train = torch.count_nonzero(view)
            curr_view_count_test = torch.count_nonzero(self.x_test[c])
            train_zeros.append(curr_view_count_train)
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


        # Adjust data in case we delete something
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


        # Casting needed for neural nets
        for c in range(self.n_folds):
            # Cast all elements to torch.float32
            self.train_folds_durations[c] = torch.from_numpy(self.train_folds_durations[c]).to(torch.float32)
            self.train_folds_events[c] = torch.from_numpy(self.train_folds_events[c]).to(torch.float32)
            self.val_folds_durations[c] = torch.from_numpy(self.val_folds_durations[c]).to(torch.float32)
            self.val_folds_events[c] = torch.from_numpy(self.val_folds_events[c]).to(torch.float32)

        # Also cast train duration & events
        self.duration_test = torch.from_numpy(self.duration_test).to(torch.float32)
        self.event_test = torch.from_numpy(self.event_test).to(torch.float32)


        # For AE feature selection, we want to have the possibility of Hyperparameter Tuning : Thus we need to save our data
        # in a .csv file

        for c_fold in range(self.n_folds):
            all_train_data = copy.deepcopy(self.train_folds[c_fold])
            all_train_data.append(self.train_folds_durations[c_fold].unsqueeze(1))
            all_train_data.append(self.train_folds_events[c_fold].unsqueeze(1))

            train_data_c = torch.cat(tuple(all_train_data), dim=1)
            train_data_df = pd.DataFrame(train_data_c)
        #    dir = os.path.expanduser('~/SUMO/Project/ProcessedNotFeatSelectedData/TrainData.csv')
            dir = os.path.expanduser("/Users/marlon/Desktop/Project/ProcessedNotFeatSelectedData/TrainData.csv")
            train_data_df.to_csv(dir)

            all_val_data = copy.deepcopy(self.val_folds[c_fold])
            all_val_data.append(self.val_folds_durations[c_fold].unsqueeze(1))
            all_val_data.append(self.val_folds_events[c_fold].unsqueeze(1))

            val_data_c = torch.cat(tuple(all_val_data), dim=1)
            val_data_df = pd.DataFrame(val_data_c)
          #  dir = os.path.expanduser('~/SUMO/Project/ProcessedNotFeatSelectedData/ValData.csv')
            dir = os.path.expanduser("/Users/marlon/Desktop/Project/ProcessedNotFeatSelectedData/ValData.csv")
            val_data_df.to_csv(dir)

            # For Convenience, also load feature_offsets to this folder
            feat_offs_df = pd.DataFrame(self.feature_offsets)
         #   dir = os.path.expanduser('~/SUMO/Project/ProcessedNotFeatSelectedData/FeatOffs.csv')
            dir = os.path.expanduser("/Users/marlon/Desktop/Project/ProcessedNotFeatSelectedData/FeatOffs.csv")
            feat_offs_df.to_csv(dir)


        all_test_data = copy.deepcopy(self.x_test)
        all_test_data.append(self.duration_test.unsqueeze(1))
        all_test_data.append(self.event_test.unsqueeze(1))

        test_data_c = torch.cat(tuple(all_test_data), dim=1)
        test_data_df = pd.DataFrame(test_data_c)
      #  dir = os.path.expanduser('~/SUMO/Project/ProcessedNotFeatSelectedData/TestData.csv')
        dir = os.path.expanduser("/Users/marlon/Desktop/Project/ProcessedNotFeatSelectedData/TestData.csv")
        test_data_df.to_csv(dir)





        return n_train_fold_samples, n_val_fold_samples, n_test_samples, self.view_names


    def cross_validation(self,data_df, event_df, duration_df, k_folds):
        """
        Cross Validation, stratified based on event indicators
        :param data_df : Train data ; dtype : (pandas) Dataframe
        :param k_folds: Amount of folds to use ; dtype : Int
        :param event_df : event indicators (targets for stratification) ; dtype : Series
        :param duration_df : duration indicators ; dtype : Dataframe
        :return: folds : all training/validation folds ; dtype : List of Lists of Dataframes [n_samples, n_features]
                 folds_targets : event values for training/validation folds ; dtype : List of Lists of numpy arrays
                 folds_durations : duration values for training/validation folds ; dtype : List of Lists of numpy arrays

        Train/Validation splits, their targets (event values), their duration values
                 ; dtype : List of Numpy Arrays, List of Numpy Arrays, List of Numpy Arrays TODO :check
        """
        if k_folds == 1:  # KFold doesn't work with 1 Split, so we need a workaround
            skfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=42) # 5 so we have roughly 20% val split
        else:
            skfold = StratifiedKFold(n_splits=k_folds,shuffle=True,random_state=42)

        # save all folds in here, each sublist of type [train_fold_df, val_fold_df]
        folds = [[] for i in range(k_folds)]
        # save all targets (event indicators) here, each sublist of type [train_fold_targets_df, val_fold_targets_df]
        folds_targets = [[] for i in range(k_folds)]
        # Save folds durations here
        folds_durations = [[] for i in range(k_folds)]

        c = 0
        for train_idx, val_idx in skfold.split(data_df, event_df):


            if k_folds == 1 and c > 0: # Use just one split
                pass

            else:
                folds[c].append(data_df.iloc[train_idx])
                folds[c].append(data_df.iloc[val_idx])

                folds_targets[c].append(event_df.iloc[train_idx].values)
                folds_targets[c].append(event_df.iloc[val_idx].values)

                folds_durations[c].append(duration_df.iloc[train_idx].values)
                folds_durations[c].append(duration_df.iloc[val_idx].values)

            c += 1

        return folds, folds_targets, folds_durations, k_folds # TODO : k_folds muss denk ich nicht returned werden


    def feature_selection(self, method = None,
                          feature_names = None, # for PPI network
                          components = None, # for PCA
                          thresholds = None):
        """

        :param method: Feature selection method (Eigengenes/PCA/Variance/AE/PPI) ; dtype : String
        :param feature_names: Names of Features [needed for PPI-Network] ; dtype : List of Strings TODO: check
        :param components: Number of Components for each view [needed for PCA] ; dtype : List of Int
        :param thresholds: Threshold for each view [needed for Variance] ; dtype : List of  Float [between 0 and 1]
        :return: Train/Validation/Test Set with respective duration & event values after feature selection
                 ; dtype : List of Lists [for each fold] of Lists [for each view] of Tensors TODO :check if tensor of numpy array
        """


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if method.lower() == 'eigengenes':
            """Eigengene matrices Feature selection
               Based on : https://github.com/huangzhii/lmQCM
               Eigengene matrix is the expression value of each 
               gene co-expression module summarized into first PC using SVD.
               We calculate eigengene matrices for each view.
            """

            eigengene_train_tensors = []
            eigengene_val_tensors = []
            eigengene_test_tensors = []


            # Train/Val for each fold
            for c_fold,fold in enumerate(self.train_folds):
                for view in range(self.n_views):






                    eg_view = FeatureSelection.F_eigengene_matrices(data=self.train_folds[c_fold][view],
                                                                    mask=self.train_mask_folds[c_fold][view],
                                                                    view_name=self.view_names[view],
                                                                    duration=self.train_folds_durations[c_fold],
                                                                    event=self.train_folds_events[c_fold],
                                                                    stage= 'train',
                                                                    cancer_name= self.cancer_name)

                    eg_view_val = FeatureSelection.F_eigengene_matrices(data=self.val_folds[c_fold][view],
                                                                        mask=self.val_mask_folds[c_fold][view],
                                                                        view_name=self.view_names[view],
                                                                        duration=self.val_folds_durations[c_fold],
                                                                        event=self.val_folds_events[c_fold],
                                                                        stage= 'val',
                                                                        cancer_name= self.cancer_name)

                    if c_fold == 0: # as our training data has no multiple folds, we just do the eigengene matrix calculation for the first fold
                        eg_view_test = FeatureSelection.F_eigengene_matrices(data=self.x_test[view],
                                                                             mask=self.x_test_mask[view],
                                                                             view_name=self.view_names[view],
                                                                             duration=self.duration_test,
                                                                             event=self.event_test,
                                                                             stage='test',
                                                                             cancer_name= self.cancer_name)


                    eg_view.preprocess()
                    eg_view_val.preprocess()
                    if c_fold == 0:
                        eg_view_test.preprocess()

                # We also need to set the R program to only calculate the test eigengenes in the first fold
                # as the test set is the same across all folds
                if c_fold == 0:
                    # Train/Val/Test data
                    mode = "all"
                    with open('/Users/marlon/Desktop/Project/TCGAData/eigengene_mode.txt', 'w') as f:
                        f.write(mode)
                else:
                    # Train/Val data
                    mode = "folds"
                    with open('/Users/marlon/Desktop/Project/TCGAData/eigengene_mode.txt', 'w') as f:
                        f.write(mode)


                eg_view.eigengene_multiplication()
                # If the mode is folds, we'll return an empty list for the test matrices
                eigengene_matrices,eigengene_matrices_val, eigengene_matrices_test = eg_view.get_eigengene_matrices(self.view_names)

                # as list as each eigengene matrix is of a different size
                eigengene_matrices_tensors = []
                eigengene_matrices_tensors_val = []
                eigengene_matrices_tensors_test = []
                for x in range(self.n_views):
                    eigengene_matrices_tensors.append([])
                    eigengene_matrices_tensors_val.append([])
                    eigengene_matrices_tensors_test.append([])

                #Dataframe to tensor structure
                for c, view in enumerate(eigengene_matrices):
                    for x in range(len(view.index)):
                        temp = view.iloc[x, :].values.tolist()
                        eigengene_matrices_tensors[c].append(temp)
                    eigengene_matrices_tensors[c] = torch.tensor(eigengene_matrices_tensors[c])


                for c, view in enumerate(eigengene_matrices_val):
                    for x in range(len(view.index)):
                        temp = view.iloc[x, :].values.tolist()
                        eigengene_matrices_tensors_val[c].append(temp)
                    eigengene_matrices_tensors_val[c] = torch.tensor(eigengene_matrices_tensors_val[c])




                if c_fold == 0:
                    for c, view in enumerate(eigengene_matrices_test):
                        for x in range(len(view.index)):
                            temp = view.iloc[x, :].values.tolist()
                            eigengene_matrices_tensors_test[c].append(temp)
                        eigengene_matrices_tensors_test[c] = torch.tensor(eigengene_matrices_tensors_test[c])
                else:
                    # Already add the tensor from the first fold to our list of lists for each fold, so we
                    # dont get indexing problems in the next part
                    eigengene_matrices_tensors_test = eigengene_test_tensors[0].copy()
                    
                # save values : We still need to find the minimum number of eigengenes across
                # all folds for each view respectively and set all feature sizes to the minimum so we have the same
                # structural input for neural nets
                eigengene_train_tensors.append(eigengene_matrices_tensors)
                eigengene_val_tensors.append(eigengene_matrices_tensors_val)
                eigengene_test_tensors.append(eigengene_matrices_tensors_test)
            
            # Now find minimum

            for c_view in range(self.n_views):
                min_holder = []
                for c_fold in range(self.n_folds):
                    # Minimum over current fold and view
                    minimum = min(eigengene_train_tensors[c_fold][c_view].size(1),
                                  eigengene_val_tensors[c_fold][c_view].size(1),
                                  eigengene_test_tensors[c_fold][c_view].size(1))
                    min_holder.append(minimum)
                # Now we can access the minimum eigengene feature size for the current view across all folds
                actual_minimum = min(min_holder)
                # and resize

                for c_fold in range(self.n_folds):
                    eigengene_train_tensors[c_fold][c_view] = eigengene_train_tensors[c_fold][c_view][:,0:actual_minimum]
                    eigengene_val_tensors[c_fold][c_view] = eigengene_val_tensors[c_fold][c_view][:,0:actual_minimum]
                    eigengene_test_tensors[c_fold][c_view] = eigengene_test_tensors[c_fold][c_view][:,0:actual_minimum]


            return eigengene_train_tensors,eigengene_val_tensors,eigengene_test_tensors, \
                   self.train_folds_durations,self.train_folds_events, \
                   self.val_folds_durations,self.val_folds_events, \
                   self.duration_test,self.event_test






        if method.lower() == 'pca':
            """ Principal Component Analysis Feature Selection
                Selecting n components for each view individually.
            """
            PCA_train_tensors = [[] for i in range(len(self.train_folds))]
            PCA_val_tensors = [[] for i in range(len(self.train_folds))]
            PCA_test_tensors = [[] for i in range(len(self.train_folds))]
            feat_offs = [[] for i in range(len(self.train_folds))]




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


                feat_offs[c] = components






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




        if method.lower() == 'variance':
            """ 
            Variance based feature selection
            Removing low-variance features based on a threshold for each view individually.
            """
            variance_train_tensors = [[] for i in range(len(self.train_folds))]
            variance_val_tensors = [[] for i in range(len(self.train_folds))]
            variance_test_tensors = [[] for i in range(len(self.train_folds))]




            for c,fold in enumerate(self.train_folds):
                for view in range(self.n_views):
                    # Initialize Variance objects for both train and test with same components
                    while True:
                        try:
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
                        except ValueError:
                            thresholds[view] = thresholds[view] - 0.01
                        else:
                            print("Threshold of view ", self.view_names[view], "is ", thresholds[view])
                            break



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


        if method.lower() == 'none':

            test_folds = []

            # Needed for right structure
            for c,fold in enumerate(self.train_folds):
                test_folds.append(self.x_test)


            return self.train_folds, self.val_folds, test_folds,\
                   self.train_folds_durations,self.train_folds_events, \
                   self.val_folds_durations,self.val_folds_events, \
                   self.duration_test,self.event_test



        if method.lower() == 'ae':
            """
            Autoencoder Feature Selection. Train an AE for each view and store the bottleneck representation
            as selected features. This AE can be tuned with Optuna.
            """


            # Define learning rate
            learning_rate_train = 9.787680019790807e-05
         #   learning_rate_val = 0.0001
         #   learning_rate_test = 0.0001

            # Define Loss
            criterion = nn.MSELoss()

            # Define device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            #Define number of epochs
            n_epochs = 50


            # Batch Sizes
            batch_size_train = 8
          #  batch_size_val = 16
          #  batch_size_test = 16

            # Decide whether to print losses
            print_bool = True


            # List of lists to save output for each fold
            selected_train_features = [[] for _ in range(self.n_folds)]
            selected_val_features = [[] for _ in range(self.n_folds)]
            selected_test_features = [[] for _ in range(self.n_folds)]



            for c_fold in range(self.n_folds):
                dimensions_train = [x.shape[1] for x in self.train_folds[c_fold]]
          #      dimensions_val = [x.shape[1] for x in self.val_folds[c_fold]]
            #    dimensions_test = [x.shape[1] for x in self.x_test]

                # Layer Sizes
           #     layers_train = [[256,128] for x in range(self.n_views)]
                layers_train = [[236,159], [181,35], [290,128]]
             #   layers_val = [[512,256] for x in range(self.n_views)]
             #   layers_test = [[256,128] for x in range(self.n_views)]

                # Activation Functions
           #     activation_functions_train = [['relu'] for i in range(self.n_views)]
                activation_functions_train = [['relu','relu'],['relu','sigmoid'],['relu','sigmoid']]
            #    activation_functions_val = [['relu'] for i in range(self.n_views)]
            #    activation_functions_test = [['relu'] for i in range(self.n_views)]
                # Batch Normalization
             #   batch_normalization_train = [[] for i in range(self.n_views)]
                batch_normalization_train = [['yes','yes'],['no','yes'], ['yes','yes']]
            #    batch_normalization_val = [[] for i in range(self.n_views)]
            #    batch_normalization_test = [[] for i in range(self.n_views)]
                # Dropout Layers
            #    dropout_layers_train = [[] for i in range(self.n_views)]
                dropout_layers_train = [['yes','no'], ['yes','no'], ['no','no']]
            #    dropout_layers_val = [[] for i in range(self.n_views)]
            #    dropout_layers_test = [[] for i in range(self.n_views)]

            #    for c,dim in enumerate(layers_train):
            #        for i in range(len(dim)):
            #            batch_normalization_train[c].append('no')
               #         batch_normalization_val[c].append('no')
               #         batch_normalization_test[c].append('no')
            #            dropout_layers_train[c].append('no')
               #         dropout_layers_val[c].append('no')
               #         dropout_layers_test[c].append('no')


                # Call train net ; we use none as type, as we want to keep the structure of multiple views
                net_train = featAE.AE(self.view_names,
                                      dimensions_train, layers_train,
                                      activ_funcs= activation_functions_train,
                                      batch_norm= batch_normalization_train,
                                      dropout_layers= dropout_layers_train,
                                      dropout_prob= 0.4,
                                      dropout_bool=True,
                                      batch_norm_bool=False,
                                      type_ae='none',
                                      print_bool=False)

                # Call val net (same structure)
          #      net_val = featAE.AE(self.view_names,
          #                          dimensions_val, layers_val,
          #                          activ_funcs= activation_functions_val,
          #                          batch_norm= batch_normalization_val,
          #                          dropout_layers= dropout_layers_val,
          #                          dropout_prob= 0.1,
          #                          dropout_bool=False,
          #                          batch_norm_bool=False,
          #                          type_ae='none',
           #                         print_bool=False)


                # Call test net (same structure)
           #     net_test = featAE.AE(self.view_names,
           #                          dimensions_test, layers_test,
           #                          activ_funcs= activation_functions_test,
            #                         batch_norm= batch_normalization_test,
           #                          dropout_layers= dropout_layers_test,
           ##                          dropout_prob= 0.1,
            #                         dropout_bool=False,
           #                          batch_norm_bool=False,
             #                        type_ae='none',
             #                        print_bool=False)

                # optimizers
                optimizer_train = Adam(net_train.parameters(), lr= learning_rate_train)
             #   optimizer_val = Adam(net_val.parameters(), lr= learning_rate_val)
             #   optimizer_test = Adam(net_test.parameters(), lr= learning_rate_test)



                # Load Data for current fold with Dataloaders for batching structure
                self.train_set = MultiOmicsDataset(self.train_folds[c_fold], self.train_folds_durations[c_fold], self.train_folds_events[c_fold], type = 'tensor')
                self.val_set = MultiOmicsDataset(self.val_folds[c_fold], self.val_folds_durations[c_fold], self.val_folds_events[c_fold], type = 'tensor')
                self.test_set = MultiOmicsDataset(self.x_test, self.duration_test, self.event_test, type = 'tensor')

                # drop last false since we are in feature selection and don't want to lose data here
                ae_trainloader = DataLoader(self.train_set,batch_size=batch_size_train,shuffle=True,drop_last=False)
                ae_valloader = DataLoader(self.val_set, batch_size=batch_size_train, shuffle=True,drop_last=False)
                ae_testloader = DataLoader(self.test_set,batch_size=batch_size_train,shuffle=True,drop_last=False)


                # Train for current epoch
                for epoch in range(n_epochs):
                    loss_train = 0
                    loss_val = 0
                    loss_test = 0
                    ############################# TRAIN SET ##################################
                    for train_batch, train_duration, train_event in ae_trainloader:
                        # Send data to device if possible
                        for view in range(len(train_batch)):
                            train_batch[view] = train_batch[view]

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
                    if print_bool == True:
                        print("epoch : {}/{}, loss = {:.6f} for training data".format(epoch + 1, n_epochs, loss_train))


                    ###################### VALIDATION SET ##########################
                    for val_batch, val_duration, val_event in ae_valloader:

                        # Send data to device if possible
                        for view in range(len(train_batch)):
                            val_batch[view] = val_batch[view]

                        # Structure must be a tuple
                        val_batch = tuple(val_batch)

                        optimizer_train.zero_grad()

                        data_middle, final_out, input_data_raw = net_train(*val_batch)

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
                    if print_bool == True:
                        print("epoch : {}/{}, loss = {:.6f} for validation data".format(epoch + 1, n_epochs, loss_val))



                    ######################### TEST SET ##########################
                    for test_batch, test_duration, test_event in ae_testloader:
                        # Send data to device if possible
                        for view in range(len(test_batch)):
                            test_batch[view] = test_batch[view]

                        # Structure must be a tuple
                        test_batch = tuple(test_batch)

                        optimizer_train.zero_grad()

                        data_middle, final_out, input_data_raw = net_train(*test_batch)

                        if epoch == n_epochs - 1:
                            selected_test_features[c_fold].append(data_middle)

                        test_loss = 0
                        for view in range(len(test_batch)):
                            test_loss += criterion(input_data_raw[view], final_out[view])

                        test_loss.backward()

                        optimizer_train.step()

                        loss_test = test_loss.item()


                    loss_test = loss_test / len(ae_testloader)
                    if print_bool == True:
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






        if method.lower() == 'ppi':
            """Protein-Protein Feature Selection"""

            # simpler structure, bc we just have folds, but no different views anymore
            ppi_train_tensors = []
            ppi_val_tensors = []
            ppi_test_tensors = []




            for c_fold in range(self.n_folds):
                ppi_train = FeatureSelection.PPI(self.train_folds[c_fold], feature_names, self.view_names)
                ppi_val = FeatureSelection.PPI(self.val_folds[c_fold], feature_names, self.view_names)

                print("Getting PPI train matrices for fold : {}".format(c_fold + 1))
                data_train,edge_index_train, proteins_used_train = ppi_train.get_matrices()
                print("Getting PPI validation matrices for fold : {}".format(c_fold + 1))
                data_val, edge_index_val, proteins_used_val = ppi_val.get_matrices()

                if c_fold == 0: # only need to get test set once
                    print("Getting PPI test matrices")
                    ppi_test = FeatureSelection.PPI(self.x_test, feature_names, self.view_names)
                    data_test, edge_index_test, proteins_used_test = ppi_test.get_matrices()


                # edge indexes and used proteins are to be the same amongst train/val/test
                edge_index = edge_index_train
                proteins_used = proteins_used_train

                ppi_train_tensors.append(data_train)
                ppi_val_tensors.append(data_val)
                ppi_test_tensors.append(data_test)


            return edge_index, proteins_used, ppi_train_tensors,ppi_val_tensors,ppi_test_tensors, \
                   self.train_folds_durations,self.train_folds_events, \
                   self.val_folds_durations,self.val_folds_events, \
                   self.duration_test,self.event_test





    def train_dataloader(self, batch_size):
        """
        Build training dataloader
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
        """
        test_loader = DataLoader(
            dataset=self.test_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=10)
        return test_loader






def objective(trial):
    """
    Optuna Optimization for Hyperparameters.
    :param trial: Settings of the current trial of Hyperparameters
    :return: MSELoss (sum of train- & val- & test loss) ; dtype : Float
    """
    # Load in data (##### For testing for first fold, later on

    # List of lists to save output for each fold
    selected_train_features = []
    selected_val_features = []
    selected_test_features = []
    # Define Loss
    criterion = nn.MSELoss()

    trainset, valset,testset, featoffs = load_data()

    featoffs = list(featoffs.values)
    for idx,_ in enumerate(featoffs):
        featoffs[idx] = featoffs[idx].item()


    train_data = []
    val_data = []
    test_data = []
    for c,feat in enumerate(featoffs):
        if c < len(featoffs) - 3: # train data views
            train_data.append(np.array((trainset.iloc[:, featoffs[c] : featoffs[c+1]]).values).astype('float32'))
            val_data.append(np.array((valset.iloc[:, featoffs[c]: featoffs[c + 1]]).values).astype('float32'))
            test_data.append(np.array((testset.iloc[:, featoffs[c]: featoffs[c + 1]]).values).astype('float32'))
        elif c == len(featoffs) - 3: # duration
            train_duration = (np.array((trainset.iloc[:, featoffs[c] : featoffs[c+1]]).values).astype('float32')).squeeze(axis=1)
            val_duration = (np.array((valset.iloc[:, featoffs[c]: featoffs[c + 1]]).values).astype('float32')).squeeze(axis=1)
            test_duration = (np.array((testset.iloc[:, featoffs[c]: featoffs[c + 1]]).values).astype('float32')).squeeze(axis=1)
        elif c == len(featoffs) -2: # event
            train_event = (np.array((trainset.iloc[:, featoffs[c] : featoffs[c+1]]).values).astype('float32')).squeeze(axis=1)
            val_event = (np.array((valset.iloc[:, featoffs[c]: featoffs[c + 1]]).values).astype('float32')).squeeze(axis=1)
            test_event = (np.array((testset.iloc[:, featoffs[c]: featoffs[c + 1]]).values).astype('float32')).squeeze(axis=1)





    views = []
    read_in = open('/Users/marlon/Desktop/Project/TCGAData/cancerviews.txt', 'r')
    for view in read_in:
        views.append(view)

    view_names = [line[:-1] for line in views]


    dimensions_train = [x.shape[1] for x in train_data]
    dimensions_val = [x.shape[1] for x in val_data]
    dimensions_test = [x.shape[1] for x in test_data]

    assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/test'

    dimensions = dimensions_train

    ####################################### DEFINE HYPERPARAMETERS #######################################
    l2_regularization_bool = trial.suggest_categorical('l2_regularization_bool', [True,False])
    learning_rate = trial.suggest_float("learning_rate", 1e-5,1e-1,log=True)
    l2_regularization_rate = trial.suggest_float("l2_regularization_rate", 1e-6,1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8,16,32,64,128,256])
   # n_epochs = trial.suggest_int("n_epochs", 10,100) ########### TESTING
    n_epochs = 30
    dropout_prob = trial.suggest_float("dropout_prob", 0,0.5,step=0.1)
    dropout_bool = trial.suggest_categorical('dropout_bool', [True,False])
    batchnorm_bool = trial.suggest_categorical('batchnorm_bool',[True,False])

    layers = []
    activation_functions = []
    dropouts = []
    batchnorms = []

    if 'MRNA' in view_names:
        layers_1_mRNA = trial.suggest_int('layers_1_mRNA', 1000, 2000)
        layers_2_mRNA = trial.suggest_int('layers_2_mRNA', 500, 1000)
        layers_1_mRNA_activfunc = trial.suggest_categorical('layers_1_mRNA_activfunc', ['relu','sigmoid'])
        layers_2_mRNA_activfunc = trial.suggest_categorical('layers_2_mRNA_activfunc', ['relu','sigmoid'])
        layers_1_mRNA_dropout = trial.suggest_categorical('layers_1_mRNA_dropout', ['yes','no'])
        layers_2_mRNA_dropout = trial.suggest_categorical('layers_2_mRNA_dropout', ['yes','no'])
        layers_1_mRNA_batchnorm = trial.suggest_categorical('layers_1_mRNA_batchnorm', ['yes', 'no'])
        layers_2_mRNA_batchnorm = trial.suggest_categorical('layers_2_mRNA_batchnorm', ['yes', 'no'])

        layers.append([layers_1_mRNA,layers_2_mRNA])
        activation_functions.append([layers_1_mRNA_activfunc, layers_2_mRNA_activfunc])
        dropouts.append([layers_1_mRNA_dropout, layers_2_mRNA_dropout])
        batchnorms.append([layers_1_mRNA_batchnorm, layers_2_mRNA_batchnorm])


    if 'DNA' in view_names:
        layers_1_DNA = trial.suggest_int('layers_1_DNA', 1000, 2000)
        layers_2_DNA = trial.suggest_int('layers_2_DNA', 500,1000)
        layers_1_DNA_activfunc = trial.suggest_categorical('layers_1_DNA_activfunc', ['relu','sigmoid'])
        layers_2_DNA_activfunc = trial.suggest_categorical('layers_2_DNA_activfunc', ['relu','sigmoid'])
        layers_1_DNA_dropout = trial.suggest_categorical('layers_1_DNA_dropout', ['yes','no'])
        layers_2_DNA_dropout = trial.suggest_categorical('layers_2_DNA_dropout', ['yes','no'])
        layers_1_DNA_batchnorm = trial.suggest_categorical('layers_1_DNA_batchnorm', ['yes', 'no'])
        layers_2_DNA_batchnorm = trial.suggest_categorical('layers_2_DNA_batchnorm', ['yes', 'no'])

        layers.append([layers_1_DNA,layers_2_DNA])
        activation_functions.append([layers_1_DNA_activfunc, layers_2_DNA_activfunc])
        dropouts.append([layers_1_DNA_dropout, layers_2_DNA_dropout])
        batchnorms.append([layers_1_DNA_batchnorm, layers_2_DNA_batchnorm])

    if 'MICRORNA' in view_names:
        layers_1_microRNA = trial.suggest_int('layers_1_microRNA', 1000, 2000)
        layers_2_microRNA = trial.suggest_int('layers_2_microRNA', 500, 1000)
        layers_1_microRNA_activfunc = trial.suggest_categorical('layers_1_microRNA_activfunc', ['relu','sigmoid'])
        layers_2_microRNA_activfunc = trial.suggest_categorical('layers_2_microRNA_activfunc', ['relu','sigmoid'])
        layers_1_microRNA_dropout = trial.suggest_categorical('layers_1_microRNA_dropout', ['yes','no'])
        layers_2_microRNA_dropout = trial.suggest_categorical('layers_2_microRNA_dropout', ['yes','no'])
        layers_1_microRNA_batchnorm = trial.suggest_categorical('layers_1_microRNA_batchnorm', ['yes', 'no'])
        layers_2_microRNA_batchnorm = trial.suggest_categorical('layers_2_microRNA_batchnorm', ['yes', 'no'])


        layers.append([layers_1_microRNA,layers_2_microRNA])
        activation_functions.append([layers_1_microRNA_activfunc, layers_2_microRNA_activfunc])
        dropouts.append([layers_1_microRNA_dropout, layers_2_microRNA_dropout])
        batchnorms.append([layers_1_microRNA_batchnorm, layers_2_microRNA_batchnorm])

    if 'RPPA' in view_names:
        layers_1_RPPA = trial.suggest_int('layers_1_RPPA', 1000, 2000)
        layers_2_RPPA = trial.suggest_int('layers_2_RPPA', 500, 1000)
        layers_1_RPPA_activfunc = trial.suggest_categorical('layers_1_RPPA_activfunc', ['relu','sigmoid'])
        layers_2_RPPA_activfunc = trial.suggest_categorical('layers_2_RPPA_activfunc', ['relu','sigmoid'])
        layers_1_RPPA_dropout = trial.suggest_categorical('layers_1_RPPA_dropout', ['yes','no'])
        layers_2_RPPA_dropout = trial.suggest_categorical('layers_2_RPPA_dropout', ['yes','no'])
        layers_1_RPPA_batchnorm = trial.suggest_categorical('layers_1_RPPA_batchnorm', ['yes', 'no'])
        layers_2_RPPA_batchnorm = trial.suggest_categorical('layers_2_RPPA_batchnorm', ['yes', 'no'])


        layers.append([layers_1_RPPA,layers_2_RPPA])
        activation_functions.append([layers_1_RPPA_activfunc, layers_2_RPPA_activfunc])
        dropouts.append([layers_1_RPPA_dropout, layers_2_RPPA_dropout])
        batchnorms.append([layers_1_RPPA_batchnorm, layers_2_RPPA_batchnorm])

    layers_train = copy.deepcopy(layers)
  #  layers_val = copy.deepcopy(layers)
  #  layers_test = copy.deepcopy(layers)
    activ_func_train = copy.deepcopy(activation_functions)
  #  activ_func_val = copy.deepcopy(activation_functions)
  #  activ_func_test = copy.deepcopy(activation_functions)
    batchnorms_train = copy.deepcopy(layers)
  #  batchnorms_val = copy.deepcopy(layers)
  #  batchnorms_test = copy.deepcopy(layers)
    dropouts_train = copy.deepcopy(layers)
  #  dropouts_val = copy.deepcopy(layers)
  #  dropouts_test = copy.deepcopy(layers)

    # Call train net
    net_train = featAE.AE(view_names,
                          dimensions, layers_train,
                          activ_funcs= activ_func_train,
                          batch_norm= batchnorms_train,
                          dropout_layers= dropouts_train,
                          dropout_prob= dropout_prob,
                          dropout_bool=dropout_bool,
                          batch_norm_bool=batchnorm_bool,
                          type_ae='none',
                          print_bool=False)

    # Call val net (same structure)
 #   net_val = featAE.AE(view_names,
 #                         dimensions, layers_val,
 #                         activ_funcs= activ_func_val,
 #                         batch_norm= batchnorms_val,
 #                         dropout_layers= dropouts_val,
 #                         dropout_prob= dropout_prob,
 #                         dropout_bool=dropout_bool,
 #                         batch_norm_bool=batchnorm_bool,
 #                         type_ae='none',
 #                         print_bool=False)


    # Call test net (same structure)
#    net_test = featAE.AE(view_names,
#                          dimensions, layers_test,
#                          activ_funcs= activ_func_test,
#                          batch_norm= batchnorms_test,
#                          dropout_layers= dropouts_test,
#                          dropout_prob= dropout_prob,
#                          dropout_bool=dropout_bool,
#                          batch_norm_bool=batchnorm_bool,
#                          type_ae='none',
#                          print_bool=False)

    # optimizers

    if l2_regularization_bool == True:
        optimizer_train = Adam(net_train.parameters(), lr= learning_rate,weight_decay= l2_regularization_rate)
  #      optimizer_val = Adam(net_val.parameters(), lr= learning_rate, weight_decay= l2_regularization_rate)
   #     optimizer_test = Adam(net_test.parameters(), lr= learning_rate, weight_decay= l2_regularization_rate)

    else:
        optimizer_train = Adam(net_train.parameters(), lr= learning_rate)
   #     optimizer_val = Adam(net_val.parameters(), lr= learning_rate)
   #     optimizer_test = Adam(net_test.parameters(), lr= learning_rate)





    # Load Data for current fold with Dataloaders for batching structure
    train_set = MultiOmicsDataset(train_data,train_duration, train_event, type = 'np')
    val_set = MultiOmicsDataset(val_data, val_duration, val_event, type = 'np')
    test_set = MultiOmicsDataset(test_data, test_duration, test_event, type = 'np')



    # drop last false since we are in feature selection and don't want to lose data here
    ae_trainloader = DataLoader(train_set,batch_size=batch_size,shuffle=True,drop_last=False)
    ae_valloader = DataLoader(val_set, batch_size=batch_size, shuffle=True,drop_last=False)
    ae_testloader = DataLoader(test_set,batch_size=batch_size,shuffle=True,drop_last=False)


    # Train for current epoch
    for epoch in range(n_epochs):
        loss_train = 0
        loss_val = 0
        loss_test = 0
        ############################# TRAIN SET ##################################
        for train_batch, duration, event in ae_trainloader:
            # Send data to device if possible
            for view in range(len(train_batch)):
                train_batch[view] = train_batch[view]

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


        ###################### VALIDATION SET ##########################
        for val_batch, duration, event in ae_valloader:

            # Send data to device if possible
            for view in range(len(train_batch)):
                val_batch[view] = val_batch[view]

            # Structure must be a tuple
            val_batch = tuple(val_batch)

            optimizer_train.zero_grad()

            data_middle, final_out, input_data_raw = net_train(*val_batch)

            # If we're in the last epoch, we save our data in the middle (our selected features)
            if epoch == n_epochs - 1:
                selected_val_features.append(data_middle)

            val_loss = 0
            for view in range(len(val_batch)):
                val_loss += criterion(input_data_raw[view], final_out[view])

            val_loss.backward()

            optimizer_train.step()

            loss_val += val_loss.item()


        loss_val = loss_val / len(ae_valloader)

        print("epoch : {}/{}, loss = {:.6f} for validation data".format(epoch + 1, n_epochs, loss_val))



        ######################### TEST SET ##########################
        for test_batch, duration, event in ae_testloader:
            # Send data to device if possible
            for view in range(len(test_batch)):
                test_batch[view] = test_batch[view]

            # Structure must be a tuple
            test_batch = tuple(test_batch)

            optimizer_train.zero_grad()

            data_middle, final_out, input_data_raw = net_train(*test_batch)

            if epoch == n_epochs - 1:
                selected_test_features.append(data_middle)

            test_loss = 0
            for view in range(len(test_batch)):
                test_loss += criterion(input_data_raw[view], final_out[view])

            test_loss.backward()

            optimizer_train.step()

            loss_test = test_loss.item()


        loss_test = loss_test / len(ae_testloader)

        print("epoch : {}/{}, loss = {:.6f} for test data".format(epoch + 1, n_epochs, loss_test))

    full_loss = loss_train + loss_val + loss_test

    return full_loss


def load_data(data_dir="/Users/marlon/Desktop/Project/ProcessedNotFeatSelectedData/"):
    """
    Function to load data. Needed for Optuna Optimization.
    :param data_dir: Directory in which data is stored.
    :return: data and feature offsets (for feature values, duration and event)
    """
    trainset = pd.read_csv(
        os.path.join(data_dir + "TrainData.csv"), index_col=0)

    valset = pd.read_csv(
        os.path.join(data_dir + "ValData.csv"), index_col=0)

    testset = pd.read_csv(
        os.path.join(data_dir +  "TestData.csv"), index_col=0)

    feat_offs = pd.read_csv(
        os.path.join(data_dir + "FeatOffs.csv"), index_col=0)


    return trainset, valset, testset, feat_offs


def optuna_optimization(fold = 1):
    """
    Optuna Optimization for Hyperparameters.
    """


    EPOCHS = 150
    study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials = EPOCHS)

    trial = study.best_trial

    print("Lowest MSE-Loss", trial.value)
    print("Best Hyperparamters : {}".format(trial.params))

