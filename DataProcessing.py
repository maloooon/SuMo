import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
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
from sys import exit



class MultiOmicsDataset(Dataset):

    def __init__(self, X, duration, event, type= 'tensor'):
        """

        :param X: Data input ; dtype : List of tensors or ndarrays of floats (n_samples,n_features) , one for each view
        :param duration: duration (time-to-event or time-to-censorship) ; dtype : Tensor/ndarray of Int (n_samples,1 [duration value])
        :param event: event (1 : not censored, 0 : censored) ; dtype : Tensor/ndarray of Int (n_samples, 1 [event value])
        :param type: Type of the data input X (tensors or ndarrays) ; dtype : String ['tensor', 'np']
        """

        self.type = type
        self.duration = duration
        self.event = event

        if self.type == 'tensor':
            self.n_views = len(X)
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
        :return: data for each view ; dtype : List of Tensors/ndarrays
                 duration value ; dtype : Tensor/ndarray
                 event value ; dtype : Tensor/ndarray
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
    Preprocessing data.

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
                 ; dtype : String
    :param preprocess_type : Type of Preprocessing (Standardization/Normalization/None) ; dtype : String
    :return: Feature values ordered by views for train/validation/test ;
             dtype : List of Tensors (n_samples, n_features) [for each view]
    """


    if cols_std is not None and preprocess_type.lower() != 'none':

        if preprocess_type.lower() == 'standardize':
            standardize = [([col], StandardScaler()) for col in cols_std]
        if preprocess_type.lower() == 'minmax':
            standardize = [([col], MinMaxScaler()) for col in cols_std]
        if preprocess_type.lower() == 'robust':
            standardize = [([col], RobustScaler()) for col in cols_std]
        if preprocess_type.lower() == 'maxabs':
            standardize = [([col], MaxAbsScaler()) for col in cols_std]
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

        if mode == 'test_preprocess':
            x_test = mapper.transform(df_test).astype(np.float32)
            x_test_df = pd.DataFrame(x_test)

        for x in range(len(feature_offset) - 3): # -3 bc we don't have duration/event in training tensor
            x_train_ordered_by_view.append(torch.tensor((x_train_df.iloc[:, feature_offset[x]:
                                                                            feature_offset[x + 1]]).values))

            x_val_ordered_by_view.append(torch.tensor((x_val_df.iloc[:, feature_offset[x]:
                                                                        feature_offset[x + 1]]).values))
            if mode == 'test_preprocess':
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
            if mode == 'test_preprocess':
                x_test_ordered_by_view.append(torch.tensor((df_test.iloc[:, feature_offset[x]:
                                                                            feature_offset[x + 1]]).values))


    return x_train_ordered_by_view, x_test_ordered_by_view,x_val_ordered_by_view


class SurvMultiOmicsDataModule(pl.LightningDataModule):

    def __init__(
            self,
            df,
            feature_offsets,
            view_names,
            cancer_name=None,
            which_views = [],
            n_folds = 2,
            type_preprocess = 'standardize',
            save_folds = False,
            folds_folder_name = None,
            saved_folds_processing = False,
            direc_set = 'SUMO'):
        """
        :param df: Complete data (feature values each view, duration, event)
                   ; dtype : (pandas) Dataframe (n_samples, (n_features,1 [duration], 1 [event]))
        :param feature_offsets: Feature Offsets for different views aswell as duration & event [List ends with [..,x,x+1,x+2]
                               [0, n_feats_view_1, n_feats_view_1 + n_feats_view_2,..]
                   ; dtype : List of Int [Cumulative Sum]
        :param view_names: Names of all views ; dtype : List of Strings
        :param cancer_name: Name of current looked at cancer ; dtype : String
        :param which_views: Name of views we currently want to analyze ; if empty, all possible views are taken
                   ; dtype : List of Strings
        :param n_folds: Number of Folds for Cross-Validation ; dtype : Int
        :param type_preprocess: Type of Preprocessing (Standardization/Normalization/None) ; dtype : String
        :param save_folds : Decide whether folds should be saved ; dtype : Boolean
        :param folds_folder_name : Name for the folder if folds should be saved ; dtype : String
        :param saved_folds_processing : Decide whether saved folds should be processed ; dtype : Boolean
        :param direc_set : Home folder to load from ; dtype : String
        """
        super().__init__()
        self.df = df
        self.feature_offsets = feature_offsets # cumulative sum of features in list of features
        self.view_names = [x.upper() for x in view_names]
        self.n_views = len(view_names)
        self.cancer_name = cancer_name
        self.which_views = [x.upper() for x in which_views] # Decide which views to use for survival analysis
        self.n_folds = n_folds
        self.save_folds = save_folds
        self.folds_folder_name = folds_folder_name
        self.type_preprocess = type_preprocess
        self.saved_folds_processing = saved_folds_processing
        self.direc_set = direc_set # dir is Desktop for own CPU or SUMO for GPU

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
                # Remove duplicates
                self.feature_offsets = set(self.feature_offsets)
                # Return to right structure
                self.feature_offsets = sorted(list(self.feature_offsets))


        print("There are {} views : {} with feature offsets {}".format(self.n_views, self.view_names, self.feature_offsets))

        # Columns (features) which have only NaN values
        cols_remove = self.df.columns[self.df.isna().all()].tolist()
        print("Deleting features with only 0 values...")
        for feature_name in cols_remove:
            if 'DNA' in feature_name.upper():
                idx = self.view_names.index('DNA')
                # Remove one entity from feature offsets of respective view, as we will delete this feature from this view
                for c,_ in enumerate(self.feature_offsets):
                    if c >= (idx + 1):
                        self.feature_offsets[c] -= 1
            if 'MRNA' in feature_name.upper():
                idx = self.view_names.index('MRNA')
                for c,_ in enumerate(self.feature_offsets):
                    if c >= (idx + 1):
                        self.feature_offsets[c] -= 1
            if 'MIRNA' in feature_name.upper():
                idx = self.view_names.index('MICRORNA')
                for c,_ in enumerate(self.feature_offsets):
                    if c >= (idx + 1):
                        self.feature_offsets[c] -= 1
            if 'RPPA' in feature_name.upper():
                idx = self.view_names.index('RPPA')
                for c,_ in enumerate(self.feature_offsets):
                    if c >= (idx + 1):
                        self.feature_offsets[c] -= 1

        # Drop "empty" feature columns
        self.df.drop(cols_remove, inplace=True, axis=1)


        # Check if any view has no data anymore
        for c_offset in range(len(self.feature_offsets) -2):
            if self.feature_offsets[c_offset] == self.feature_offsets[c_offset + 1]:
                print("View", self.view_names[c_offset], "consists of only 0 values and thus won't be taken"
                                                         "into consideration for analysis.")

                # Delete from views :
                del self.view_names[c_offset]
                self.n_views -= 1
                # Delete from offsets
                del self.feature_offsets[c_offset]

        print("After Deletion, we have {} views left : {} and feature offsets {}".format(self.n_views, self.view_names, self.feature_offsets))
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


        if cols_leave is None:
            cols_leave = []

        if 'event' in cols_leave:
            cols_leave.remove('event')


        # Columns we don't want to standardize
        cols_survival = [col_duration, col_event]
        cols_drop = cols_survival


        # Features with numeric values
        if cols_std is None:
            cols_std = [
                col for col in self.df.columns if col not in cols_leave + cols_drop + cols_remove
            ]


        data_folds, data_folds_targets, data_folds_durations = self.cross_validation(df_train_temp,
                                                                                     self.event_train_df,
                                                                                     self.duration_train_df,
                                                                                     self.n_folds)



        if self.save_folds == True:
            # Main directory
            dir_main = '~/{}/Project/FoldsNew/{}'.format(self.direc_set,self.folds_folder_name)
            for fold in range(self.n_folds):

                dir = os.path.expanduser("{}/TrainFold_{}.csv".format(dir_main,fold))
                data_folds[fold][0].to_csv(dir)
                dir = os.path.expanduser("{}/ValFold_{}.csv".format(dir_main,fold))
                data_folds[fold][1].to_csv(dir)
                dir = os.path.expanduser("{}/TrainFoldEvent_{}.csv".format(dir_main,fold))
                np.savetxt(dir, data_folds_targets[fold][0], delimiter=",")
                dir = os.path.expanduser("{}/ValFoldEvent_{}.csv".format(dir_main,fold))
                np.savetxt(dir,data_folds_targets[fold][1],delimiter=",")
                dir = os.path.expanduser("{}/TrainFoldDuration_{}.csv".format(dir_main,fold))
                np.savetxt(dir,data_folds_durations[fold][0],delimiter=",")
                dir = os.path.expanduser("{}/ValFoldDuration_{}.csv".format(dir_main,fold))
                np.savetxt(dir,data_folds_durations[fold][1],delimiter=",")

            dir =os.path.expanduser("{}/Testset.csv".format(dir_main))
            df_test.to_csv(dir)
            dir = os.path.expanduser(r'{}/cols_std.txt'.format(dir_main))
            with open(dir, 'w') as fp:
                for item in cols_std:
                    #write each item on a new line
                    fp.write("%s\n" % item)
            dir = os.path.expanduser("{}/Traintemp.csv".format(dir_main))
            df_train_temp.to_csv(dir)


            dir = os.path.expanduser(r'{}/cols_remove.txt'.format(dir_main))
            with open(dir, 'w') as fp:
                for item in cols_remove:
                    #write each item on a new line
                    fp.write("%s\n" % item)

            print("FOLD SAVING DONE")
            exit()





        if self.saved_folds_processing == True:

            # Load saved folds
            data_folds = [[] for x in range(self.n_folds)]
            data_folds_targets = [[] for x in range(self.n_folds)]
            data_folds_durations = [[] for x in range(self.n_folds)]

            # Main directory
            dir_main = '~/{}/Project/FoldsNew/{}'.format(self.direc_set,self.folds_folder_name)

            for fold in range(self.n_folds):
                dir = os.path.expanduser("{}/TrainFold_{}.csv".format(dir_main,fold))
                data = pd.read_csv(dir)
                data.drop(columns=data.columns[0], axis=1, inplace=True)
                data_folds[fold].append(data)
                dir = os.path.expanduser("{}/ValFold_{}.csv".format(dir_main,fold))
                data = pd.read_csv(dir)
                data.drop(columns=data.columns[0], axis=1, inplace=True)
                data_folds[fold].append(data)
                dir = os.path.expanduser("{}/TrainFoldEvent_{}.csv".format(dir_main,fold))
                data = pd.read_csv(dir, header=None)
                data = data.to_numpy().squeeze(axis=1)
                data_folds_targets[fold].append(data)
                dir = os.path.expanduser("{}/ValFoldEvent_{}.csv".format(dir_main,fold))
                data = pd.read_csv(dir, header=None)
                data = data.to_numpy().squeeze(axis=1)
                data_folds_targets[fold].append(data)
                dir = os.path.expanduser("{}/TrainFoldDuration_{}.csv".format(dir_main,fold))
                data = pd.read_csv(dir, header=None)
                data = data.to_numpy().squeeze(axis=1)
                data_folds_durations[fold].append(data)
                dir = os.path.expanduser("{}/ValFoldDuration_{}.csv".format(dir_main,fold))
                data = pd.read_csv(dir, header=None)
                data = data.to_numpy().squeeze(axis=1)
                data_folds_durations[fold].append(data)

            dir =os.path.expanduser("{}/Testset.csv".format(dir_main))
            df_test = pd.read_csv(dir)
            df_test.drop(columns=df_test.columns[0], axis=1, inplace=True)
            dir = os.path.expanduser("{}/Traintemp.csv".format(dir_main))
            df_train_temp = pd.read_csv(dir)
            df_train_temp.drop(columns=df_train_temp.columns[0], axis=1, inplace=True)
            cols_std = []
            # Open file and read the content in a list
            dir = os.path.expanduser(r'{}/cols_std.txt'.format(dir_main))
            with open(dir, 'r') as fp:
                for line in fp:
                    # Remove linebreak from a current name
                    # Linebreak is the last character of each line
                    x = line[:-1]

                    # Add current item to the list
                    cols_std.append(x)

            # Columns we don't want to standardize
            cols_survival = [col_duration, col_event]
            cols_drop = cols_survival

            n_train_samples = df_train_temp.shape[0]






        n_train_fold_samples = []
        n_val_fold_samples = []

        print("Cross validation : {} splits".format(self.n_folds))
        for fold in range(self.n_folds):
            print("Split {} : ".format(fold + 1))
            print("non censored events in train : {} with {} samples in total".
                  format(int(np.sum(data_folds_targets[fold][0])), data_folds_targets[fold][0].size))
            print("non censored events in validation : {} with {} samples in total".
                  format(int(np.sum(data_folds_targets[fold][1])), data_folds_targets[fold][1].size))
            n_train_fold_samples.append(data_folds_targets[fold][0].size)
            n_val_fold_samples.append(data_folds_targets[fold][1].size)


        self.duration_test =(df_test[col_duration].values)
        self.event_test = (df_test[col_event].values)



        self.train_folds_events = [x[0] for x in data_folds_targets]
        self.val_folds_events = [x[1] for x in data_folds_targets]
        self.train_folds_durations = [x[0] for x in data_folds_durations]
        self.val_folds_durations = [x[1] for x in data_folds_durations]
        self.train_folds = []
        self.val_folds = []





        # Preprocess train, validation data
        if self.type_preprocess.lower() != 'none':
            print("Preprocessing data....")
        for fold in range(self.n_folds):
            print("Fold {} done".format(fold))
            self.x_train, self.x_test_actual, self.x_val = preprocess_features(
                df_train=data_folds[fold][0].drop(cols_drop, axis = 1), # Drop duration/event from df, as we don't want these
                df_test=df_test.drop(cols_drop, axis = 1),
                df_val= data_folds[fold][1].drop(cols_drop, axis = 1),
                cols_std=cols_std,
                cols_leave=cols_leave,
                feature_offset= self.feature_offsets,
                mode= fold,
                preprocess_type= self.type_preprocess
            )

            self.train_folds.append(self.x_train)
            self.val_folds.append(self.x_val)


        #       if fold == 0:
        #           self.x_test = self.x_test_actual

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


        # Until now, we scaled only train & validation folds. We also need to scale test data, but we
        # do this on the whole train set :
        self.x_train_complete, self.x_test_actual, temp2 = preprocess_features(
            df_train=df_train_temp.drop(cols_drop, axis = 1),
            df_test=df_test.drop(cols_drop, axis = 1),
            df_val= data_folds[0][1].drop(cols_drop, axis = 1),
            cols_std=cols_std,
            cols_leave=cols_leave,
            feature_offset= self.feature_offsets,
            mode= 'test_preprocess',
            preprocess_type= self.type_preprocess)

        self.x_test = self.x_test_actual

        self.x_test = [torch.nan_to_num(x_view) for x_view in self.x_test]
        self.x_test_mask = [torch.isnan(x_view) for x_view in self.x_test]


        # We use this function simply to get data into the right structure to check for NaN values ;
        # We don't preprocess here, as we are just interested in which values are NaN in the raw dataset
        self.x_train_complete, temp, temp2 = preprocess_features(
            df_train=df_train_temp.drop(cols_drop, axis = 1),
            df_test=df_test.drop(cols_drop, axis = 1),
            df_val= data_folds[0][1].drop(cols_drop, axis = 1),
            cols_std=cols_std,
            cols_leave=cols_leave,
            feature_offset= self.feature_offsets,
            mode= 1,
            preprocess_type= 'none')

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
            if train_zeros[x] < int(0.1 * ((self.feature_offsets[x + 1] - self.feature_offsets[x]) * n_train_samples)) \
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



        ######################################## STORE PREPROCESSED DATA ##########################################
        # Save preprocessed data
        dir_main = '~/{}/Project/ProcessedNotFeatSelectedData/{}/{}'.format(self.direc_set,self.type_preprocess,self.folds_folder_name)
        for c_fold in range(self.n_folds):
            all_train_data = copy.deepcopy(self.train_folds[c_fold])
            all_train_data.append(self.train_folds_durations[c_fold].unsqueeze(1))
            all_train_data.append(self.train_folds_events[c_fold].unsqueeze(1))

            train_data_c = torch.cat(tuple(all_train_data), dim=1)
            train_data_df = pd.DataFrame(train_data_c)
            dir = os.path.expanduser('{}/TrainData_{}.csv'.format(dir_main,c_fold))
            train_data_df.to_csv(dir)

            all_val_data = copy.deepcopy(self.val_folds[c_fold])
            all_val_data.append(self.val_folds_durations[c_fold].unsqueeze(1))
            all_val_data.append(self.val_folds_events[c_fold].unsqueeze(1))

            val_data_c = torch.cat(tuple(all_val_data), dim=1)
            val_data_df = pd.DataFrame(val_data_c)
            dir = os.path.expanduser('{}/ValData_{}.csv'.format(dir_main,c_fold))
            val_data_df.to_csv(dir)

            # For Convenience, also load feature_offsets to this folder
            feat_offs_df = pd.DataFrame(self.feature_offsets)
            dir = os.path.expanduser('{}/FeatOffs.csv'.format(dir_main))
            feat_offs_df.to_csv(dir)


        all_test_data = copy.deepcopy(self.x_test)
        all_test_data.append(self.duration_test.unsqueeze(1))
        all_test_data.append(self.event_test.unsqueeze(1))

        test_data_c = torch.cat(tuple(all_test_data), dim=1)
        test_data_df = pd.DataFrame(test_data_c)
        dir = os.path.expanduser('{}/TestData.csv'.format(dir_main))
        test_data_df.to_csv(dir)

        dir = os.path.expanduser(r'{}/ViewNames.txt'.format(dir_main))
        with open(dir, 'w') as fp:
            for item in self.view_names:
                # write each item on a new line
                fp.write("%s\n" % item)

        dir = os.path.expanduser(r'{}/cols_remove.txt'.format(dir_main))
        with open(dir, 'w') as fp:
            for item in cols_remove:
                #write each item on a new line
                fp.write("%s\n" % item)


        ######################################## STORE PREPROCESSED DATA ##########################################


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
                 ; dtype : List of Numpy Arrays, List of Numpy Arrays, List of Numpy Arrays
        """
        if k_folds == 1:  # If only a single split is wanted, we split in 80/20 train/validation
            skfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        else:
            skfold = StratifiedKFold(n_splits=k_folds,shuffle=True,random_state=42)

        # Save all folds in here, each sublist of type [train_fold_df, val_fold_df]
        folds = [[] for i in range(k_folds)]
        # Save all targets (event indicators) here, each sublist of type [train_fold_targets_df, val_fold_targets_df]
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

        return folds, folds_targets, folds_durations


    def feature_selection(self, method = None,
                          feature_names = None, # for PPI network
                          components = None, # for PCA
                          thresholds = [0,0,0,0],
                          saved_data_loading = False,
                          saved_data_preprocessing = 'MaxAbs',
                          saved_data_folder_name = 'KIRP4VIEWS',
                          columns_removed = [],
                          k_variance_features = 2000): # for Variance
        """
        :param method: Feature selection method (Eigengenes/PCA/Variance/AE/PPI) ; dtype : String
        :param feature_names: Names of Features [needed for PPI-Network] ; dtype : List of Strings
        :param components: Number of Components for each view [needed for PCA] ; dtype : List of Int
        :param thresholds: Threshold for each view [needed for Variance] ; dtype : List of  Float [between 0 and 1]
        :param saved_data_loading : Load saved data ; dtype : Boolean
        :param saved_data_preprocessing : Preprocessed type of saved data ; dtype : String
        :param saved_data_folder_name : Name of folder where data is saved ; dtype : String
        :return: Train/Validation/Test Set with respective duration & event values after feature selection
                 ; dtype : List of Lists [for each fold] of Lists [for each view] of Tensors
        """





        ######################################## LOAD DATA IN DIRECTLY ##########################################

        if saved_data_loading == True:
            dir = os.path.expanduser("~/{}/Project/ProcessedNotFeatSelectedData/{}/{}/".format(self.direc_set,saved_data_preprocessing,saved_data_folder_name))
            trainset_0,trainset_1,trainset_2,trainset_3,trainset_4,valset_0,valset_1,valset_2,valset_3,valset_4,featoffs,testset,view_names = load_data(data_dir=dir)

            ####### SET VIEW COUNT ########
            self.n_views = len(view_names)
            self.view_names = view_names
            featoffs = list(featoffs.values)
            for idx,_ in enumerate(featoffs):
                featoffs[idx] = featoffs[idx].item()

            trainset = [trainset_0 ,trainset_1,trainset_2,trainset_3,trainset_4]
            valset = [valset_0 ,valset_1,valset_2,valset_3,valset_4]
            train_data_folds = []
            train_duration_folds = []
            train_event_folds = []
            val_data_folds = []
            val_duration_folds = []
            val_event_folds = []

            for c2,_ in enumerate(trainset):
                train_data = []
                val_data = []
                test_data = []
                for c,feat in enumerate(featoffs):
                    if c < len(featoffs) - 3: # train data views
                        train_data.append(np.array((trainset[c2].iloc[:, featoffs[c] : featoffs[c+1]]).values).astype('float32'))
                        val_data.append(np.array((valset[c2].iloc[:, featoffs[c]: featoffs[c + 1]]).values).astype('float32'))
                        test_data.append(np.array((testset.iloc[:, featoffs[c]: featoffs[c + 1]]).values).astype('float32'))
                    elif c == len(featoffs) - 3: # duration
                        train_duration = (np.array((trainset[c2].iloc[:, featoffs[c] : featoffs[c+1]]).values).astype('float32')).squeeze(axis=1)
                        val_duration = (np.array((valset[c2].iloc[:, featoffs[c]: featoffs[c + 1]]).values).astype('float32')).squeeze(axis=1)
                        test_duration = (np.array((testset.iloc[:, featoffs[c]: featoffs[c + 1]]).values).astype('float32')).squeeze(axis=1)
                    elif c == len(featoffs) -2: # event
                        train_event = (np.array((trainset[c2].iloc[:, featoffs[c] : featoffs[c+1]]).values).astype('float32')).squeeze(axis=1)
                        val_event = (np.array((valset[c2].iloc[:, featoffs[c]: featoffs[c + 1]]).values).astype('float32')).squeeze(axis=1)
                        test_event = (np.array((testset.iloc[:, featoffs[c]: featoffs[c + 1]]).values).astype('float32')).squeeze(axis=1)

                train_data_folds.append(train_data)
                val_data_folds.append(val_data)
                train_duration_folds.append(train_duration)
                val_duration_folds.append(val_duration)
                train_event_folds.append(train_event)
                val_event_folds.append(val_event)


            # Overwrite
            self.train_folds = train_data_folds
            self.val_folds = val_data_folds
            self.train_folds_durations = train_duration_folds
            self.val_folds_durations = val_duration_folds
            self.train_folds_events = train_event_folds
            self.val_folds_events = val_event_folds
            self.x_test = test_data
            self.event_test = test_event
            self.duration_test = test_duration

            # Casting needed for neural nets
            for c in range(self.n_folds):
                # Cast all elements to torch.float32
                self.train_folds_durations[c] = torch.from_numpy(self.train_folds_durations[c]).to(torch.float32)
                self.train_folds_events[c] = torch.from_numpy(self.train_folds_events[c]).to(torch.float32)
                self.val_folds_durations[c] = torch.from_numpy(self.val_folds_durations[c]).to(torch.float32)
                self.val_folds_events[c] = torch.from_numpy(self.val_folds_events[c]).to(torch.float32)
                for c_view in range(len(self.train_folds[c])):
                    self.train_folds[c][c_view] = torch.from_numpy(self.train_folds[c][c_view]).to(torch.float32)
                    self.val_folds[c][c_view] =torch.from_numpy(self.val_folds[c][c_view]).to(torch.float32)

            # Also cast train duration & events
            self.duration_test = torch.from_numpy(self.duration_test).to(torch.float32)
            self.event_test = torch.from_numpy(self.event_test).to(torch.float32)
            for c_view in range(len(self.x_test)):
                self.x_test[c_view] = torch.from_numpy(self.x_test[c_view]).to(torch.float32)



            dimensions_train = [x.shape[1] for x in train_data]
            dimensions_val = [x.shape[1] for x in val_data]
            dimensions_test = [x.shape[1] for x in test_data]

            assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/test'


        ######################################## LOAD DATA IN DIRECTLY ##########################################



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
                dir = os.path.expanduser('~/{}/Project/TCGAData/eigengene_mode.txt'.format(self.direc_set))
                if c_fold == 0:
                    # Train/Val/Test data
                    mode = "all"
                    with open(dir, 'w') as f:
                        f.write(mode)
                else:
                    # Train/Val data
                    mode = "folds"
                    with open(dir, 'w') as f:
                        f.write(mode)


                eg_view.eigengene_multiplication()
                # If the mode is folds, we'll return an empty list for the test matrices
                eigengene_matrices,eigengene_matrices_val, eigengene_matrices_test = eg_view.get_eigengene_matrices(self.view_names)

                # As list as each eigengene matrix is of a different size
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

                # Save values : We still need to find the minimum number of eigengenes across
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
            """ Principal Component Analysis Feature Selection.
                Selecting n components for each view individually.
            """
            PCA_train_tensors = [[] for i in range(len(self.train_folds))]
            PCA_val_tensors = [[] for i in range(len(self.train_folds))]
            PCA_test_tensors = [[] for i in range(len(self.train_folds))]
            feat_offs = [[] for i in range(len(self.train_folds))]



            print("PCA feature selection...")
            for c,fold in enumerate(self.train_folds):
                print("PCA Fold {}".format(c))
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



            return PCA_train_tensors,PCA_val_tensors,PCA_test_tensors, \
                   self.train_folds_durations,self.train_folds_events, \
                   self.val_folds_durations,self.val_folds_events, \
                   self.duration_test,self.event_test




        if method.lower() == 'variance_2':
            """ 
            Variance based feature selection
            Select the top k features with highest variance.
            """
            variance_train_tensors = [[] for i in range(len(self.train_folds))]
            variance_val_tensors = [[] for i in range(len(self.train_folds))]
            variance_test_tensors = [[] for i in range(len(self.train_folds))]



            for c,fold in enumerate(self.train_folds):
                for view in range(self.n_views):
                    # Select the top 2000 features
                    ordered_feature_subset_train = list(np.nanvar(self.train_folds[c][view], axis=0).argsort()[::-1][:k_variance_features])
                    # ordered_feature_subset_validation = list(np.nanvar(val_data, axis=0).argsort()[::-1][:2000])
                    # ordered_feature_subset_test = list(np.nanvar(test_data, axis=0).argsort()[::-1][:2000])
                    train_data = np.empty([self.train_folds[c][view].size(0),k_variance_features])
                    val_data = np.empty([self.val_folds[c][view].size(0),k_variance_features])
                    test_data = np.empty([self.x_test[view].size(0),k_variance_features])
                    for sample_train in range(self.train_folds[c][view].size(0)):
                        for c_idx_train,idx_train in enumerate(ordered_feature_subset_train):
                            train_data[sample_train][c_idx_train] = self.train_folds[c][view][sample_train][idx_train]

                    for sample_val in range(self.val_folds[c][view].size(0)):
                        for c_idx_val,idx_val in enumerate(ordered_feature_subset_train):
                            val_data[sample_val][c_idx_val] = self.val_folds[c][view][sample_val][idx_val]

                    for sample_test in range(self.x_test[view].size(0)):
                        for c_idx_test,idx_test in enumerate(ordered_feature_subset_train):
                            test_data[sample_test][c_idx_test] = self.x_test[view][sample_test][idx_test]

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


        if method.lower() == 'variance':
            """ 
            Variance based feature selection
            Removing low-variance features based on a threshold for each view individually.
            """
            variance_train_tensors = [[] for i in range(len(self.train_folds))]
            variance_val_tensors = [[] for i in range(len(self.train_folds))]
            variance_test_tensors = [[] for i in range(len(self.train_folds))]




            for c,fold in enumerate(self.train_folds):
                # Reset thresholds ; Currently, we implemented the variance feature selection in a way that
                # each view will be represented with about 10/15 % of all its features. For that to work,
                # the threshold needs to be resetted for each fold (as it gets increased in 0.001 steps till we reach
                # 10/11 %)
                thresholds = [0,0,0,0]
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

                            # Check that we select between 10/15 % of original features
                            if ((torch.tensor(train_data)).size(dim=1) >= 0.10 * (self.train_folds[c][view]).size(dim=1)) and ((torch.tensor(train_data)).size(dim=1) <= 15 * (self.train_folds[c][view]).size(dim=1)):
                                print("Threshold of view ", self.view_names[view], "is ", thresholds[view])
                                break
                            if view == 2 or view == 3:
                                # Currently only for DNA & mRNA
                                break
                            else:
                                #mRNA & DNA (view 0&1) have about 20000 features, so sometimes we should increment with a
                                # higher value bc else the search will take too long
                                if view == 0 or view == 1:
                                    print((torch.tensor(train_data)).size(dim=1), "for view", view, "and fold", c)
                                    thresholds[view] = thresholds[view] + 0.01
                                else:
                                    # Currently just analyzing mRNA & DNA
                                    print((torch.tensor(train_data)).size(dim=1), "for view", view, "and fold", c)
                                    thresholds[view] = thresholds[view] + 0.001
                        except ValueError:
                            thresholds[view] = thresholds[view] - 0.001



                    # For cross setting tests in AE, views need to have the same amount of features. Thus we may need to delete about some features to get the same sizes


                    variance_train_tensors[c].append(torch.tensor(train_data))
                    variance_val_tensors[c].append(torch.tensor(val_data))
                    variance_test_tensors[c].append(torch.tensor(test_data))

                if variance_train_tensors[c][0].size(dim=1) != variance_train_tensors[c][1].size(dim=1):
                    minimal_size = min(variance_train_tensors[c][0].size(dim=1),variance_train_tensors[c][1].size(dim=1))
                    if variance_train_tensors[c][0].size(dim=1) == minimal_size:
                        print("Old Size:", variance_train_tensors[c][1].size(dim=1))
                        variance_train_tensors[c][1] = variance_train_tensors[c][1][:,0:minimal_size]
                        variance_val_tensors[c][1] = variance_val_tensors[c][1][:,0:minimal_size]
                        variance_test_tensors[c][1] = variance_test_tensors[c][1][:,0:minimal_size]
                        print("New Size:", variance_train_tensors[c][1].size(dim=1))

                    else:
                        print("Old Size:", variance_train_tensors[c][0].size(dim=1))
                        variance_train_tensors[c][0] = variance_train_tensors[c][0][:,0:minimal_size]
                        variance_val_tensors[c][0] = variance_val_tensors[c][0][:,0:minimal_size]
                        variance_test_tensors[c][0] = variance_test_tensors[c][0][:,0:minimal_size]
                        print("New Size:", variance_train_tensors[c][0].size(dim=1))


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


            return self.train_folds, self.val_folds, test_folds, \
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
                ppi_train = FeatureSelection.PPI(self.train_folds[c_fold], feature_names, self.view_names, columns_removed)
                ppi_val = FeatureSelection.PPI(self.val_folds[c_fold], feature_names, self.view_names, columns_removed)

                print("Getting PPI train matrices for fold : {}".format(c_fold + 1))
                data_train,edge_index_train, proteins_used_train = ppi_train.get_matrices()
                print("Getting PPI validation matrices for fold : {}".format(c_fold + 1))
                data_val, edge_index_val, proteins_used_val = ppi_val.get_matrices()

                if c_fold == 0: # only need to get test set once
                    print("Getting PPI test matrices")
                    ppi_test = FeatureSelection.PPI(self.x_test, feature_names, self.view_names, columns_removed)
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
        :param batch_size : Batch size ; dtype : Int
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
        :param batch_size : Batch size ; dtype : Int
        """
        test_loader = DataLoader(
            dataset=self.test_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=10)
        return test_loader







def load_data(data_dir):
    """
    Function to load data. Needed for Optuna Optimization.
    :param data_dir: Directory in which data is stored.
    :return: data and feature offsets (for feature values, duration and event)
    """

    trainset_0 = pd.read_csv(
        os.path.join(data_dir + "TrainData_0.csv"), index_col=0)

    valset_0 = pd.read_csv(
        os.path.join(data_dir + "ValData_0.csv"), index_col=0)


    trainset_1 = pd.read_csv(
        os.path.join(data_dir + "TrainData_1.csv"), index_col=0)

    valset_1 = pd.read_csv(
        os.path.join(data_dir + "ValData_1.csv"), index_col=0)

    trainset_2 = pd.read_csv(
        os.path.join(data_dir + "TrainData_2.csv"), index_col=0)

    valset_2 = pd.read_csv(
        os.path.join(data_dir + "ValData_2.csv"), index_col=0)

    trainset_3 = pd.read_csv(
        os.path.join(data_dir + "TrainData_3.csv"), index_col=0)

    valset_3 = pd.read_csv(
        os.path.join(data_dir + "ValData_3.csv"), index_col=0)

    trainset_4 = pd.read_csv(
        os.path.join(data_dir + "TrainData_3.csv"), index_col=0)

    valset_4 = pd.read_csv(
        os.path.join(data_dir + "ValData_3.csv"), index_col=0)



    testset = pd.read_csv(
        os.path.join(data_dir +  "TestData.csv"), index_col=0)

    featoffs = pd.read_csv(
        os.path.join(data_dir + "FeatOffs.csv"), index_col=0)

    view_names = open(os.path.join(data_dir + "ViewNames.txt"),"r")
    views = view_names.read().split("\n")
    # "" gets appened as last element in view list somehow
    del views[-1]






    return trainset_0,trainset_1,trainset_2,trainset_3,trainset_4,valset_0,valset_1,valset_2,valset_3,valset_4,featoffs,testset,views



