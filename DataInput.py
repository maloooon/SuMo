"""Survival data, source: https://github.com/havakv/pycox/blob/refactor_out_torchtuples/examples/lightning_logistic_hazard.py."""
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
from pycox.preprocessing.label_transforms import LabTransDiscreteTime


class MultiviewDataset(Dataset[List[torch.Tensor]]):

    X: List[torch.Tensor]#data structure inputX
    # [tensor von allen samples view 1,  tensor von allen samples view 2, ... ]
    # in dem tensor jeweils features sample1, features sample 2, ...
    # Code geht bis jetzt davon aus, dass der Input der csv files sortiert ist nach samples
    # TODO : für unsortierten Input (also nicht sortiert nach Samples) Sortierfunktion einbauen (mit pd.concat)
    meta_data: Dict[str, torch.Tensor]
    # *X --> beliebig großer input für X
    # **kwargs --> beliebig viele Argumente
    # -> None : return None

    def __init__(self, *X, **kwargs) -> None: # removed *
        self.n_samples = X[0].size(0) # Sample size
        self.n_views = len(X) # Views (mRNA, DNA, microRNA, RPPA)

        # Check if for each view (each tensor containing all samples for one view) the amount of samples
        # is the same as the sample size, otherwise print mismatch
        assert all(
            view.size(0) == self.n_samples for view in X
        ), "Size mismatch between tensors"


        # ~ : Invert, so basically torch."isnotnan"
        # torch.isnan : returns boolean tensor with same size ; True when value is NaN
        # torch.nan_to_num : replace NaN, +inf, -inf with values (0, 3e^38, -3e^38 respectively)
        self.mask = [~torch.isnan(x_view) for x_view in X]
        self.X = [torch.nan_to_num(x_view) for x_view in X]


    def __len__(self):
        """Return the amount of samples"""
        return self.n_samples

    def __getitem__(self, index):
        """for each data type (m) return feature-values of samples at position index"""
        #index, : slicing for multidimensional array (multidim tensor)
        return [self.X[m][index, :] for m in range(self.n_views)], [
            self.mask[m][index, :] for m in range(self.n_views)
        ]

    def values(self):
        return self.X



class MultiviewDatasetSurv(MultiviewDataset):
    """Survival data,
       duration : float/integer (?) value representing time till event happened or patient was censored
       event : (float/integer (?) ; 0 : patient was censored, 1 : event happened
       """

    duration: torch.Tensor
    event: torch.Tensor

    def __init__(
            self, *X, duration: torch.Tensor, event: torch.Tensor, **kwargs
    ) -> None:
        super().__init__(*X, **kwargs)

        # check that we have all samples in duration and event tensors
        assert (
                duration.size(0) == self.n_samples
        ), "Shape mismatch between `X` and `duration`"
        assert event.size(0) == self.n_samples, "Shape mismatch between `X` and `event`"

        self.duration = duration
        self.event = event

    def __getitem__(self, index):
        X, mask = super().__getitem__(index)
        return X, mask, self.duration[index], self.event[index]



def tensor_to_dataframe(
        data : torch.Tensor,
) -> pd.DataFrame:
    """Convert a tensor into a Dataframe"""
    return pd.DataFrame(data.numpy())



def preprocess_features(
        df_train: pd.DataFrame, # train dataset
        df_test: pd.DataFrame, # test dataset
        cols_std: List[str], # feature names, numeric variables
        cols_leave: List[str], # feature names, binary variables
) -> Tuple[torch.Tensor]:
    """Preprocess different data
    #Numeric variables: Standardize
    #Binary variables: No preprocessing necessary
    (#Categorical variables: Create embeddings)

    PRAD data only contains numeric variables (?)

    see pycox tutorial  : https://nbviewer.org/github/havakv/pycox/blob/master/examples/01_introduction.ipynb
    or https://towardsdatascience.com/how-to-implement-deep-neural-networks-for-time-to-event-analyses-9aa0aeac4717
    """

    if cols_std is not None: # If there is a need to standardize certain features ???
        standardize = [([col], StandardScaler()) for col in cols_std]
        leave = [(col, None) for col in cols_leave]
        # map together so we have all features present again
        mapper = DataFrameMapper(standardize + leave)
        x_train = mapper.fit_transform(df_train).astype(np.float32)
        x_test = mapper.transform(df_test).astype(np.float32)
    else:
        x_train = df_train.values.astype(np.float32)
        x_test = df_test.values.astype(np.float32)
    # torch.from_numpy : create tensor from numpy array ; why dont we directly transform to tensor
    # and first to numpy ?
    return torch.from_numpy(x_train), torch.from_numpy(x_test)


# see https://github.com/havakv/pycox/issues/60
class SurvMultiviewDataModule(pl.LightningDataModule):
    """Input is the whole dataframe : We merge all data types together, with the feature offsets we can access
       certain data types"""
    def __init__(
            self, df, feature_offsets, n_durations: int = 10, batch_size=100, **kwargs
    ):
        super().__init__()
        self.df = df
        self.feature_offsets = feature_offsets # cumulative sum of features in list of features [6000, 6000, 336, 148]
        self.n_views = len(feature_offsets) - 1
        self.n_durations = n_durations
        self.batch_size = batch_size
        self.kwargs = kwargs

    def setup(
            self,
            test_size=0.2,
            cols_std=None,   #numeric feature names
            cols_leave=None, #binary feature names
            col_duration="duration",
            col_event="event",
            cols_meta=[],
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
        cols_drop = cols_survival + cols_meta # ??? meta_data ?


        if cols_leave is None:
            cols_leave = []

        # features with numeric values
        if cols_std is None:
            cols_std = [
                col for col in self.df.columns if col not in cols_leave + cols_drop
            ]

        # ???
        self.meta_data_train, self.meta_data_test = (
            df_train[cols_meta],
            df_test[cols_meta],
        )

        # Preprocess train and test data with programmed function
        self.x_train, self.x_test = preprocess_features(
            df_train=df_train, # drop duration/event from df, as we only want the numeric values
            df_test=df_test,   # for training and testing
            cols_std=cols_std,
            cols_leave=cols_leave,
        )

        # Save event,duration for train and test set
        self.label_train = df_train[["event","duration"]]
        self.label_test = df_test[["event","duration"]]

        #drop
        df_train.drop(cols_drop, axis=1)
        df_test.drop(cols_drop, axis=1)

        #LabTransDiscreteTime : Discretize continuous (duration, event) pairs based on a set of cut points.
        # no parameter given, so I think equidistant discretization (scheme='equidistant' in pycox module)
        #logistic_hazard : A discrete-time survival model that minimize the likelihood for right-censored data by
        #parameterizing the hazard function
        #example : dog and human life span : dog grow older seven times faster than humans
        # -->  SD(t) = SH(7t), where t is time, SD survival function dog, SH survival function human
        # smoker/ non smoker : Snonsmoker(t) = Ssmoker(a*t) (Der nicht Raucher ist, wenn er 20 Jahre alt ist, so alt wie
        # der Raucher, wenn er a * 20 Jahre alt ist)
        # a : acceleration factor which can be parameterized as exp(a) in a regression framework, a is to be estimated
        #(learnt) from the data --> Snonsmoker(t) = Ssmoker(exp(a)*t)
        # Needed later on to distinguish between multiple cancer types ? or also within one cancer ?


        #LabTransDiscreteTime : init(self, cuts, scheme='equidistant', min_=0., dtype=None)
        # cuts : cuts {int, array} -- Defining cut points, either the number of cuts, or the actual cut points.
        # in our case we take the times until event happens/person is right-censored as cut points
        #   self.labtrans = logistic_hazard.LabTransDiscreteTime(self.n_durations)
    #    self.labtrans = LabTransDiscreteTime(self.n_durations)
        if stage == "fit" or stage is None:
            # Pre-process features and targets
    #        self.y_train = self.labtrans.fit_transform(duration_train, event_train)
    #        self.y_train_duration = torch.from_numpy(self.y_train[0])
    #        self.y_train_event = torch.from_numpy(self.y_train[1])
            # Input train_set as list of lists , with each view in a singe list (test ! )
            self.train_set = MultiviewDatasetSurv(
                *[
                    self.x_train[
                    :, self.feature_offsets[m] : self.feature_offsets[m + 1]
                    ]
                    for m in range(self.n_views)
                ],
                duration=torch.tensor(self.label_train["duration"].values),
                event=torch.tensor(self.label_train["event"].values),
            )

        #    self.n_out_features = self.labtrans.out_features

        if stage == "test" or stage is None:
          #  self.test_set =


            self.df_test = df_test

    def train_dataloader(self):
        """
        Build training dataloader
        num_workers set to 0 by default because of some thread issue
        """
        # Dataloader : Python iterable over a dataset
        train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            **self.kwargs,
        )
        return train_loader



if __name__ == "__main__":

    #os.path.join --> einzelne paths joinen ; os refers to Betriebssystem
    #index_col = 0 since first column are patient references (tcga-xy-000)
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

    # First column : First feature (column[0])

    data_survival.rename(
        {"vital_status": "event", "duration": "duration"}, axis="columns", inplace=True
    )

    data_survival.drop(labels= 'cancer_type', axis='columns', inplace=True)
    # Add duration columns to multi-omics data dataframes for dropping samples
    duration = data_survival["duration"]
    data_mRNA = data_mRNA.join(duration)
    data_DNA = data_DNA.join(duration)
    data_microRNA = data_microRNA.join(duration)
    data_RPPA= data_RPPA.join(duration)
    #TODO : Diesen ganzen Absatz kann man auch direkt über die pd.concat Liste machen unten,
    #TODO : Wusste nur nicht am Anfang, dass man die brauchen wird
    #Drop all samples (rows) which have survival duration 0 or smaller
    data_survival.drop(data_survival[data_survival["duration"] <= 0].index, axis = 0, inplace= True)
    data_mRNA.drop(data_mRNA[data_mRNA["duration"] <= 0].index, axis = 0, inplace= True)
    data_DNA.drop(data_DNA[data_DNA["duration"] <= 0].index, axis = 0, inplace= True)
    data_microRNA.drop(data_microRNA[data_microRNA["duration"] <= 0].index, axis = 0, inplace= True)
    data_RPPA.drop(data_RPPA[data_RPPA["duration"] <= 0].index, axis = 0, inplace= True)

    #Delete duration columns from multi-omics data dataframes
    data_mRNA.drop("duration", axis="columns", inplace=True)
    data_DNA.drop("duration", axis="columns", inplace=True)
    data_microRNA.drop("duration", axis="columns", inplace=True)
    data_RPPA.drop("duration", axis="columns", inplace=True)

    # TODO : Implement possibility of dataframes where we don't have data for certain data types ?
    n_samples = data_survival.shape[0]
    #print(n_samples) # 498

    n_features = [len(data_mRNA.columns), len(data_DNA.columns), len(data_microRNA.columns), len(data_RPPA.columns), len(data_survival.columns)]
    #print(n_features) # 6000, 6000, 336, 148

    # cumulative sum of features in list
    feature_offsets = [0] + np.cumsum(n_features).tolist()
    #print("foff", feature_offsets) # 0,6000,12000,12336,12484,12486


    df = pd.concat([data_mRNA, data_DNA, data_microRNA, data_RPPA, data_survival], axis=1)


    """
    row_names = []
    for row in df.index:
        row_names.append(row)

    # COLUMN NAMES
    column_names = []
    for column in df.columns:
        column_names.append(column)


    # changing NaN, +-inf to numeric values
    data_tensor_list = [torch.nan_to_num(torch.tensor(data_mRNA.values))
        , torch.nan_to_num(torch.tensor(data_DNA.values))
        ,torch.nan_to_num(torch.tensor(data_microRNA.values))
        ,torch.nan_to_num(torch.tensor(data_RPPA.values))
        ,torch.nan_to_num(torch.tensor(data_survival.values))]


    data_mRNA = tensor_to_dataframe(data_tensor_list[0])
    data_DNA = tensor_to_dataframe(data_tensor_list[1])
    data_microRNA = tensor_to_dataframe(data_tensor_list[2])
    data_RPPA = tensor_to_dataframe(data_tensor_list[3])
    data_survival = tensor_to_dataframe((data_tensor_list[4]))

    # Dataframe with fixed values
    df = pd.concat([data_mRNA, data_DNA, data_microRNA, data_RPPA, data_survival], axis=1)
    #rename columns and rows (index)
    df.columns = column_names
    df.index = row_names
    """
    data_test = MultiviewDataset(torch.tensor(df))

    #print(df.shape) # 498 x 12486
    data = SurvMultiviewDataModule(df,feature_offsets)
    data.setup()
    # print(data.df.isnull().values.any())
    data_loader = data.train_dataloader()
    x_train, x_mask, y_train_duration, y_train_event = next(iter(data_loader))
    print(x_train)




#%%
