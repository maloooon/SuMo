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



class MultiOmicsDataset(Dataset):

    # [tensor von allen samples view 1,  tensor von allen samples view 2, ... ]
    # in dem tensor jeweils features sample1, features sample 2, ...

    def __init__(self, X, duration, event, type= 'new'):
        self.type = type # type of data : new data is unprocessed, 'processed' means already feature selected
        self.n_views = len(X) # number views (mRNA, DNA, microRNA, RPPA)


        if self.type == 'new':
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
            self.X = X
            self.duration = duration
            self.event = event

            # Change dtypes of tensors for usage of NN later on
            for view in range(len(self.X)):
                self.X[view] = self.X[view].to(torch.float32)
            self.duration = torch.from_numpy(self.duration).to(torch.float32)
            self.event = torch.from_numpy(self.event).to(torch.float32)


            self.n_samples = X[0].size(0)

#y = torch.from_numpy(boston.target.reshape(-1, 1)).float()

            # no mask anymore --> we have selected features now, so old mask doesn't make sense (?)
            # make new mask based on new data ?

            # no size check as e.g. in eigengenematrix diff. amount of samples due to preprocessing


      #  self.duration = duration
      #  self.event = event
        self.type = type


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
            x_test_ordered_by_view.append(torch.tensor((x_test_df.iloc[:, feat_offsets_PRAD[x]:
                                                                           feat_offsets_PRAD[x + 1]]).values))




    else:
        # Order by view so it works with Dataset class
        x_train_ordered_by_view = []
        x_test_ordered_by_view = []
        for x in range(len(feature_offset) - 3): # -3 bc we don't have duration/event in training tensor
            x_train_ordered_by_view.append(torch.tensor((df_train.iloc[:, feat_offsets_PRAD[x]:
                                                                          feat_offsets_PRAD[x + 1]]).values))
            x_test_ordered_by_view.append(torch.tensor((df_test.iloc[:, feat_offsets_PRAD[x]:
                                                                         feat_offsets_PRAD[x + 1]]).values))



    return x_train_ordered_by_view, x_test_ordered_by_view


class SurvMultiOmicsDataModule(pl.LightningDataModule):
    """Input is the whole dataframe : We merge all data types together, with the feature offsets we can access
       certain data types ; dataframe also contains duration and event !"""
    def __init__(
            self, df, feature_offsets, view_names, n_durations = 10): # 389 all training samples
        super().__init__()
        self.df = df
        self.feature_offsets = feature_offsets # cumulative sum of features in list of features
        self.n_durations = n_durations
        self.view_names = view_names
        self.n_views = len(view_names)


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
        n_train_samples = df_train.shape[0]
        n_test_samples = df_test.shape[0]
        self.duration_train, self.duration_test = (
            df_train[col_duration].values,
            df_test[col_duration].values,
        )
        self.event_train, self.event_test = df_train[col_event].values, df_test[col_event].values



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

        self.x_train = [torch.nan_to_num(x_view) for x_view in self.x_train]
        self.x_train_mask = [torch.isnan(x_view) for x_view in self.x_train] #List of booleans for each numeric value in samples ;True : NaN values

        self.x_test = [torch.nan_to_num(x_view) for x_view in self.x_test]
        self.x_test_mask = [torch.isnan(x_view) for x_view in self.x_test]

        return n_train_samples, n_test_samples


    def feature_selection(self, method = None):

        if method.lower() == 'eigengenes':
            # for train sets
         #   eigengene_duration_tensor = []
         #   eigengene_event_tensor = []
            # for test sets
         #   eigengene_duration_tensor_test = []
         #   eigengene_event_tensor_test = []

            for view in range(self.n_views):

                eg_view = FeatureSelection.F_eigengene_matrices(train=self.x_train[view],
                                                                mask=self.x_train_mask[view],
                                                                view_name=self.view_names[view],
                                                                duration=self.duration_train,
                                                                event=self.event_train,
                                                                stage= 'train')

                eg_view_test = FeatureSelection.F_eigengene_matrices(train=self.x_test[view],
                                                                     mask=self.x_test_mask[view],
                                                                     view_name=self.view_names[view],
                                                                     duration=self.duration_test,
                                                                     event=self.event_test,
                                                                     stage='test')

                eg_view.preprocess()
                eg_view_test.preprocess()


            #    eigengene_duration_tensor.append(duration_tensor)
            #    eigengene_event_tensor.append(event_tensor)
            #    eigengene_duration_tensor_test.append(duration_tensor_test)
            #    eigengene_event_tensor_test.append(event_tensor_test)


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
            variance = [0.8,0.8,0.8,0.8] # TODO : Grid search


            for view in range(self.n_views):
                view_train_PCA = FeatureSelection.F_PCA(self.x_train[view], keep_variance=variance[view])
                view_test_PCA = FeatureSelection.F_PCA(self.x_test[view], keep_variance=variance[view])
                pc_train_view = view_train_PCA.apply_pca()
                pc_test_view = view_test_PCA.apply_pca()
                PCA_train_tensors.append(torch.tensor(pc_train_view))
                PCA_test_tensors.append(torch.tensor(pc_test_view))



            for c,view in enumerate(self.view_names):
                print("{} PCA feature selection : {} of size {} (samples, PC components).".format(view,
                                                                                                  PCA_train_tensors[c],
                                                                                                  PCA_train_tensors[c].shape))



            self.train_set = MultiOmicsDataset(PCA_train_tensors,
                                                   self.duration_train,
                                                   self.event_train,
                                                   type = 'processed')

            self.test_set = MultiOmicsDataset(PCA_test_tensors,
                                              self.duration_test,
                                              self.event_test,
                                              type = 'processed')





        if method.lower() == 'variance':
            variance_train_tensors = []
            variance_test_tensors = []
            #variance_selected_features = []
            thresholds = [0.8,0.8,0.8,0.6] # TODO : Grid Search

            for view in range(self.n_views):
                view_train_variance = FeatureSelection.F_VARIANCE(self.x_train[view], threshold= thresholds[view])
                view_test_variance = FeatureSelection.F_VARIANCE(self.x_test[view], threshold = thresholds[view])
                data_train_variance, mask_train_variance = view_train_variance.apply_variance()
                data_test_variance, mask_test_variance = view_test_variance.apply_variance()
             #   variance_selected_features.append([DataInputNew.features[view][index] for index in mask_variance]) #TODO : wenn man feature namen kennen will
                variance_train_tensors.append(torch.tensor(data_train_variance))
                variance_test_tensors.append(torch.tensor(data_test_variance))



            for c,view in enumerate(self.view_names):
          #      print("Reduction {} data with variance :".format(view), 1 - (len(variance_selected_features[c]) / len(DataInputNew.features[c])))
                print("{} variance feature selection : {} of size : {} (samples, latent features)".format(view, variance_train_tensors[c],
                                                                                                          variance_train_tensors[c].shape))


            self.train_set = MultiOmicsDataset(variance_train_tensors,
                                               self.duration_train,
                                                self.event_train,
                                               type = 'processed')

            self.test_set = MultiOmicsDataset(variance_test_tensors,
                                              self.duration_test,
                                              self.event_test,
                                              type = 'processed')




        if method.lower() == 'ae':


            AE_all_compressed_train_features = []
            AE_all_compressed_test_features = []


            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for view in range(self.n_views):
                model = FeatureSelection.F_AE(train= self.x_train[view]).to(device)

                optimizer = Adam(model.parameters(), lr=1e-3)

                criterion = nn.MSELoss() # reconstrution loss

                self.train_set = MultiOmicsDataset(
                           self.x_train, self.duration_train, self.event_train, type = 'new')



                self.test_set = MultiOmicsDataset(self.x_test, self.duration_test, self.event_test, type = 'new')

                ae_trainloader = DataLoader(self.train_set,batch_size=398,shuffle=True,drop_last=True)

                ae_testloader = DataLoader(self.test_set, batch_size=20, shuffle=True, drop_last=True)
                epochs = 1
                temp = []
                temp2 = []
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

                        train_loss = criterion(reconstructed, batch_data)

                        train_loss.backward()

                        optimizer.step()

                        loss += train_loss.item()



                    loss = loss / len(ae_testloader)

                    print("epoch : {}/{}, loss = {:.6f} for {} test data".format(epoch + 1, epochs, loss, self.view_names[view]))


                compressed_train_features_view = torch.cat(temp, 0)
                AE_all_compressed_train_features.append(compressed_train_features_view)

                compressed_test_features_view = torch.cat(temp, 0)
                AE_all_compressed_test_features.append(compressed_test_features_view)

            for x in range(len(AE_all_compressed_train_features)):
                AE_all_compressed_train_features[x] = torch.detach(AE_all_compressed_train_features[x]) #detach gradient as we only need
                # selected features

            for x in range(len(AE_all_compressed_test_features)):
                AE_all_compressed_test_features[x] = torch.detach(AE_all_compressed_test_features[x]) #detach gradient as we only need
                # selected features

            for c, view in enumerate(self.view_names):
                print("{} AE feature selection : {} of size {} (samples,features)".format(view, AE_all_compressed_train_features[c],
                                                                                          AE_all_compressed_train_features[c].shape))


            # TODO : lots of values pressed to 0 ! --> less layers, other activation ?


            # Here we can create a tensor for all data, because we have the same feature size for each view
            # due to AE feature selection
            # (views, samples, features)
            data_AE_selected_train_PRAD = tensor_helper(AE_all_compressed_train_features)
            data_AE_selected_test_PRAD = tensor_helper(AE_all_compressed_test_features)

            self.train_set = MultiOmicsDataset(data_AE_selected_train_PRAD,
                                              self.duration_train,
                                              self.event_train,
                                              type = 'processed')

            self.test_set = MultiOmicsDataset(data_AE_selected_test_PRAD,
                                              self.duration_test,
                                              self.event_test,
                                              type = 'processed')



        if method.lower() == 'ppi':
            ppn_train_mRNA = FeatureSelection.F_PPI_NETWORK(self.x_train)
            ppn_test_mRNA = FeatureSelection.F_PPI_NETWORK(self.x_test)
            adjacency_matrix_train, feature_matrices_train = ppn_train_mRNA.setup()
            adjacency_matrix_test, feature_matrices_test = ppn_test_mRNA.setup()

            print("Adjacency matrix : {}".format(adjacency_matrix_train))
            print("Feature matrix mRNA : {} of size {} (samples, proteins, features). "
                  "For each sample, we have {} proteins and {} possible features".format
                  (feature_matrices_train, feature_matrices_train.shape, feature_matrices_train.size(1), feature_matrices_train.size(2)))

            # TODO : build own dataloader for this class as differs from other data structures
            return adjacency_matrix_train, feature_matrices_train, adjacency_matrix_test, feature_matrices_test














    #    self.train_set = MultiOmicsDataset(
     #       self.x_train, self.duration_train, self.event_train, type = 'new')

      #  self.test_set = MultiOmicsDataset(
      #      self.x_test, self.duration_test, self.event_test, type = 'new')






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


view_names_PRAD = ['mRNA','DNA','microRNA','RPPA']
#Get names of features
features = []
for a in range(len(feat_offsets_PRAD) - 1):
    features.append(list(data_PRAD.columns.values[feat_offsets_PRAD[a]:
                                                  feat_offsets_PRAD[a+1]]))


multimodule = SurvMultiOmicsDataModule(data_PRAD, feat_offsets_PRAD, view_names_PRAD)
  #  n_train_samples, n_test_samples = multimodule.setup()


   # multimodule.feature_selection(method='ae')

  #  loader = multimodule.train_dataloader(batch_size= 20)

  #  for data,duration,event in loader:

   #     break










#tensor_data_order_by_sample = torch.tensor(data_PRAD.values)
#tensor_data_order_by_view = []

#for x in range(len(feat_offsets_PRAD) - 1):
#    tensor_data_order_by_view.append(torch.tensor((data_PRAD.iloc[:, feat_offsets_PRAD[x]:
#                                                                     feat_offsets_PRAD[x + 1]]).values))

#

