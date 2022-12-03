
# sk learn feature selection methods
# https://scikit-learn.org/stable/modules/feature_selection.html


from typing import Dict, List, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import DataInputNew
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from torch import nn
from torch.optim import Adam
import os
import time



"""Implementation of different feature selections
   	eigengene matrices of gene co-expression modules
	SVD (Singular Value Decomposition)
	mRMR (maximum relevance minimum redundancy)
	PCA
	unsupervised variance-based feature selection
	AE hidden layer 
    graph attention
"""

class F_PCA():

    input : List[torch.Tensor]
    components : int

    def __init__(self, train, test = None, components = None, keep_variance = 0.5):
        """

        :param train:
        :param test:
        :param components:
        :param keep_variance: choose minimal number of PC components such that x % variance is retained
        """
        self.components = components
        self.train = train #training input
        self.test = test
        self.keep_variance = keep_variance


    def apply_pca(self):
        self.pca = PCA(self.keep_variance)

        #As we already standardized & transformed the data with StandardScaler(), we can apply PCA directly
        self.pca.fit(self.train)
        pca_train_data = self.pca.transform(self.train)

        return pca_train_data



    def plot(self):
        # TODO : Fix
        # Percentage of variance explained by each of the selected components
        per_var = np.round(self.pca.explained_variance_ratio_ * 100, decimals=1)
        labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

        plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal Component')
        plt.title('PCA Plot')
        plt.show()
        #TODO : in Jupyter Notebooks



class F_VARIANCE():
    """Removing features with low variance
       https://scikit-learn.org/stable/modules/feature_selection.html
       https://towardsdatascience.com/how-to-use-variance-thresholding-for-robust-feature-selection-a4503f2b5c3f
       Unsupervised variance-based feature selection
    """

    def __init__(self, train, features = None, test = None, threshold = 0.5):                                           # TODO : Threshold for each diff data type, grid search ?
        self.train = train
        self.features = features
        self.test = test
        self.threshold = threshold



    def apply_variance(self):
        # Get rid of features with 0 variance
        vt = VarianceThreshold(self.threshold)
        data_selected = vt.fit_transform(self.train)


        # get mask showing the features that were selected/not selected
        # True : selected, False : not selected (variance 0)
        mask = vt.get_support()

        # Find features that were selected
        index_list = []

        index = 0
        for x in mask:
            if x == True:
                index_list.append(index)
            index += 1





        return data_selected, index_list


class F_mRMR():
    """For classification tasks"""
    pass


class F_eigengene_matrices():
    """Implemented with R :
       https://github.com/huangzhii/lmQCM
    """

    def __init__(self, train, mask,view, test = None ):
        self.train = train
        self.test = test
        self.mask = mask
        self.view = view


    def preprocess(self):
        """For the eigengene matrices computation, we need to remove rows (samples) which had all NaN values
        (missing sample) aswell as each column (feature), which had value 0 for each patient (sample) (this feature
        is irrelevant either way, as it doesn't have any impact on any patient)"""


        train_df = pd.DataFrame(self.train.numpy())

        to_drop_columns = []

        for x in range(len(train_df.columns)):
            if (train_df.iloc[:, x] == train_df.iloc[:, x][0]).all():

                to_drop_columns.append(x)

        train_df = train_df.drop(train_df.columns[to_drop_columns], axis = 1)

        to_drop_index = []

        for x in range(len(train_df.index)):
            if torch.all(self.mask[x] == True):

                to_drop_index.append(x)

        train_df = train_df.drop(to_drop_index, axis = 0)

        train_df.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/{}_for_r.csv".format(self.view))

    # TODO : R Implementation in Python, sodass man Code nicht getrennt ausführen muss
    def get_eigengene_matrix(self):
        """Loading the eigengene matrix as a panda Dataframe created from R._"""
        while not os.path.exists(os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                                              "{}_eigengene_matrix.csv".format(self.view))):
            time.sleep(1)

        eigengene_matrix = pd.read_csv(os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                                                     "{}_eigengene_matrix.csv".format(self.view)), index_col=0)


        return eigengene_matrix




class F_AE(nn.Module):
    """Use the hidden layer between Encoder and Decoder.
    Based on : https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
    """
# TODO : Modell konvergiert ?
# TODO : test : variance-explained : wv varianz explained die approximation (encoder_output_layer) vs original daten
# TODO : preprocessing AE centering & standardizing
# normalize ? was genau gemeint ?
    def __init__(self, train, **kwargs):
        super().__init__()
        self.train = train # Need train for the dimension of features for current view

        self.encoder_hidden_1 = nn.Linear(in_features= train.size(dim=1), out_features= 512)
        self.encoder_hidden_2 = nn.Linear(in_features= 512, out_features= 256)
        self.encoder_hidden_3 = nn.Linear(in_features=256, out_features= 128)

        self.encoder_output_layer = nn.Linear(in_features=128, out_features= 128)

        self.decoder_hidden_1 = nn.Linear(in_features=128, out_features=128)
        self.decoder_hidden_2 = nn.Linear(in_features=128, out_features=256)
        self.decoder_hidden_3 = nn.Linear(in_features=256, out_features=512)
        self.decoder_hidden_output_layer = nn.Linear(in_features=512, out_features=train.size(dim=1))


    def forward(self,train, **kwargs):
        # Encoder hidden layer
        in_1 = self.encoder_hidden_1(train)
        out_1 = torch.relu(in_1)
        in_2 = self.encoder_hidden_2(out_1)
        out_2 = torch.relu(in_2)
        in_3 = self.encoder_hidden_3(out_2)
        out_3 = torch.relu(in_3)

        # select features from here
        in_layer_reduced = self.encoder_output_layer(out_3)
        out_layer_reduced = torch.relu(in_layer_reduced)

        # decode again (for training purposes)
        in_d_1 = self.decoder_hidden_1(out_layer_reduced)
        out_d_1 = torch.relu(in_d_1)
        in_d_2 = self.decoder_hidden_2(out_d_1)
        out_d_2 = torch.relu(in_d_2)
        in_d_3 = self.decoder_hidden_3(out_d_2)
        out_d_3 = torch.relu(in_d_3)
        in_d_final = self.decoder_hidden_output_layer(out_d_3)
        out_d_final = torch.relu(in_d_final)

        return out_d_final, out_layer_reduced























# TODO : allgemein je nachdem wv features eine view hat, relativ davon nur bestimmt viele für feature selection nehmen?

#if __name__ == '__main__':
    # Set batch size to size of training set ; We are currently preparing data,
    # thus we can use all samples as one batch (train with single batches later)
train_loader = DataInputNew.multimodule.train_dataloader(batch_size = 10)

train_data = []
mask_data = []


for data,mask, duration, event in train_loader:
    train_data.append(data)
    mask_data.append(mask)
   # print(data)
   # print(data[0]) # mRNA data
    break



views = len(train_data[0])

#   print(train_data)

#print(train_data[0][0]) # data all samples view 1 . shape (389, 6000)

print("PCA based selection")
mRNA_PCA = F_PCA(train_data[0][0], keep_variance= 0.9)
principal_components_mRNA = mRNA_PCA.apply_pca()
plot = mRNA_PCA.plot()
rows, columns = principal_components_mRNA.shape
principal_components_mRNA_df = pd.DataFrame(principal_components_mRNA,
                                            columns= [["PC {}".format(x) for x in range(columns)]])

print("Reduction mRNA data with PCA :", 1 - (columns / len(DataInputNew.features[0])))



DNA_PCA = F_PCA(train_data[0][1], keep_variance= 0.9)
principal_components_DNA = DNA_PCA.apply_pca()
rows, columns = principal_components_DNA.shape
principal_components_DNA_df = pd.DataFrame(principal_components_DNA,
                                           columns= [["PC {}".format(x) for x in range(columns)]])

print("Reduction DNA data with PCA :", 1 - (columns / len(DataInputNew.features[1])))


microRNA_PCA = F_PCA(train_data[0][2], keep_variance= 0.7)

principal_components_microRNA = microRNA_PCA.apply_pca()
rows, columns = principal_components_microRNA.shape

principal_components_microRNA_df = pd.DataFrame(principal_components_microRNA,
                                               columns= [["PC {}".format(x) for x in range(columns)]])


print("Reduction microRNA data with PCA :", 1 - (columns / len(DataInputNew.features[2])))





RPPA_PCA = F_PCA(train_data[0][3], keep_variance= 0.7)
principal_components_RPPA = RPPA_PCA.apply_pca()
rows, columns = principal_components_RPPA.shape



principal_components_RPPA_df = pd.DataFrame(principal_components_RPPA,
                                            columns= [["PC {}".format(x) for x in range(columns)]])


print("Reduction RPPA data with PCA :", 1 - (columns / len(DataInputNew.features[3])))

# print(principal_components_microRNA_df)

# ADD LABEL
#principal_components_RPPA_df['duration'] = duration
                                                                                                                    # TODO : variance based features change for different training sets ;
                                                                                                                    # TODO :  Rather do on all data and pick best there ?
                                                                                                                    # TODO : in ConcatAE paper 1000 best features chosen, but for mRNA and DNA
  # variance based : 50% reduction                                                                                                                  # TODO : we get about 1000-2000 even with a threshold of 1

mRNA_variance = F_VARIANCE(train_data[0][0], threshold= 1)
DNA_variance = F_VARIANCE(train_data[0][1], threshold= 1)
microRNA_variance = F_VARIANCE(train_data[0][2], threshold = 0.8)
RPPA_variance = F_VARIANCE(train_data[0][3], threshold = 0.6)

data_mRNA,mask_mRNA = mRNA_variance.apply_variance()
data_DNA,mask_DNA = DNA_variance.apply_variance()
data_microRNA,mask_microRNA = microRNA_variance.apply_variance()
data_RPPA,mask_RPPA = RPPA_variance.apply_variance()

mRNA_features_selected = [DataInputNew.features[0][index] for index in mask_mRNA]
DNA_features_selected = [DataInputNew.features[1][index] for index in mask_DNA]
microRNA_features_selected = [DataInputNew.features[2][index] for index in mask_microRNA]
RPPA_features_selected = [DataInputNew.features[3][index] for index in mask_RPPA]

print()
print("Variance based selection")
print("Reduction mRNA data with variance :", 1 - (len(mRNA_features_selected) / len(DataInputNew.features[0])))
print("Reduction DNA data with variance :", 1 - (len(DNA_features_selected) / len(DataInputNew.features[1])))
print("Reduction microRNA data with variance :", 1 - (len(microRNA_features_selected) / len(DataInputNew.features[2])))
print("Reduction RPPA data with variance :", 1 - (len(RPPA_features_selected) / len(DataInputNew.features[3])))
print()


print("Autoencoder based selection")
# F_AE

views_names = ['mRNA','DNA','microRNA','RPPA']
AE_all_compressed_features = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for x in range(views):
    model = F_AE(train= train_data[0][x]).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    criterion = nn.MSELoss() # reconstrution loss

    train_loader_2 = DataInputNew.multimodule.train_dataloader(batch_size = 80)         # TODO: training samples gibt es 389 viele, Primzahl : kein batch außer 1 um alle durchzugehen ; für 80 werden nur 320 samples durchgegangen und die letzten samples nicht beachtet !
    epochs = 1
    temp = []
    temp2 = []
    for epoch in range(epochs):
        loss = 0
        for batch_data, mask, duration, event in train_loader_2:

            batch_data = batch_data[x].view(-1, batch_data[x].size(1)).to(device) #mRNA

            optimizer.zero_grad()

            # compressed features is what we are interested in
            reconstructed, compressed_features = model(batch_data)
            if epoch == epochs - 1: # save compressed_features of last epoch for each batch
                temp.append(compressed_features) # list of tensors of compressed for each batch

            train_loss = criterion(reconstructed, batch_data)

            train_loss.backward()

            optimizer.step()

            loss += train_loss.item()



        loss = loss / len(train_loader_2)

        print("epoch : {}/{}, loss = {:.6f} for {} data".format(epoch + 1, epochs, loss, views_names[x]))
                                                                                                                         # TODO : Smaller loss (weniger layers? weniger input ? vllt feature selection

    compressed_features_view = torch.cat(temp, 0)
    AE_all_compressed_features.append(compressed_features_view)

for x in range(len(AE_all_compressed_features)):
    AE_all_compressed_features[x] = torch.detach(AE_all_compressed_features[x]) #detach gradient as we only need
                                                                          # selected features


print(AE_all_compressed_features) # TODO : lots of values pressed to 0 ! --> less layers, other activation ?




print()
print("Eigengene matrices")




# mRNA, works
a = F_eigengene_matrices(train= data[0][0], mask= mask[0][0], view = 'mRNA')
a.preprocess()
mRNA_eigengene_matrix = (a.get_eigengene_matrix()).transpose()



























