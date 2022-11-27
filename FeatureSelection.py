from typing import Dict, List, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import DataInputNew
import pandas as pd


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

    def __init__(self, train, test = None, components = 2, keep_variance = None):
        self.components = components
        self.train = train #training input
        self.test = test
        self.keep_variance = keep_variance


    def apply_pca(self):
        self.pca = PCA(self.components)

        #As we already standardized & transformed the data with StandardScaler(), we can apply PCA directly
        self.pca.fit(self.train)
        pca_train_data = self.pca.transform(self.train)

        return pca_train_data



    def plot(self):
        # Percentage of variance explained by each of the selected components
        per_var = np.round(self.pca.explained_variance_ratio_ * 100, decimals=1)
        labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

        plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal Component')
        plt.title('PCA Plot')
        plt.show()


if __name__ == '__main__':

    train_loader = DataInputNew.multimodule.train_dataloader(batch_size = 389)

    train_data = []

    for data,mask, duration, event in train_loader:
        train_data.append(data)

    #print(train_data[0][0]) # data all samples view 1 . shape (389, 6000)

    x = F_PCA(train_data[0][0])
    principal_components = x.apply_pca()
    principal_components_df = pd.DataFrame(principal_components, columns= ["PC1", "PC2"])
    print(principal_components_df)












