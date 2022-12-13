import torch
import pandas
import os
import DataInputNew
import FeatureSelection


if __name__ == '__main__':
    # Eigengene matrices
    eigengene_matrices_tensor = FeatureSelection.eigengene_matrices_tensors
    eigengene_duration_tensor = FeatureSelection.eigengene_duration_tensor
    eigengene_event_tensor = FeatureSelection.eigengene_event_tensor

    dataset_eigengene = DataInputNew.MultiOmicsDataset(eigengene_matrices_tensor,
                                                       eigengene_duration_tensor,
                                                       eigengene_event_tensor,
                                                       type='processed')



"""Different modules to process the data after preprocessing and feature selection
   AE (AutoEncoder), NN (NeuralNet), GCN (Graph Convolutional Neural Net)"""


