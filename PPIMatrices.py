import pandas as pd
import torch
import os
import numpy
import gzip
import ReadInData
import DataInputNew





if __name__ == '__main__':
    cancer_data = ReadInData.readcancerdata()
    multimodule = DataInputNew.SurvMultiOmicsDataModule(cancer_data[0][0],cancer_data[0][1],cancer_data[0][2])

    # Setup all the data
    n_train_samples, n_test_samples, n_val_samples = multimodule.setup()

    feature_names = cancer_data[0][3]

    multimodule.feature_selection("ppi", feature_names)








