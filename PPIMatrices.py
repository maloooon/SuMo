import pandas as pd
import torch
import os
import numpy
import gzip
import ReadInData
import DataInputNew






cancer_data = ReadInData.readcancerdata()
multimodule = DataInputNew.SurvMultiOmicsDataModule(cancer_data[0][0],cancer_data[0][1],cancer_data[0][2])

# Setup all the data
n_train_samples, n_test_samples, n_val_samples = multimodule.setup()

multimodule.feature_selection("ppi")







