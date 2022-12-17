import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import DataInputNew

import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

if __name__ == '__main__':

    np.random.seed(1234)
    _ = torch.manual_seed(123)


    df_train = metabric.read_df()
    df_test = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_test.index)
    df_val = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_val.index)


    cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
    cols_leave = ['x4', 'x5', 'x6', 'x7']

    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]

    x_mapper = DataFrameMapper(standardize + leave)
    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_val = x_mapper.transform(df_val).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')


    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train = get_target(df_train)
    y_val = get_target(df_val)
    durations_test, events_test = get_target(df_test)
    val = x_val, y_val




    in_features = x_train.shape[1]
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                  dropout, output_bias=output_bias)

    model = CoxPH(net, torch.optim.Adam(net.parameters(), lr=0.01))

    batch_size = 256

    model.optimizer.set_lr(0.01)


    epochs = 512
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = True

    print(x_train.shape)
    x_train = torch.rand(1218,9)
    y_train_a = torch.rand(1218)
    y_train_b = torch.rand(1218)

    survival = (y_train_b, y_train_a)

    print(y_train[0].shape)

    log = model.fit(x_train, survival, batch_size, epochs, callbacks, verbose)


