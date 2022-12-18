import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import DataInputNew
import pandas as pd
import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import sys

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

    print("test" , x_test)

    print("COX data: ", x_train)



    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train = get_target(df_train)
    y_val = get_target(df_val)
    durations_test, events_test = get_target(df_test)
    val = x_val, y_val
    print("COX survival", y_train)

    print("val", val)






#    print(x_train.shape)
    x_train_rand = torch.rand(398,9)
    y_train_a = torch.rand(398)
    y_train_b = torch.rand(398)

    survival_rand = (y_train_b, y_train_a)

  #  print(y_train[0].shape)



    module = DataInputNew.multimodule
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_train_samples, n_test_samples = module.setup()
    module.feature_selection(method='variance')
    trainloader = module.train_dataloader(batch_size=398) # all training examples

    # load just for size measures
    for data, duration, event in trainloader:
        train_data = data
        for view in range(len(train_data)):
            data[view] = data[view].to(device=device)

        duration = (duration.to(device=device)).numpy()
        event = (event.to(device=device)).numpy()
        event_tester = (torch.ones(398).to(torch.int32)).numpy()

        survival = (duration,event_tester)



    input_dim_per_view = [x.size(1) for x in train_data]

    # transforming data input for pycox : all views as one tensor, views accessible via feature offset
    feature_offsets = [0] + np.cumsum(input_dim_per_view).tolist()
    # train_data[0].size(0) --> number training samples
    train_data_all = torch.empty(train_data[0].size(0), feature_offsets[-1]).to(torch.float32) # -1 largest index --> gives number of all features together


    # TODO : chck if correct
    for idx_view,view in enumerate(train_data):
        for idx_sample, sample in enumerate(view):
            train_data_all[idx_sample][feature_offsets[idx_view]:feature_offsets[idx_view+1]] = sample

    x_train = train_data_all[:, 0:9]
    x_train = x_train.numpy()

    x_train = pd.DataFrame(x_train)
    x_train.to_csv(("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/PCAbig.csv"))


    x_train = train_data_all[:, 0:9]
    x_train = x_train.numpy()


    print("my data: {}".format(x_train))
    print("my survival : {}".format(survival))

    in_features = x_train_rand.size(1)
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                  dropout, output_bias=output_bias)

    model = CoxPH(net, torch.optim.Adam(net.parameters(), lr=0.01))

    batch_size = 50

    model.optimizer.set_lr(0.01)


    epochs = 512
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = True

   # print("random data : {}, shape : {} ".format(x_train_rand, x_train_rand.shape))
  #  print("random survival : {}, shape : {}".format(survival_rand,survival_rand[0].shape))
    log = model.fit(x_train, survival, batch_size, epochs, callbacks, verbose)



"""
(tensor([[-3.3927e+01, -7.7767e+00, -1.4200e+01,  ..., -6.0447e-01,
          1.4105e+00, -9.2582e-01],
        [-6.5399e+01,  3.6517e+01, -6.1904e+00,  ...,  1.3607e-01,
          4.1694e-01,  1.0941e+00],
        [-3.1344e+01,  2.2092e+01, -2.6459e+01,  ...,  2.7269e-01,
         -7.4791e-01, -4.6536e-03],
        ...,
        [ 1.2033e+01, -1.0276e+01, -2.4575e+00,  ...,  1.2573e+00,
         -2.4517e-01, -8.7801e-01],
        [-1.0189e+01, -1.1526e+01,  5.2632e+00,  ..., -4.1775e-10,
          8.6459e-10, -1.3901e-10],
        [-9.4241e+00,  2.6354e+01,  1.0013e+01,  ...,  2.6937e+00,
          5.6015e-02, -8.4216e-01]]), (tensor([1391.,  131., 1363., 1099.,  393.,  882.,  615.,  269.,  145., 1171.,
         636.,  400., 1518.,  889., 3467., 1815.,  543.,  384.,  787.,  589.,
        1099., 1127., 1280., 3716.,  771., 1226., 1434.,  775., 1383., 1855.,
         875.,  417., 1349.,  308.,  583., 1373.,  349., 1775.,  941.,  833.,
        1832.,  321.,  717., 1365.,  844.,  956.,  395.,  617., 1156.,  526.,
        1396.,  700., 1247., 1733., 2068., 2304.,  213.,  112.,  770., 1078.,
        2684.,  928.,  516.,  967.,  442., 2469.,  792., 1754.,  763., 1611.,
        1184., 1962., 3130.,  473.,  197.,  958., 1221.,  148.,  682.,  860.,
        1133.,  665.,   31.,  614., 1004.,  789.,  607., 1832.,   27.,  824.,
        1774., 1829.,  328.,  710.,   94.,  987., 1968., 3096., 1915., 1383.]), tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])))"""