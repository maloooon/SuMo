import numpy as np
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch # For building the networks
from torch import nn
import torch.nn.functional as F
import torchtuples as tt # Some useful functions

from pycox.datasets import metabric
from pycox.models import LogisticHazard
from pycox.models.loss import NLLLogistiHazardLoss
from pycox import models



class LossAELogHaz(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        assert (alpha >= 0) and (alpha <= 1), 'Need `alpha` in [0, 1].'
        self.alpha = alpha
        self.loss_surv = models.loss.CoxPHLoss()
        self.loss_ae = nn.MSELoss()

    def forward(self, phi, decoded, target_loghaz, target_ae):
        idx_durations, events = target_loghaz
        loss_surv = self.loss_surv(phi, idx_durations, events)
        loss_ae = self.loss_ae(decoded, target_ae)
        return self.alpha * loss_surv + (1 - self.alpha) * loss_ae

class NetAESurv(nn.Module):
    def __init__(self, in_features, encoded_features, out_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, encoded_features),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoded_features, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, in_features),
        )
        self.surv_net = nn.Sequential(
            nn.Linear(encoded_features, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, out_features),
        )

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        phi = self.surv_net(encoded)
        return phi, decoded



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


    num_durations = 10
    labtrans = LogisticHazard.label_transform(num_durations)
    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train_surv = labtrans.fit_transform(*get_target(df_train))
    y_val_surv = labtrans.transform(*get_target(df_val))

    train = tt.tuplefy(x_train, (y_train_surv, x_train))
    val = tt.tuplefy(x_val, (y_val_surv, x_val))

    # We don't need to transform the test labels
    durations_test, events_test = get_target(df_test)



    in_features = x_train.shape[1]
    encoded_features = 4
    out_features = labtrans.out_features
    print(out_features)
    net = NetAESurv(in_features, encoded_features, out_features)


    loss = LossAELogHaz(0.6)

    model = models.CoxPH(net,tt.optim.Adam(0.01), loss=LossAELogHaz(0.6))


    batch_size = 256
    epochs = 100
    log = model.fit(*train, batch_size, epochs)



"""

    #TODO : working ?
    if lr_scheduler_type.lower() == 'lambda':
        lambda1 = lambda epoch: 0.65 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda1)



    if lr_scheduler_type.lower() == 'onecyclecos':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10)



    lrs = []

    for i in range(10):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])

        scheduler.step()
"""