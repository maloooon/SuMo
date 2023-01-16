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
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH

from pycox import models


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
        return phi, decoded, input

    def predict(self, input):
        # Will be used by model.predict later.
        # As this only has the survival output,
        # we don't have to change LogisticHazard.
        encoded = self.encoder(input)
        return self.surv_net(encoded)



in_features = x_train.shape[1]
encoded_features = 4
out_features = 1
net = NetAESurv(in_features, encoded_features, out_features)


class LossAELogHaz(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        assert (alpha >= 0) and (alpha <= 1), 'Need `alpha` in [0, 1].'
        self.alpha = alpha
        self.loss_surv = models.loss.CoxPHLoss()
        self.loss_ae = nn.MSELoss()

    def forward(self, phi, decoded,input, durations,events):
        loss_surv = self.loss_surv(phi, durations, events)
        loss_ae = self.loss_ae(decoded, input)
        return self.alpha * loss_surv + (1 - self.alpha) * loss_ae



loss = LossAELogHaz(0.6)


model = CoxPH(net, tt.optim.Adam(0.01), loss=loss)



epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True


log = model.fit(x_train, y_train, 30, epochs, callbacks, verbose)


_ = model.compute_baseline_hazards()


surv = model.predict_surv_df(x_test)
surv.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')