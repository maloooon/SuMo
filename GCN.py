import numpy as np
import torch
import os
import DataInputNew
import ReadInData
import torchtuples as tt
import pandas as pd

def train(module,views, batch_size =25, n_epochs = 512, lr_scheduler_type = 'onecyclecos', l2_regularization = False):
    """

    :param module: basically the dataset to be used
    :param batch_size: batch size for training
    :param n_epochs: number of epochs
    :param lr_scheduler_type: type of lr_scheduler : 'lambda','mulitplicative','step','multistep','exponential',...
    :param l2_regularization : bool whether should be applied or not
    """




    # Setup all the data
    n_train_samples, n_test_samples, n_val_samples = module.setup()



    #Select method for feature selection
    module.feature_selection(method='pca') # Placeholder ; to get duration & event values

    # Load Dataloaders
    trainloader = module.train_dataloader(batch_size=n_train_samples) # all training examples
    testloader =module.test_dataloader(batch_size=n_test_samples)
    valloader = module.validation_dataloader(batch_size=n_val_samples)

    # Load data and set device to cuda if possible

    #Train
    for train_data, train_duration, train_event in trainloader:
        for view in range(len(train_data)):
            train_data[view] = train_data[view].to(device=device)

        train_duration.to(device=device)
        train_event.to(device=device)

    for c,_ in enumerate(train_data):
        print("Train data shape after feature selection {}".format(train_data[c].shape))

    #Validation
    for val_data, val_duration, val_event in valloader:
        for view in range(len(val_data)):
            val_data[view] = val_data[view].to(device=device)

        val_duration.to(device=device)
        val_event.to(device=device)

    for c,_ in enumerate(val_data):
        print("Validation data shape after feature selection {}".format(val_data[c].shape))

    #Test
    for test_data, test_duration, test_event in testloader:
        for view in range(len(test_data)):
            test_data[view] = test_data[view].to(device=device)


        test_duration.to(device=device)
        test_event.to(device=device)

    for c,_ in enumerate(test_data):
        print("Test data shape after feature selection {}".format(test_data[c].shape))



    # Input dimensions (features for each view) for NN based on different data (train/validation/test)
    # Need to be the same for NN to work
    dimensions_train = [x.size(1) for x in train_data]
    dimensions_val = [x.size(1) for x in val_data]
    dimensions_test = [x.size(1) for x in test_data]

    assert (dimensions_train == dimensions_val == dimensions_test), 'Feature mismatch between train/val/test'

    dimensions = dimensions_train

    # Get feature offsets for train/validation/test
    # Need to be the same for NN to work
    feature_offsets_train = [0] + np.cumsum(dimensions_train).tolist()
    feature_offsets_val = [0] + np.cumsum(dimensions_val).tolist()
    feature_offsets_test = [0] + np.cumsum(dimensions_test).tolist()

    feature_offsets = feature_offsets_train

    # Number of all features (summed up) for train/validation/test
    # These need to be the same, otherwise NN won't work
    feature_sum_train = feature_offsets_train[-1]
    feature_sum_val = feature_offsets_val[-1]
    feature_sum_test = feature_offsets_test[-1]

    feature_sum = feature_sum_train

    # Initialize empty tensors to store the data for train/validation/test
    train_data_pycox = torch.empty(n_train_samples, feature_sum_train).to(torch.float32)
    val_data_pycox = torch.empty(n_val_samples, feature_sum_val).to(torch.float32)
    test_data_pycox = torch.empty(n_test_samples, feature_sum_test).to(torch.float32)

    # Train
    for idx_view,view in enumerate(train_data):
        for idx_sample, sample in enumerate(view):
            train_data_pycox[idx_sample][feature_offsets_train[idx_view]:
                                         feature_offsets_train[idx_view+1]] = sample

    # Validation
    for idx_view,view in enumerate(val_data):
        for idx_sample, sample in enumerate(view):
            val_data_pycox[idx_sample][feature_offsets_val[idx_view]:
                                       feature_offsets_val[idx_view+1]] = sample

    # Test
    for idx_view,view in enumerate(test_data):
        for idx_sample, sample in enumerate(view):
            test_data_pycox[idx_sample][feature_offsets_test[idx_view]:
                                        feature_offsets_test[idx_view+1]] = sample


    # Turn validation (duration,event) in correct structure for pycox .fit() call
    # dde : data duration event ; de: duration event ; d : data
    train_duration_numpy = train_duration.detach().cpu().numpy()
    train_event_numpy = train_event.detach().cpu().numpy()
    val_duration_numpy = val_duration.detach().cpu().numpy()
    val_event_numpy = val_event.detach().cpu().numpy()



    train_de_pycox = (train_duration, train_event)
    val_dde_pycox = val_data_pycox, (val_duration, val_event)
    val_de_pycox = (val_duration, val_event)
    test_d_pycox = test_data_pycox

    train_data_pycox_numpy = train_data_pycox.detach().cpu().numpy()
    val_data_pycox_numpy = val_data_pycox.detach().cpu().numpy()
    train_de_pycox_numpy = (train_duration_numpy, train_event_numpy)#event_temporary_placeholder_train_numpy)
    val_de_pycox_numpy = (val_duration_numpy, val_event_numpy)
    train_ded_pycox = tt.tuplefy(train_de_pycox, train_data_pycox) # TODO : Problem hier wegen (221,) und (221,20) als shapes

    full_train = tt.tuplefy(train_data_pycox, (train_de_pycox, train_data_pycox))
    full_validation = tt.tuplefy(val_data_pycox, (val_de_pycox, val_data_pycox))



    train_data = pd.read_csv(
        os.path.join("/Users", "marlon", "Desktop", "Project", "TrainPPIPRAD.csv"), index_col=0)


    interaction_data = pd.read_csv(
        os.path.join("/Users", "marlon", "Desktop", "Project","interactions.csv"), index_col=0)


if __name__ == '__main__':
    cancer_data = ReadInData.readcancerdata()
    multimodule = DataInputNew.SurvMultiOmicsDataModule(cancer_data[0][0],cancer_data[0][1],cancer_data[0][2])
    views = cancer_data[0][2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(module= multimodule,views= views, l2_regularization=True)
