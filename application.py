import pandas as pd
import NN
import AE
import GCN
import ReadInData
import DataInputNew
import torch
import numpy as np



if __name__ == '__main__':
    cancer_data = ReadInData.readcancerdata('KICH')
    data = cancer_data[0][0]
    feature_offsets = cancer_data[0][1]
    view_names = cancer_data[0][2]
    feature_names = cancer_data[0][3]
    cancer_name = cancer_data[0][4][0]
    which_views = ['microRNA'] # no input to use all the given views
    n_folds = 1

    # needed for R, cant read in cancer name directly for some weird reason...
    with open('/Users/marlon/Desktop/Project/TCGAData/currentcancer.txt', 'w') as f:
        f.write(cancer_name)







    # Possible Cancers : 'DNA', 'microRNA' , 'mRNA', 'RPPA'
    # Leaving which_views empty will take all views into consideration
    multimodule = DataInputNew.SurvMultiOmicsDataModule(data,
                                                        feature_offsets,
                                                        view_names,
                                                        onezeronorm_bool=False, ########################## Preprocessing : One-Zero-Normalization (for Cross/ConcatAE)
                                                        cancer_name= cancer_name,
                                                        which_views = which_views,
                                                        n_folds = n_folds,
                                                        preprocess_bool = True)  ######################### basic preprocessing as in PyCox tutorial (works good with just FCNN and PCA, but not so well with ConcatAE)

    # kurze Einführung worum geht es, was sind die Ziele, was ist die Vorgehensweise

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup all the data
    # view_names_fix : RPPA data often is all 0s, so we don't read in that data
    n_train_samples, n_test_samples,n_val_samples, view_names_fix = multimodule.setup()



    with open('/Users/marlon/Desktop/Project/TCGAData/cancerviews.txt', 'w') as fp:
        for item in view_names_fix:
            # write each item on a new line
            fp.write("%s\n" % item)



    # Grid search config

    config = {
        "Layers_mRNA" : [[64,32,16], [32,16,8], [16,8,4,2]],
        "Layers_DNA" : [[64,32,16], [32,16,8], [16,8,4,2]],
        "Layers_microRNA" : [[64,32,16], [32,16,8], [16,8,4,2]],
        "Layers_RPPA" : [[64,32,16], [32,16,8], [16,8,4,2]],
        "BatchSize" : [150,128,100,75,64,50],
        "BatchSizeVal" : [100,75,64,50,40,32],
        "LearningRate" : [0.001,0.0008,0.0006,0.0004,0.0001,0.00005,0.00001],
        "DropoutBool" : ['no'], # trying without dropout
        "BatchNormBool" : ['no'], # trying without batch norm
        "NNLayersConcat" : [[64,32,16], [16,8,4,2]],
        "ConcatAELoss" : [0.4,0.5,0.6]

    }

    LUAD_PCA_FCNN_config = {
        "Layers_mRNA" : [[1024], [64], [16]],
        "Layers_DNA" : [[512], [128], [32]],
        "Layers_microRNA" : [[1024], [256], [16]],
        "Layers_RPPA" : [[1024,512,256], [256, 128, 64], [64,32,16]],
        "BatchSize" : [75],
        "BatchSizeVal" : [32],
        "LearningRate" : [0.0006],
        "DropoutBool" : ['no'], # trying without dropout
        "BatchNormBool" : ['no'], # trying without batch norm
        "NNLayersConcat" : [[256,128,64], [64,32,16], [16,8,4,2]],
        "ConcatAELoss" : [0.2,0.3,0.4,0.5,0.6,0.7,0.8]

    }


    feature_select_method = 'pca'
    components = [35,35,35,35]
    thresholds = [0.8,0.8,0.8,0.8]



    train_data, val_data, test_data, \
    train_duration, train_event, \
    val_duration, val_event, \
    test_duration, test_event = multimodule.feature_selection(method=feature_select_method,
                                                              components= components,
                                                              thresholds= thresholds,
                                                              feature_names= feature_names)



    # for Optuna we store prepared data and feature offsets and will load them with a function later on for each fold

    for c_fold in range(n_folds):
        feat_offs_train = [0]
        all_train_data = train_data[c_fold]

        all_train_data.append(train_duration[c_fold].unsqueeze(1))
        all_train_data.append(train_event[c_fold].unsqueeze(1))
        # all_train_data is now a list containing data for all views, the durations and the events
        # we can get the feature offsets by accessing dimension 1 of each
        for idx, _ in enumerate(all_train_data):
            feat_offs_train.append(all_train_data[idx].size(1))
        train_data = torch.cat(tuple(all_train_data), dim=1)
        train_data_df = pd.DataFrame(train_data)
        feat_offs_train = np.cumsum(feat_offs_train)
        feat_offs_train_df = pd.DataFrame(feat_offs_train)
        train_data_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/TrainData.csv")
        feat_offs_train_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/TrainDataFeatOffs.csv")

        feat_offs_val = []
        all_val_data = val_data[c_fold]
        all_val_data.append(val_duration[c_fold].unsqueeze(1))
        all_val_data.append(val_event[c_fold].unsqueeze(1))
        for idx, _ in enumerate(all_val_data):
            feat_offs_val.append(all_val_data[idx].size(1))
        val_data = torch.cat(tuple(all_val_data), dim=1)
        val_data_df = pd.DataFrame(val_data)
        feat_offs_val = np.cumsum(feat_offs_val)
        feat_offs_val_df = pd.DataFrame(feat_offs_val)
        val_data_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/ValData.csv")
        feat_offs_val_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/ValDataFeatOffs.csv")

        feat_offs_test = []
        all_test_data = test_data[c_fold]
        all_test_data.append(test_duration.unsqueeze(1))
        all_test_data.append(test_event.unsqueeze(1))
        for idx, _ in enumerate(all_test_data):
            feat_offs_test.append(all_test_data[idx].size(1))
        test_data = torch.cat(tuple(all_test_data), dim=1)
        test_data_df = pd.DataFrame(test_data)
        feat_offs_test = np.cumsum(feat_offs_test)
        feat_offs_test_df = pd.DataFrame(feat_offs_test)
        test_data_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/TestData.csv")
        feat_offs_test_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/TestDataFeatOffs.csv")




        NN.optuna_optimization()







    print("######################## RUNNING FULLY CONNECTED NEURAL NET ####################################")
    # FEATURE SELECTION SETTINGS
    selection_method_NN = 'pca'
    components_PCA_NN = [98,98,98,98]
    thresholds_VARIANCE_NN = [0.8,0.8,0.8,0.8]






        





 #   train,val,test = NN.load_data(n_fold=1)




    # FULLY CONNECTED NEURAL NET SETTINGS
    # LAYER SETTINGS
    # ACTIVATION FUNCTIONS SETTINGS
    activations_layers_NN = [['relu'] for i in range(len(view_names_fix))]
    activation_last_layer_NN = ['none']
    activations_layers_NN.append(activation_last_layer_NN)
    activations_NN = activations_layers_NN
    # DROPOUT SETTINGS
    dropout_bool_NN = False
    dropout_rate_NN = 0.1
    dropout_layers_NN = [['yes' for _ in range(len(config["Layers_mRNA"][0]))] for i in range(len(view_names_fix))]
    # BATCH NORMALIZATION SETTINGS
    batchnorm_bool_NN = False
    batchnorm_layers_NN = [['yes' for _ in range(len(config["Layers_mRNA"][0]))] for i in range(len(view_names_fix))]
    # L2 REGULARIZATION SETTINGS
    l2_regularization_bool_NN = False
    l2_regularization_rate_NN = 0.000001
    # BATCH SIZE SETTINGS
    batch_size_NN = 64
    val_batch_size_NN = 16
    # EPOCH SETTINGS
    n_epochs_NN = 100
    # LEARNING RATE OPTIMIZER (ADAM)
    learning_rate_NN = 0.005





#    NN.train(module= multimodule,
#          feature_select_method= selection_method_NN,
#          components = components_PCA_NN,
#          thresholds= thresholds_VARIANCE_NN,
#          feature_names= None,
#          batch_size=batch_size_NN,
#          n_epochs=n_epochs_NN,
#          learning_rate= learning_rate_NN,
#          l2_regularization=l2_regularization_bool_NN,
#          l2_regularization_rate=l2_regularization_rate_NN,
#          val_batch_size=val_batch_size_NN,
#          dropout=dropout_bool_NN,
#          dropout_rate=dropout_rate_NN,
#          batchnorm = batchnorm_bool_NN,
#          activation_layers = activations_NN,
#          batchnorm_layers = batchnorm_layers_NN,
#          dropout_layers = dropout_layers_NN,
#          view_names = view_names_fix,
#          config=config,
#          n_grid_search_iterations= 50,
#          testing_config = LUAD_PCA_FCNN_config)


    print("######################## FULLY CONNECTED NEURAL NET FINISHED ####################################")

    print("######################## RUNNING GCN ####################################")


#    GCN.train(module= multimodule,
#              device=device,
#              batch_size=128,
#              n_epochs=100,
#              l2_regularization=False,
#              val_batch_size=32,
#              number_folds=2,
#              feature_names=feature_names,
#              n_train_samples = n_train_samples,
#              n_test_samples = n_test_samples,
#              n_val_samples = n_val_samples,
#              view_names = view_names,
#              processing_bool = False)



    print("######################## GCN FINISHED ####################################")

    print("######################## RUNNING AUTOENCODER ####################################")


    # AE SETTINGS
    activations_AE = [['relu'] for i in range(len(view_names_fix))]
    # DROPOUT SETTINGS
    dropout_bool_AE = False
    dropout_rate_AE = 0.1
    dropout_layers_AE = [['yes' for _ in range(len(config["Layers_mRNA"][0]))] for i in range(len(view_names_fix))]
    # BATCH NORMALIZATION SETTINGS
    batchnorm_bool_AE = False
    batchnorm_layers_AE = [['yes' for _ in range(len(config["Layers_mRNA"][0]))] for i in range(len(view_names_fix))]
    # L2 REGULARIZATION SETTINGS
    l2_regularization_bool_AE = False
    l2_regularization_rate_AE = 0.000001
    # BATCH SIZE SETTINGS
    batch_size_AE = 64
    val_batch_size_AE = 16
    # EPOCH SETTINGS
    n_epochs_AE = 100
    # LEARNING RATE OPTIMIZER (ADAM)
    learning_rate_AE = 0.005


    AE.train(module= multimodule,
      feature_select_method= selection_method_NN,
      components = components_PCA_NN,
      thresholds= thresholds_VARIANCE_NN,
      feature_names= None,
      batch_size=batch_size_AE,
      n_epochs=n_epochs_AE,
      learning_rate= learning_rate_AE,
      l2_regularization=l2_regularization_bool_AE,
      l2_regularization_rate=l2_regularization_rate_AE,
      val_batch_size=val_batch_size_AE,
      dropout=dropout_bool_AE,
      dropout_rate=dropout_rate_AE,
      batchnorm = batchnorm_bool_AE,
      activation_layers = activations_AE,
      batchnorm_layers = batchnorm_layers_AE,
      dropout_layers = dropout_layers_AE,
      view_names = view_names_fix,
      config=config,
      n_grid_search_iterations= 10)






    print("######################## AUTOENCODER FINISHED ####################################")

    
