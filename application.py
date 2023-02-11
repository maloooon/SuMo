import pandas as pd
import FCNN
import AE
import GCN
import ReadInData
import DataInputNew
import torch
import numpy as np
import copy


if __name__ == '__main__':
    cancer_data = ReadInData.readcancerdata('LUAD')
    data = cancer_data[0][0]
    feature_offsets = cancer_data[0][1]
    view_names = cancer_data[0][2]
    feature_names = cancer_data[0][3]
    cancer_name = cancer_data[0][4][0]
    which_views = [] # no input to use all the given views
    n_folds = 1

    # needed for R, cant read in cancer name directly for some weird reason...
    with open('/Users/marlon/Desktop/Project/TCGAData/currentcancer.txt', 'w') as f:
        f.write(cancer_name)


    # Possible Cancers : 'DNA', 'microRNA' , 'mRNA', 'RPPA'
    # Leaving which_views empty will take all views into consideration
    multimodule = DataInputNew.SurvMultiOmicsDataModule(data,
                                                        feature_offsets,
                                                        view_names,
                                                        cancer_name= cancer_name,
                                                        which_views = which_views,
                                                        n_folds = n_folds,
                                                        type_preprocess= 'standardize')  ######################### basic preprocessing as in PyCox tutorial (works good with just FCNN and PCA, but not so well with ConcatAE)

    # kurze Einführung worum geht es, was sind die Ziele, was ist die Vorgehensweise

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup all the data
    # view_names_fix : RPPA data often is all 0s, so we don't read in that data
    n_train_samples, n_test_samples,n_val_samples, view_names_fix = multimodule.setup()



    with open('/Users/marlon/Desktop/Project/TCGAData/cancerviews.txt', 'w') as fp:
        for item in view_names_fix:
            # write each item on a new line
            fp.write("%s\n" % item)



    ############# AE FEATURE SELECTION HYPERPARAMETER OPTIMIZATION ##############

    method = 'GCN'


    if method == 'GCN':

        edge_index, proteins_used, train_data, val_data, test_data, \
        train_duration, train_event, \
        val_duration, val_event, \
        test_duration, test_event = multimodule.feature_selection('ppi', feature_names)

        for c_fold in range(n_folds): # TODO :optuna can be done on just the one fold
            # for Optuna we store prepared data and feature offsets and will load them with a function later on for each fold


            print("Train data has shape : {}".format(train_data[c_fold].shape))
            print("Validation data has shape : {}".format(val_data[c_fold].shape))
            print("Test data has shape : {}".format(test_data[c_fold].shape))


            views_with_proteins = train_data[c_fold].size(2)
            # Needed for GCN
            num_features = views_with_proteins
            num_nodes = len(proteins_used)


            feat_offs_train = [0]
            feat_offs_val = [0]
            feat_offs_test = [0]


            # data is 3-dimensional : [n_samples,n_proteins,n_features_per_protein]
            # to load data into a csv file, we need to reshape into 2-dimensional data

            # -1 in reshape for the first dimension is used so we just keep the samples :
            # but if we only have one feature per protein node, -1 leads to wrong reshaping,
            # thus we just use train_duration.size(0) (which is the number of samples for the train set)
            train_samples = train_duration[c_fold].size(0)
            val_samples = val_duration[c_fold].size(0)
            test_samples = test_duration.size(0)

            train_data[c_fold] = train_data[c_fold].reshape(train_samples,num_nodes * num_features)
            val_data[c_fold] = val_data[c_fold].reshape(val_samples,num_nodes * num_features)
            test_data[c_fold] = test_data[c_fold].reshape(test_samples, num_nodes * num_features)

            all_train_data = []

            all_train_data.append(train_data[c_fold])
            all_train_data.append(train_duration[c_fold].unsqueeze(1))
            all_train_data.append(train_event[c_fold].unsqueeze(1))
            # all_train_data is now a list containing data for all views, the durations and the events
            # we can get the feature offsets by accessing dimension 1 of each

            for idx, _ in enumerate(all_train_data):
                feat_offs_train.append(all_train_data[idx].size(1))
            train_data_c = torch.cat(tuple(all_train_data), dim=1)
            train_data_df = pd.DataFrame(train_data_c)
            feat_offs_train = np.cumsum(feat_offs_train)
            feat_offs_train_df = pd.DataFrame(feat_offs_train)
            train_data_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/TrainData.csv")
            feat_offs_train_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/TrainDataFeatOffs.csv")


            all_val_data = []
            all_val_data.append(val_data[c_fold])
            all_val_data.append(val_duration[c_fold].unsqueeze(1))
            all_val_data.append(val_event[c_fold].unsqueeze(1))
            for idx, _ in enumerate(all_val_data):
                feat_offs_val.append(all_val_data[idx].size(1))
            val_data_c = torch.cat(tuple(all_val_data), dim=1)
            val_data_df = pd.DataFrame(val_data_c)
            feat_offs_val = np.cumsum(feat_offs_val)
            feat_offs_val_df = pd.DataFrame(feat_offs_val)
            val_data_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/ValData.csv")
            feat_offs_val_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/ValDataFeatOffs.csv")




            all_test_data = []
            all_test_data.append(test_data[c_fold])
            all_test_data.append(test_duration.unsqueeze(1))
            all_test_data.append(test_event.unsqueeze(1))
            for idx, _ in enumerate(all_test_data):
                feat_offs_test.append(all_test_data[idx].size(1))
            test_data_c = torch.cat(tuple(all_test_data), dim=1)
            test_data_df = pd.DataFrame(test_data_c)
            feat_offs_test = np.cumsum(feat_offs_test)
            feat_offs_test_df = pd.DataFrame(feat_offs_test)
            test_data_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/TestData.csv")
            feat_offs_test_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/TestDataFeatOffs.csv")



            # also load number of features per node & number of nodes used aswell as edge indices



            with open('/Users/marlon/Desktop/Project/PreparedData/num_features.txt', 'w') as f:
                f.write(str(num_features))
            with open('/Users/marlon/Desktop/Project/PreparedData/num_nodes.txt', 'w') as f:
                f.write(str(num_nodes))
            with open('/Users/marlon/Desktop/Project/PreparedData/edge_index_1.txt', 'w') as f:
                f.write(','.join(str(i) for i in edge_index[0]))
            with open('/Users/marlon/Desktop/Project/PreparedData/edge_index_2.txt', 'w') as f:
                    f.write(','.join(str(i) for i in edge_index[1]))


        GCN.optuna_optimization()




    elif method == 'FCNN' or method == 'AE': # for GCN, we always have PPI feature selection
        feature_select_method = 'pca'
        components = [100,100,100,100]
        thresholds = [0.8,0.8,0.8,0.8]



        train_data, val_data, test_data, \
        train_duration, train_event, \
        val_duration, val_event, \
        test_duration, test_event = multimodule.feature_selection(method=feature_select_method,
                                                                  components= components,
                                                                  thresholds= thresholds,
                                                                  feature_names= feature_names)



        for c_fold in range(n_folds): # TODO :optuna can be done on just the one fold
            # for Optuna we store prepared data and feature offsets and will load them with a function later on for each fold

            for c_view, view in enumerate(view_names_fix):
                print("Train data has shape : {} for view {}".format(train_data[c_fold][c_view].shape, view_names_fix[c_view]))
                print("Validation data has shape : {} for view {}".format(val_data[c_fold][c_view].shape, view_names_fix[c_view]))
                print("Test data has shape : {} for view {}".format(test_data[c_fold][c_view].shape, view_names_fix[c_view]))

            feat_offs_train = [0] # TODO : need to be the same size over train/val/test : reicht hier nur 1 x zu rechnen
            feat_offs_val = [0]
            feat_offs_test = [0]
            all_train_data = copy.deepcopy(train_data[c_fold])

            all_train_data.append(train_duration[c_fold].unsqueeze(1))
            all_train_data.append(train_event[c_fold].unsqueeze(1))
            # all_train_data is now a list containing data for all views, the durations and the events
            # we can get the feature offsets by accessing dimension 1 of each
            for idx, _ in enumerate(all_train_data):
                feat_offs_train.append(all_train_data[idx].size(1))
            train_data_c = torch.cat(tuple(all_train_data), dim=1)
            train_data_df = pd.DataFrame(train_data_c)
            feat_offs_train = np.cumsum(feat_offs_train)
            feat_offs_train_df = pd.DataFrame(feat_offs_train)
            train_data_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/TrainData.csv")
            feat_offs_train_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/TrainDataFeatOffs.csv")

            all_val_data = copy.deepcopy(val_data[c_fold])
            all_val_data.append(val_duration[c_fold].unsqueeze(1))
            all_val_data.append(val_event[c_fold].unsqueeze(1))
            for idx, _ in enumerate(all_val_data):
                feat_offs_val.append(all_val_data[idx].size(1))
            val_data_c = torch.cat(tuple(all_val_data), dim=1)
            val_data_df = pd.DataFrame(val_data_c)
            feat_offs_val = np.cumsum(feat_offs_val)
            feat_offs_val_df = pd.DataFrame(feat_offs_val)
            val_data_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/ValData.csv")
            feat_offs_val_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/ValDataFeatOffs.csv")


            all_test_data = copy.deepcopy(test_data[c_fold])
            all_test_data.append(test_duration.unsqueeze(1))
            all_test_data.append(test_event.unsqueeze(1))
            for idx, _ in enumerate(all_test_data):
                feat_offs_test.append(all_test_data[idx].size(1))
            test_data_c = torch.cat(tuple(all_test_data), dim=1)
            test_data_df = pd.DataFrame(test_data_c)
            feat_offs_test = np.cumsum(feat_offs_test)
            feat_offs_test_df = pd.DataFrame(feat_offs_test)
            test_data_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/TestData.csv")
            feat_offs_test_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/TestDataFeatOffs.csv")


        if method == 'FCNN':
            FCNN.optuna_optimization()
        elif method == 'AE':
            AE.optuna_optimization()




    ######## SET OWN SETTINGS FOR NN CALL ############

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
    dropout_layers_NN = []
    # BATCH NORMALIZATION SETTINGS
    batchnorm_bool_NN = False
    batchnorm_layers_NN = []
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

    print("######################## RUNNING FULLY CONNECTED NEURAL NET ####################################")
 #   NN.train(train_data, val_data, test_data, train_duration, train_event, val_duration, val_event, test_duration,test_event,
 #         batch_size=batch_size_NN,
 #         n_epochs=n_epochs_NN,
 #         learning_rate= learning_rate_NN,
 #         l2_regularization=l2_regularization_bool_NN,
 #         l2_regularization_rate=l2_regularization_rate_NN,
 #         val_batch_size=val_batch_size_NN,
 #         dropout=dropout_bool_NN,
 #         dropout_rate=dropout_rate_NN,
 #         batchnorm = batchnorm_bool_NN,
 #         activation_layers = activations_NN,
 #         batchnorm_layers = batchnorm_layers_NN,
 #         dropout_layers = dropout_layers_NN,
 #         view_names = view_names_fix)


    print("######################## FULLY CONNECTED NEURAL NET FINISHED ####################################")

    print("######################## RUNNING GCN ####################################")


#    GCN.train(module= multimodule,
#              device=device,
#              batch_size=128,
#              n_epochs=100,
#              l2_regularization=False,
#              val_batch_size=32,
#              number_folds=2,
#             feature_names=feature_names,
#              n_train_samples = n_train_samples,
#              n_test_samples = n_test_samples,
#              n_val_samples = n_val_samples,
#              view_names = view_names,
#              processing_bool = False)



    print("######################## GCN FINISHED ####################################")

    print("######################## RUNNING AUTOENCODER ####################################")

    # LAYERS
    layers_AE = [[64,32]]
    # AE SETTINGS
    activations_AE = [['relu'] for i in range(len(view_names_fix))]
    # DROPOUT SETTINGS
    dropout_bool_AE = False
    dropout_rate_AE = 0.1
    dropout_layers_AE = [['yes' for _ in range(len(a))] for a in layers_AE]
    # BATCH NORMALIZATION SETTINGS
    batchnorm_bool_AE = False
    batchnorm_layers_AE = [['yes' for _ in range(len(a))] for a in layers_AE]
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


#    AE.train(train_data, val_data, test_data, train_duration, train_event, val_duration, val_event, test_duration,test_event,
#      batch_size=batch_size_AE,
#      n_epochs=n_epochs_AE,
#      learning_rate= learning_rate_AE,
#      l2_regularization=l2_regularization_bool_AE,
#      l2_regularization_rate=l2_regularization_rate_AE,
#      val_batch_size=val_batch_size_AE,
#      dropout=dropout_bool_AE,
#      dropout_rate=dropout_rate_AE,
#      batchnorm = batchnorm_bool_AE,
#      activation_layers = activations_AE,
#      batchnorm_layers = batchnorm_layers_AE,
#      dropout_layers = dropout_layers_AE,
#      view_names = view_names_fix,
#      layers = layers_AE)


    print("######################## AUTOENCODER FINISHED ####################################")

    
