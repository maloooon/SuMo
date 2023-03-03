import pandas as pd
import FCNN
import AE
import GCN
import ReadInData
import DataProcessing
import torch
import numpy as np
import copy
import os


if __name__ == '__main__':
    """ 
    Main function ; Decide cancer and views to analyze, number of folds for cross validation, preprocessing type,
    whether to apply hyperparameter optimization, which neural network method.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on device : ", device)


    # 4 views STAD,LUAD,LAML,KIRC,KIRP,LIHC,LUSC,MESO,UVM
    # 2 views ACC, BRCA, CHOL,DLBC, ESCA, GBM, KICH, KIRC, PAAD, PCPG, READ, TGCT, THYM, UCEC, UCS (miRNA,RPPA)
    # 1 view BLCA, CESC, COAD, HNSC, LGG, SARC, SKCM, THCA (RPPA)

    # Define Cancer to load
    # Possible Cancers are :
    #PRAD, ACC, BLCA, BRCA,CESC,CHOL,COAD,DLBC,ESCA,GBM, HNSC,KICH,KIRC,KIRP,LAML,LGG,
    #LIHC,LUAD,LUSC,MESO,PAAD,PCPG,READ,SARC,SKCM,STAD,TGCT,THCA,THYM,UCEC,UCS,UVM
    cancer_data = ReadInData.readcancerdata('KIRC')
    data = cancer_data[0][0]
    feature_offsets = cancer_data[0][1]
    view_names = cancer_data[0][2]
    feature_names = cancer_data[0][3]
    cancer_name = cancer_data[0][4][0]



    # Decide which views to use
    # Cancers can have mRNA, DNA, microRNA, RPPA data
    # Leaving which_views empty will take all views into consideration will take all possible views into consideration
    # If you want to choose specific cancers, put them in the right order : mRNA, DNA, microRNA, RPPA
    which_views = ['mRNA','microRNA','RPPA']
    # Decide number of folds for cross validation. For Optuna optimization, use just one fold.
    n_folds = 5 # Use 1 for Optuna Hyperparameter Optimization
    # Decide type of preprocessing (standardize/minmaxscaling (between 0 & 1)/robustscaling (good when we have outliers)/maxabsvalscaling (between -1 and 1)
    type_of_preprocessing = 'maxabsvalscaling'
    # Hyperparameter Optimization NN method
    method_tune = 'none'
    # Train NN method
    method_train = 'AE'
    # Feature selection method hyperparamter Tuning FCNN/AE (For GCN its always PPI)
    selection_method_tuning = 'eigengenes'
    components_tuning = [50,30,50,50]
    thresholds_tuning = [0.1,0.1,0.1,0.1]
    # Choose feature selection method (PCA,Variance,AE,Eigengenes)
    selection_method_train = 'PCA'
    components_train = [100,100,100,100]
    thresholds_train = [0.1,0.1,0.1,0.1]

    print("Preprocessing Type : ", type_of_preprocessing)
    print("Tuning Neural Network : ", method_tune)
    print("Feature Selection Method Tuning: " ,selection_method_tuning)

    direc_set = 'Desktop' # dir is Desktop for own CPU or SUMO for GPU


    dir = os.path.expanduser('~/{}/Project/TCGAData/currentcancer.txt'.format(direc_set)) # for mlo gpu
  #  dir = os.path.expanduser('/Users/marlon/Desktop/Project/TCGAData/currentcancer.txt')
    with open(dir, 'w') as f:
        f.write(cancer_name)




    # Call the module
    multimodule = DataProcessing.SurvMultiOmicsDataModule(data,
                                                        feature_offsets,
                                                        view_names,
                                                        cancer_name= cancer_name,
                                                        which_views = which_views,
                                                        n_folds = n_folds,
                                                        type_preprocess= type_of_preprocessing)





    # Setup all the data
    n_train_samples, n_test_samples,n_val_samples, view_names_fix = multimodule.setup()    ###### TESTING ON SAME FOLD #########


    ######## TESTING PURPOSES #######
   # view_names_fix = ['MICRORNA','RPPA']
    ######## TESTING PURPOSES #########
    # Write used views for input into NN to .txt

    dir = os.path.expanduser('~/{}/Project/TCGAData/cancerviews.txt'.format(direc_set))
    with open(dir, 'w') as fp:
        for item in view_names_fix:
            # write each item on a new line
            fp.write("%s\n" % item)

    ############################################ HYPERPARAMETER OPTIMIZATION ##########################################

    if method_tune == 'GCN':

        # PPI feature selection
        edge_index, proteins_used, train_data, val_data, test_data, \
        train_duration, train_event, \
        val_duration, val_event, \
        test_duration, test_event = multimodule.feature_selection('ppi', feature_names)

        # We go through each fold (but in Optuna optimization, we only produce one fold, so this loop is not necessary
        # per se (just easier to read and if hyperparameter optimization was to be implemented with multiple folds,
        # one can build code based on this)
        for c_fold in range(n_folds):
            print("For Fold {}".format(c_fold))
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
            dir = os.path.expanduser('~/{}/Project/PreparedData/TrainData_{}.csv'.format(direc_set,c_fold))
          #  dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/TrainData.csv")
            train_data_df.to_csv(dir)
            dir = os.path.expanduser('~/{}/Project/PreparedData/TrainDataFeatOffs_{}.csv'.format(direc_set,c_fold))
          #  dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/TrainDataFeatOffs.csv")
            feat_offs_train_df.to_csv(dir)

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
            dir = os.path.expanduser('~/{}/Project/PreparedData/ValData_{}.csv'.format(direc_set,c_fold))
          #  dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/ValData.csv")
            val_data_df.to_csv(dir)
            dir = os.path.expanduser('~/{}/Project/PreparedData/ValDataFeatOffs_{}.csv'.format(direc_set,c_fold))
           # dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/ValDataFeatOffs.csv")
            feat_offs_val_df.to_csv(dir)


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
            dir = os.path.expanduser('~/{}/Project/PreparedData/TestData_{}.csv'.format(direc_set,c_fold))
            #dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/TestData.csv")
            test_data_df.to_csv(dir)
            dir = os.path.expanduser('~/{}/Project/PreparedData/TestDataFeatOffs_{}.csv'.format(direc_set,c_fold))
          #  dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/TestDataFeatOffs.csv")
            feat_offs_test_df.to_csv(dir)


            dir = os.path.expanduser('~/{}/Project/PreparedData/num_features.txt'.format(direc_set))
          #  dir = os.path.expanduser('/Users/marlon/Desktop/Project/PreparedData/num_features.txt')
            with open(dir, 'w') as f:
                f.write(str(num_features))
            dir = os.path.expanduser('~/{}/Project/PreparedData/num_nodes.txt'.format(direc_set))
          #  dir = os.path.expanduser('/Users/marlon/Desktop/Project/PreparedData/num_nodes.txt')
            with open(dir, 'w') as f:
                f.write(str(num_nodes))
            dir = os.path.expanduser('~/{}/Project/PreparedData/edge_index_1.txt'.format(direc_set))
          #  dir = os.path.expanduser('/Users/marlon/Desktop/Project/PreparedData/edge_index_1.txt')
            with open(dir, 'w') as f:
                f.write(','.join(str(i) for i in edge_index[0]))
            dir = os.path.expanduser('~/{}/Project/PreparedData/edge_index_2.txt'.format(direc_set))
           # dir = os.path.expanduser('/Users/marlon/Desktop/Project/PreparedData/edge_index_2.txt')
            with open(dir, 'w') as f:
                f.write(','.join(str(i) for i in edge_index[1]))


        GCN.optuna_optimization()


    elif method_tune == 'FCNN' or method_tune == 'AE':

      #  DataProcessing.optuna_optimization()   # AE FEATURE SELECTION OPTIMIZATION

        train_data, val_data, test_data, \
        train_duration, train_event, \
        val_duration, val_event, \
        test_duration, test_event = multimodule.feature_selection(method=selection_method_tuning,
                                                                  components= components_tuning,
                                                                  thresholds= thresholds_tuning,
                                                                  feature_names= feature_names)


        for c_fold in range(n_folds):
            print("For Fold {}".format(c_fold))

            for c_view, view in enumerate(view_names_fix):
                print("Train data has shape : {} for view {}".format(train_data[c_fold][c_view].shape, view_names_fix[c_view]))
                print("Validation data has shape : {} for view {}".format(val_data[c_fold][c_view].shape, view_names_fix[c_view]))
                print("Test data has shape : {} for view {}".format(test_data[c_fold][c_view].shape, view_names_fix[c_view]))

            feat_offs_train = [0]
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

            dir = os.path.expanduser('~/{}/Project/PreparedData/TrainData_{}.csv'.format(direc_set,c_fold))
        #    dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/TrainData_{}.csv".format(c_fold))
            train_data_df.to_csv(dir)
            dir = os.path.expanduser('~/{}/Project/PreparedData/TrainDataFeatOffs_{}.csv'.format(direc_set,c_fold))
         #   dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/TrainDataFeatOffs_{}.csv".format(c_fold))
            feat_offs_train_df.to_csv(dir)


            all_val_data = copy.deepcopy(val_data[c_fold])
            all_val_data.append(val_duration[c_fold].unsqueeze(1))
            all_val_data.append(val_event[c_fold].unsqueeze(1))
            for idx, _ in enumerate(all_val_data):
                feat_offs_val.append(all_val_data[idx].size(1))
            val_data_c = torch.cat(tuple(all_val_data), dim=1)
            val_data_df = pd.DataFrame(val_data_c)
            feat_offs_val = np.cumsum(feat_offs_val)
            feat_offs_val_df = pd.DataFrame(feat_offs_val)

            dir = os.path.expanduser('~/{}/Project/PreparedData/ValData_{}.csv'.format(direc_set,c_fold))
            val_data_df.to_csv(dir)




            all_test_data = copy.deepcopy(test_data[c_fold])
            all_test_data.append(test_duration.unsqueeze(1))
            all_test_data.append(test_event.unsqueeze(1))
            for idx, _ in enumerate(all_test_data):
                feat_offs_test.append(all_test_data[idx].size(1))
            test_data_c = torch.cat(tuple(all_test_data), dim=1)
            test_data_df = pd.DataFrame(test_data_c)
            feat_offs_test = np.cumsum(feat_offs_test)
            feat_offs_test_df = pd.DataFrame(feat_offs_test)

            dir = os.path.expanduser('~/{}/Project/PreparedData/TestData_{}.csv'.format(direc_set,c_fold))
            test_data_df.to_csv(dir)





        if method_tune == 'FCNN':
            FCNN.optuna_optimization()
        elif method_tune == 'AE':
            AE.optuna_optimization()




    ###################################### SET OWN SETTINGS FOR NN CALL #############################################

    params={'l2_regularization_bool': True, 'learning_rate': 0.0005078153954199322,
            'l2_regularization_rate': 4.455144035548901e-05, 'batch_size': 16, 'dropout_prob': 0.1,
            'dropout_bool': True, 'batchnorm_bool': False, 'prelu_rate': 0.30000000000000004,
            'dropout_prob_hierachical': 0.1, 'dropout_bool_hierachical': True, 'batchnorm_bool_hierachical': True,
            'loss_surv': 0.657402294726867, 'layers_1_mRNA': 32, 'layers_2_mRNA': 32, 'layers_1_mRNA_activfunc': 'prelu',
            'layers_2_mRNA_activfunc': 'sigmoid', 'layers_1_mRNA_dropout': 'yes', 'layers_2_mRNA_dropout': 'no',
            'layers_1_mRNA_batchnorm': 'yes', 'layers_2_mRNA_batchnorm': 'no', 'layers_1_mRNA_hierarichcal': 32,
            'layers_2_mRNA_hierarichcal': 27, 'layers_1_mRNA_activfunc_hierarichcal': 'prelu',
            'layers_2_mRNA_activfunc_hierarichcal': 'relu', 'layers_1_mRNA_dropout_hierarichcal': 'yes',
            'layers_2_mRNA_dropout_hierarichcal': 'yes', 'layers_1_mRNA_batchnorm_hierarichcal': 'no',
            'layers_2_mRNA_batchnorm_hierarichcal': 'no', 'layers_1_microRNA': 52, 'layers_2_microRNA': 32,
            'layers_1_microRNA_activfunc': 'prelu', 'layers_2_microRNA_activfunc': 'sigmoid', 'layers_1_microRNA_dropout': 'no',
            'layers_2_microRNA_dropout': 'yes', 'layers_1_microRNA_batchnorm': 'yes', 'layers_2_microRNA_batchnorm': 'yes',
            'layers_1_microRNA_hierarichcal': 64, 'layers_2_microRNA_hierarichcal': 32,
            'layers_1_microRNA_activfunc_hierarichcal': 'prelu', 'layers_2_microRNA_activfunc_hierarichcal': 'prelu',
            'layers_1_microRNA_dropout_hierarichcal': 'no', 'layers_2_microRNA_dropout_hierarichcal': 'no',
            'layers_1_microRNA_batchnorm_hierarichcal': 'yes', 'layers_2_microRNA_batchnorm_hierarichcal': 'no',
            'layers_1_RPPA': 64, 'layers_2_RPPA': 32, 'layers_1_RPPA_activfunc': 'relu',
            'layers_2_RPPA_activfunc': 'prelu', 'layers_1_RPPA_dropout': 'yes', 'layers_2_RPPA_dropout': 'yes',
            'layers_1_RPPA_batchnorm': 'no', 'layers_2_RPPA_batchnorm': 'yes', 'layers_1_RPPA_hierarichcal': 55,
            'layers_2_RPPA_hierarichcal': 20, 'layers_1_RPPA_activfunc_hierarichcal': 'relu',
            'layers_2_RPPA_activfunc_hierarichcal': 'relu', 'layers_1_RPPA_dropout_hierarichcal': 'yes',
            'layers_2_RPPA_dropout_hierarichcal': 'no', 'layers_1_RPPA_batchnorm_hierarichcal': 'no',
            'layers_2_RPPA_batchnorm_hierarichcal': 'no', 'cross_decoders_3_views': (2, 1, 0),
            'layers_1_FCNN': 37, 'layers_2_FCNN': 30, 'layers_1_FCNN_activfunc': 'prelu',
            'layers_2_FCNN_activfunc': 'prelu', 'FCNN_dropout_prob': 0.4, 'FCNN_dropout_bool': False,
            'FCNN_batchnorm_bool': True, 'layers_1_FCNN_dropout': 'no', 'layers_2_FCNN_dropout': 'yes',
            'layers_1_FCNN_batchnorm': 'no', 'layers_2_FCNN_batchnorm': 'no'}




    if selection_method_train.lower() == 'pca' or selection_method_train.lower() == 'variance' or\
            selection_method_train.lower() == 'eigengenes' or selection_method_train.lower() == 'ae':
        print("Feature selection....")
        train_data, val_data, test_data, \
        train_duration, train_event, \
        val_duration, val_event, \
        test_duration, test_event = multimodule.feature_selection(method=selection_method_train,
                                                                  components= components_train,
                                                                  thresholds= thresholds_train,
                                                                  feature_names= feature_names)
    else: # PPI feature selection
        edge_index, proteins_used, train_data, val_data, test_data, \
        train_duration, train_event, \
        val_duration, val_event, \
        test_duration, test_event = multimodule.feature_selection('ppi', feature_names)

    if method_train == 'FCNN':
        print("######################## RUNNING FULLY CONNECTED NEURAL NET ####################################")
        FCNN.train(train_data, val_data, test_data,
                   train_duration, val_duration, test_duration,
                   train_event, val_event, test_event,
                   n_epochs = 100,
                   batch_size= params['batch_size'],
                   l2_regularization=params['l2_regularization_bool'],
                   l2_regularization_rate=params['l2_regularization_rate'],
                   learning_rate= params['learning_rate'],
                   prelu_rate= params['prelu_rate'],
                   layers=[[params['layers_1_mRNA'], params['layers_2_mRNA']],
                           [params['layers_1_microRNA'], params['layers_2_microRNA']],
                           [params['layers_1_RPPA'],params['layers_2_RPPA']]],
                   activation_layers= [[params['layers_1_mRNA_activfunc'], params['layers_2_mRNA_activfunc']],
                                       [params['layers_1_microRNA_activfunc'], params['layers_2_microRNA_activfunc']],
                                       [params['layers_1_RPPA_activfunc'], params['layers_2_RPPA_activfunc']]],
                   dropout = params['dropout_bool'],
                   dropout_rate= params['dropout_prob'],
                   dropout_layers= [[params['layers_1_mRNA_dropout'],params['layers_2_mRNA_dropout']],
                                    [params['layers_1_microRNA_dropout'],params['layers_2_microRNA_dropout']],
                                    [params['layers_1_RPPA_dropout'],params['layers_2_RPPA_dropout']]],
                   batchnorm= params['batchnorm_bool'],
                   batchnorm_layers= [[params['layers_1_mRNA_batchnorm'], params['layers_2_mRNA_batchnorm']],
                                      [params['layers_1_microRNA_batchnorm'], params['layers_2_microRNA_batchnorm']],
                                      [params['layers_1_RPPA_batchnorm'], params['layers_2_RPPA_batchnorm']]],
                   view_names = view_names_fix)


        print("######################## FULLY CONNECTED NEURAL NET FINISHED ####################################")


    elif method_train == 'GCN':
        print("######################## RUNNING GCN ####################################")


        GCN.train(train_data, val_data, test_data,
                  train_duration, val_duration, test_duration,
                  train_event, val_event, test_event,
                  n_epochs = 100,
                  batch_size= params['batch_size'],
                  l2_regularization=params['l2_regularization_bool'],
                  l2_regularization_rate=params['l2_regularization_rate'],
                  learning_rate= params['learning_rate'],
                  prelu_rate= params['prelu_rate'],
                  layers=[[params['layers_1_mRNA'], params['layers_2_mRNA']],
                          [params['layers_1_microRNA'], params['layers_2_microRNA']],
                          [params['layers_1_RPPA'],params['layers_2_RPPA']]],
                  activation_layers= [[params['layers_1_mRNA_activfunc'], params['layers_2_mRNA_activfunc']],
                                      [params['layers_1_microRNA_activfunc'], params['layers_2_microRNA_activfunc']],
                                      [params['layers_1_RPPA_activfunc'], params['layers_2_RPPA_activfunc']]],
                  dropout = params['dropout_bool'],
                  dropout_rate= params['dropout_prob'],
                  dropout_layers= [[params['layers_1_mRNA_dropout'],params['layers_2_mRNA_dropout']],
                                   [params['layers_1_microRNA_dropout'],params['layers_2_microRNA_dropout']],
                                   [params['layers_1_RPPA_dropout'],params['layers_2_RPPA_dropout']]],
                  batchnorm= params['batchnorm_bool'],
                  batchnorm_layers= [[params['layers_1_mRNA_batchnorm'], params['layers_2_mRNA_batchnorm']],
                                     [params['layers_1_microRNA_batchnorm'], params['layers_2_microRNA_batchnorm']],
                                     [params['layers_1_RPPA_batchnorm'], params['layers_2_RPPA_batchnorm']]],
                  processing_type= 'none',
                  edge_index= edge_index,
                  proteins_used= proteins_used,
                  ratio=None,
                  activation_layers_graphconv= None,
                  layers_graphconv= None)


        print("######################## GCN FINISHED ####################################")

    elif method_train == 'AE':
        print("######################## RUNNING AUTOENCODER ####################################")

        AE.train(train_data, val_data, test_data,
                 train_duration, val_duration, test_duration,
                 train_event, val_event, test_event,
          n_epochs = 100,
          batch_size= params['batch_size'],
          l2_regularization=params['l2_regularization_bool'],
          l2_regularization_rate=params['l2_regularization_rate'],
          learning_rate= params['learning_rate'],
          prelu_rate= params['prelu_rate'],
          layers=[[params['layers_1_mRNA'], params['layers_2_mRNA']],
                  [params['layers_1_microRNA'], params['layers_2_microRNA']],
                  [params['layers_1_RPPA'],params['layers_2_RPPA']]],
          activation_layers= [[params['layers_1_mRNA_activfunc'], params['layers_2_mRNA_activfunc']],
                              [params['layers_1_microRNA_activfunc'], params['layers_2_microRNA_activfunc']],
                              [params['layers_1_RPPA_activfunc'], params['layers_2_RPPA_activfunc']]],
          dropout = params['dropout_bool'],
          dropout_rate= params['dropout_prob'],
          dropout_layers= [[params['layers_1_mRNA_dropout'],params['layers_2_mRNA_dropout']],
                           [params['layers_1_microRNA_dropout'],params['layers_2_microRNA_dropout']],
                           [params['layers_1_RPPA_dropout'],params['layers_2_RPPA_dropout']]],
          batchnorm= params['batchnorm_bool'],
          batchnorm_layers= [[params['layers_1_mRNA_batchnorm'], params['layers_2_mRNA_batchnorm']],
                             [params['layers_1_microRNA_batchnorm'], params['layers_2_microRNA_batchnorm']],
                             [params['layers_1_RPPA_batchnorm'], params['layers_2_RPPA_batchnorm']]],
          view_names = view_names_fix,
          cross_mutation= params['cross_decoders_3_views'],
          model_types =  ['none','cross_elementwiseavg'],
          dropout_second = params['dropout_bool_hierachical'],
          dropout_rate_second = params['dropout_prob_hierachical'],
          batchnorm_second = params['batchnorm_bool_hierachical'],
          layers_second = [[params['layers_1_mRNA_hierarichcal'], params['layers_2_mRNA_hierarichcal']],
                           [params['layers_1_microRNA_hierarichcal'], params['layers_2_microRNA_hierarichcal']],
                           [params['layers_1_RPPA_hierarichcal'],params['layers_2_RPPA_hierarichcal']]],
          activation_layers_second = [[params['layers_1_mRNA_activfunc_hierarichcal'], params['layers_2_mRNA_activfunc_hierarichcal']],
                                      [params['layers_1_microRNA_activfunc_hierarichcal'], params['layers_2_microRNA_activfunc_hierarichcal']],
                                      [params['layers_1_RPPA_activfunc_hierarichcal'], params['layers_2_RPPA_activfunc_hierarichcal']]],
          dropout_layers_second = [[params['layers_1_mRNA_dropout_hierarichcal'],params['layers_2_mRNA_dropout_hierarichcal']],
                                   [params['layers_1_microRNA_dropout_hierarichcal'],params['layers_2_microRNA_dropout_hierarichcal']],
                                   [params['layers_1_RPPA_dropout_hierarichcal'],params['layers_2_RPPA_dropout_hierarichcal']]],
          batchnorm_layers_second = [[params['layers_1_mRNA_batchnorm_hierarichcal'], params['layers_2_mRNA_batchnorm_hierarichcal']],
                                     [params['layers_1_microRNA_batchnorm_hierarichcal'], params['layers_2_microRNA_batchnorm_hierarichcal']],
                                     [params['layers_1_RPPA_batchnorm_hierarichcal'], params['layers_2_RPPA_batchnorm_hierarichcal']]],
          dropout_third = params['FCNN_dropout_bool'],
          dropout_rate_third = params['FCNN_dropout_prob'],
          batchnorm_third= params['FCNN_batchnorm_bool'],
          layers_third = [[params['layers_1_FCNN'], params['layers_2_FCNN']]],
          activation_layers_third = [[params['layers_1_FCNN_activfunc'], params['layers_2_FCNN_activfunc']],['none']],
          dropout_layers_third = [params['layers_1_FCNN_dropout'],params['layers_2_FCNN_dropout']],
          batchnorm_layers_third = [params['layers_1_FCNN_batchnorm'], params['layers_2_FCNN_batchnorm']],
          loss_rate = params['loss_surv'])

        print("######################## AUTOENCODER FINISHED ####################################")

    
