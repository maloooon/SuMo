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
    print(device)

    # Define Cancer to load
    # Possible Cancers are :
    #PRAD, ACC, BLCA, BRCA,CESC,CHOL,COAD,DLBC,ESCA,GBM, HNSC,KICH,KIRC,KIRP,LAML,LGG,
    #LIHC,LUAD,LUSC,MESO,PAAD,PCPG,READ,SARC,SKCM,STAD,TGCT,THCA,THYM,UCEC,UCS,UVM
    cancer_data = ReadInData.readcancerdata('LUAD')
    data = cancer_data[0][0]
    feature_offsets = cancer_data[0][1]
    view_names = cancer_data[0][2]
    feature_names = cancer_data[0][3]
    cancer_name = cancer_data[0][4][0]



    # Decide which views to use
    # Cancers can have DNA, mRNA, microRNA, RPPA data (not all have all of them though)
    # Leaving which_views empty will take all views into consideration will take all possible views into consideration
    which_views = []
    # Decide number of folds for Cross-Validation. For Optuna Optimization, use just one fold.
    n_folds = 1 # Use 1 for Optuna Hyperparameter Optimization
    # needed for R, cant read in cancer name directly for some reason
    type_of_preprocessing = 'normalize'
    # Hyperparameter Optimization NN method
    method_tune = 'FCNN'
    # Feature selection method Hyperparamter Tuning FCNN/AE (For GCN its always PPI)
    selection_method_tuning = 'variance'

    print("Preprocessing Type : ", type_of_preprocessing)
    print("Tuning Neural Network : ", method_tune)
    print("Feature Selection Method Tuning: " ,selection_method_tuning)


   # dir = os.path.expanduser('~/SUMO/Project/TCGAData/currentcancer.txt') # for mlo gpu
    dir = os.path.expanduser('/Users/marlon/Desktop/Project/TCGAData/currentcancer.txt')
    with open(dir, 'w') as f:
        f.write(cancer_name)



    multimodule = DataProcessing.SurvMultiOmicsDataModule(data,
                                                        feature_offsets,
                                                        view_names,
                                                        cancer_name= cancer_name,
                                                        which_views = which_views,
                                                        n_folds = n_folds,
                                                        type_preprocess= type_of_preprocessing)





    # Setup all the data
    n_train_samples, n_test_samples,n_val_samples, view_names_fix = multimodule.setup()


    # Write used views for input into NN to .txt
    # RPPA data is often dismissed, because nearly all values are NaN
    #dir = os.path.expanduser('~/SUMO/Project/TCGAData/cancerviews.txt')
    dir = os.path.expanduser('/Users/marlon/Desktop/Project/TCGAData/cancerviews.txt')
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

        # We go through each fold (but in Optuna Optimization, we only produce one fold, so this loop is not necessary
        # per se (just easier to read and if Hyperparameter Optimization was to be implemented with multiple folds,
        # one can build code based on this
        for c_fold in range(n_folds):

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
           # dir = os.path.expanduser('~/SUMO/Project/PreparedData/TrainData.csv')
            dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/TrainData.csv")
            train_data_df.to_csv(dir)
           # dir = os.path.expanduser('~/SUMO/Project/PreparedData/TrainDataFeatOffs.csv')
            dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/TrainDataFeatOffs.csv")
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
          #  dir = os.path.expanduser('~/SUMO/Project/PreparedData/ValData.csv')
            dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/ValData.csv")
            val_data_df.to_csv(dir)
         #   dir = os.path.expanduser('~/SUMO/Project/PreparedData/ValDataFeatOffs.csv')
            dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/ValDataFeatOffs.csv")
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
           # dir = os.path.expanduser('~/SUMO/Project/PreparedData/TestData.csv')
            dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/TestData.csv")
            test_data_df.to_csv(dir)
        #    dir = os.path.expanduser('~/SUMO/Project/PreparedData/TestDataFeatOffs.csv')
            dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/TestDataFeatOffs.csv")
            feat_offs_test_df.to_csv(dir)



            # also load number of features per node & number of nodes used aswell as edge indices
      #      with open('/Users/marlon/Desktop/Project/PreparedData/num_features.txt', 'w') as f:
      #          f.write(str(num_features))
      #      with open('/Users/marlon/Desktop/Project/PreparedData/num_nodes.txt', 'w') as f:
      #          f.write(str(num_nodes))
      #      with open('/Users/marlon/Desktop/Project/PreparedData/edge_index_1.txt', 'w') as f:
      #          f.write(','.join(str(i) for i in edge_index[0]))
      #      with open('/Users/marlon/Desktop/Project/PreparedData/edge_index_2.txt', 'w') as f:
      #              f.write(','.join(str(i) for i in edge_index[1]))


          #  dir = os.path.expanduser('~/SUMO/Project/PreparedData/num_features.txt')
            dir = os.path.expanduser('/Users/marlon/Desktop/Project/PreparedData/num_features.txt')
            with open(dir, 'w') as f:
                f.write(str(num_features))
          #  dir = os.path.expanduser('~/SUMO/Project/PreparedData/num_nodes.txt')
            dir = os.path.expanduser('/Users/marlon/Desktop/Project/PreparedData/num_nodes.txt')
            with open(dir, 'w') as f:
                f.write(str(num_nodes))
          #  dir = os.path.expanduser('~/SUMO/Project/PreparedData/edge_index_1.txt')
            dir = os.path.expanduser('/Users/marlon/Desktop/Project/PreparedData/edge_index_1.txt')
            with open(dir, 'w') as f:
                f.write(','.join(str(i) for i in edge_index[0]))
          #  dir = os.path.expanduser('~/SUMO/Project/PreparedData/edge_index_2.txt')
            dir = os.path.expanduser('/Users/marlon/Desktop/Project/PreparedData/edge_index_2.txt')
            with open(dir, 'w') as f:
                f.write(','.join(str(i) for i in edge_index[1]))


        GCN.optuna_optimization()


    elif method_tune == 'FCNN' or method_tune == 'AE':

        # Choose feature selection method (PCA,Variance,AE,Eigengenes)
      #  feature_select_method = 'PCA'

        # Choose PCA components for each view (None : take all possible PC components for this view)
        components = [None,None,None,None]
        # Choose Variance thresholds for each view
        thresholds = [0.3,0.3,0.3,0.7]

   #     DataProcessing.optuna_optimization()   # AE FEATURE SELECTION OPTIMIZATION

        train_data, val_data, test_data, \
        train_duration, train_event, \
        val_duration, val_event, \
        test_duration, test_event = multimodule.feature_selection(method=selection_method_tuning,
                                                                  components= components,
                                                                  thresholds= thresholds,
                                                                  feature_names= feature_names)


        # We go through each fold (but in Optuna Optimization, we only produce one fold, so this loop is not necessary
        # per se (just easier to read and if Hyperparameter Optimization was to be implemented with multiple folds,
        # one can build code based on this
        for c_fold in range(n_folds):

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

        #    dir = os.path.expanduser('~/SUMO/Project/PreparedData/TrainData.csv')
            dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/TrainData.csv")
            train_data_df.to_csv(dir)
        #    dir = os.path.expanduser('~/SUMO/Project/PreparedData/TrainDataFeatOffs.csv')
            dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/TrainDataFeatOffs.csv")
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

        #    dir = os.path.expanduser('~/SUMO/Project/PreparedData/ValData.csv')
            dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/ValData.csv")
            val_data_df.to_csv(dir)
        #    dir = os.path.expanduser('~/SUMO/Project/PreparedData/ValDataFeatOffs.csv')
            dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/ValDataFeatOffs.csv")
            feat_offs_val_df.to_csv(dir)



            all_test_data = copy.deepcopy(test_data[c_fold])
            all_test_data.append(test_duration.unsqueeze(1))
            all_test_data.append(test_event.unsqueeze(1))
            for idx, _ in enumerate(all_test_data):
                feat_offs_test.append(all_test_data[idx].size(1))
            test_data_c = torch.cat(tuple(all_test_data), dim=1)
            test_data_df = pd.DataFrame(test_data_c)
            feat_offs_test = np.cumsum(feat_offs_test)
            feat_offs_test_df = pd.DataFrame(feat_offs_test)

     #       dir = os.path.expanduser('~/SUMO/Project/PreparedData/TestData.csv')
            dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/TestData.csv")
            test_data_df.to_csv(dir)
      #      dir = os.path.expanduser('~/SUMO/Project/PreparedData/TestDataFeatOffs.csv')
            dir = os.path.expanduser("/Users/marlon/Desktop/Project/PreparedData/TestDataFeatOffs.csv")
            feat_offs_test_df.to_csv(dir)




        if method_tune == 'FCNN':
            FCNN.optuna_optimization()
        elif method_tune == 'AE':
            AE.optuna_optimization()


    # Train NN method
    method_train = 'none'

    ######## SET OWN SETTINGS FOR NN CALL ############

    # FULLY CONNECTED NEURAL NET SETTINGS
    # EPOCHS
    N_EPOCHS = 30
    # BATCH SIZE
    BATCH_SIZE = 64
    # L2 REGULARIZATION
    L2_REGULARIZATION_BOOL = True
    L2_REGULARIZATION_RATE = 0.000833844235362246
    # LR
    LEARNING_RATE = 0.0017038276577365065
    # PRELU INIT VALUE
    PRELU_RATE = 0.1
    # LAYER SIZES DIFFERENT VIEWS
    layers_1_mRNA = 64
    layers_2_mRNA = 16
    layers_1_DNA = 64
    layers_2_DNA = 16
    layers_1_microRNA = 64
    layers_2_microRNA = 16
    layers_1_RPPA = 200
    layers_2_RPPA = 200
    # LAYER SIZES INTEGRATED
    layers_1 = 16
    layers_2 = 8
    # LAYER SIZES INTEGRATED 2 (for Hierachical AE)
    layers_c_1 = 16
    layers_c_2 = 8
    # LAYER SIZES INTEGRATED ALL VIEWS
    layers_1_mRNA_integrated = 100
    layers_2_mRNA_integrated = 100
    layers_1_DNA_integrated = 100
    layers_2_DNA_integrated = 100
    layers_1_microRNA_integrated = 100
    layers_2_microRNA_integrated = 100
    layers_1_RPPA_integrated = 100
    layers_2_RPPA_integrated = 100
    # ACTIVATION FUNCTIONS
    layers_1_mRNA_activfunc = 'sigmoid'
    layers_2_mRNA_activfunc = 'sigmoid'
    layers_1_DNA_activfunc = 'relu'
    layers_2_DNA_activfunc = 'sigmoid'
    layers_1_microRNA_activfunc = 'relu'
    layers_2_microRNA_activfunc = 'sigmoid'
    layers_1_RPPA_activfunc = 'relu'
    layers_2_RPPA_activfunc = 'relu'
    # ACTIVATION FUNCTIONS INTEGRATED
    layers_1_integrated_activfunc = 'relu'
    layers_2_integrated_activfunc = 'relu'
    # ACTIVATION FUNCTIONS INTEGRATED 2
    layers_c_1_integrated_activfunc = 'relu'
    layers_c_2_integrated_activfunc = 'relu'

    # ACTIVATION FUNCTIONS INTEGRATED ALL VIEWS
    layers_1_mRNA_activfunc_integrated = 'sigmoid'
    layers_2_mRNA_activfunc_integrated = 'sigmoid'
    layers_1_DNA_activfunc_integrated = 'relu'
    layers_2_DNA_activfunc_integrated = 'sigmoid'
    layers_1_microRNA_activfunc_integrated = 'relu'
    layers_2_microRNA_activfunc_integrated = 'sigmoid'
    layers_1_RPPA_activfunc_integrated = 'relu'
    layers_2_RPPA_activfunc_integrated = 'relu'
    # DROPOUT
    DROPOUT_BOOL = True
    DROPOUT_PROB = 0.2
    layers_1_mRNA_dropout = 'no'
    layers_2_mRNA_dropout = 'no'
    layers_1_DNA_dropout = 'no'
    layers_2_DNA_dropout = 'yes'
    layers_1_microRNA_dropout = 'yes'
    layers_2_microRNA_dropout = 'no'
    layers_1_RPPA_dropout = 'no'
    layers_2_RPPA_dropout = 'no'
    # DROPOUT FUNCTIONS INTEGRATED
    DROPOUT_BOOL_INTEGRATED = True
    DROPOUT_PROB_INTEGRATED = 0.2
    layers_1_integrated_dropout = 'yes'
    layers_2_integrated_dropout = 'yes'
    # DROPOUT FUNCTIONS INTEGRATED 2
    DROPOUT_C_BOOL_INTEGRATED = True
    DROPOUT_C_PROB_INTEGRATED = 0.2
    layers_c_1_integrated_dropout = 'yes'
    layers_c_2_integrated_dropout = 'yes'
    # DROPOUT FUNCTIONS INTEGRATED ALL VIEWS
    DROPOUT_BOOL_INTEGRATED_VIEWS = True
    DROPOUT_PROB_INTEGRATED_VIEWS = 0.2
    layers_1_mRNA_dropout_integrated = 'no'
    layers_2_mRNA_dropout_integrated = 'no'
    layers_1_DNA_dropout_integrated = 'no'
    layers_2_DNA_dropout_integrated = 'yes'
    layers_1_microRNA_dropout_integrated = 'yes'
    layers_2_microRNA_dropout_integrated = 'no'
    layers_1_RPPA_dropout_integrated = 'no'
    layers_2_RPPA_dropout_integrated = 'no'
    # BATCH NORMALIZATION
    BATCHNORM_BOOL = False
    layers_1_mRNA_batchnorm = 'no'
    layers_2_mRNA_batchnorm = 'no'
    layers_1_DNA_batchnorm = 'no'
    layers_2_DNA_batchnorm = 'no'
    layers_1_microRNA_batchnorm = 'yes'
    layers_2_microRNA_batchnorm = 'yes'
    layers_1_RPPA_batchnorm = 'no'
    layers_2_RPPA_batchnorm = 'no'
    # BATCH NORMALIZATION INTEGRATED
    BATCHNORM_BOOL_INTEGRATED = True
    layers_1_integrated_batchnorm = 'yes'
    layers_2_integrated_batchnorm = 'yes'
    # BATCH NORMALIZATION INTEGRATED 2
    BATCHNORM_C_BOOL_INTEGRATED = True
    layers_c_1_integrated_batchnorm = 'yes'
    layers_c_2_integrated_batchnorm = 'yes'
    # BATCH NORMALIZATION INTEGRATED ALL VIEWS
    BATCHNORM_BOOL_INTEGRATED_VIEWS = True
    layers_1_mRNA_batchnorm_integrated = 'no'
    layers_2_mRNA_batchnorm_integrated = 'no'
    layers_1_DNA_batchnorm_integrated = 'no'
    layers_2_DNA_batchnorm_integrated = 'no'
    layers_1_microRNA_batchnorm_integrated = 'yes'
    layers_2_microRNA_batchnorm_integrated = 'yes'
    layers_1_RPPA_batchnorm_integrated = 'no'
    layers_2_RPPA_batchnorm_integrated = 'no'
    # FINAL LAYER
    layer_final_activfunc = 'none'
    layer_final_dropout = 'yes'
    layer_final_batchnorm = 'yes'
    # CROSS MUTATION (AE)
    cross_mutation = [0,1,2]
    # MODEL TYPES (AE)
    model_types = ['concat']
    # GRAPHCONV LAYERS (GCN)
    layer_1_graphconv = 2 # best to take the same amount as the views we are looking at (normally DNA & mRNA) because of float errors otherwise
    layer_2_graphconv = 5
    # GRAPHCONV ACTIVATION LAYERS (GCN)
    layer_1_graphconv_activfunc = 'relu'
    layer_2_graphconv_activfunc = 'relu'



    LAYERS = [[layers_1_mRNA, layers_2_mRNA],[layers_1_DNA,layers_2_DNA],[layers_1_microRNA, layers_2_microRNA]]

    ACTIV_FUNCS = [[layers_1_mRNA_activfunc, layers_2_mRNA_activfunc], [layers_1_DNA_activfunc, layers_2_DNA_activfunc],
                   [layers_1_microRNA_activfunc, layers_2_microRNA_activfunc], [layer_final_activfunc]]
    DROPOUT_LAYERS = [[layers_1_mRNA_dropout, layers_2_mRNA_dropout], [layers_1_DNA_dropout, layers_2_DNA_dropout],
                      [layers_1_microRNA_dropout, layers_2_microRNA_dropout], [layer_final_dropout]]
    BATCHNORM_LAYERS = [[layers_1_mRNA_batchnorm, layers_2_mRNA_batchnorm],[layers_1_DNA_batchnorm, layers_2_DNA_batchnorm],
                        [layers_1_microRNA_batchnorm,layers_2_microRNA_batchnorm], [layer_final_batchnorm]]


    # Used for GCN input or final FCNN input in AE construction
    INTEGRATED_LAYERS = [[layers_1, layers_2]]
    INTEGRATED_ACTIV_FUNCS = [[layers_1_integrated_activfunc, layers_2_integrated_activfunc], [layer_final_activfunc]]
    INTEGRATED_DROPOUT_LAYERS = [[layers_1_integrated_dropout, layers_2_integrated_dropout], [layer_final_dropout]]
    INTEGRATED_BATCHNORM_LAYERS = [[layers_1_integrated_batchnorm, layers_2_integrated_batchnorm], [layer_final_batchnorm]]


    ACTIV_FUNCS_AE =[[layers_1_mRNA_activfunc, layers_2_mRNA_activfunc], [layers_1_DNA_activfunc, layers_2_DNA_activfunc],
                     [layers_1_microRNA_activfunc, layers_2_microRNA_activfunc]]
    DROPOUT_LAYERS_AE = [[layers_1_mRNA_dropout, layers_2_mRNA_dropout], [layers_1_DNA_dropout, layers_2_DNA_dropout],
                      [layers_1_microRNA_dropout, layers_2_microRNA_dropout]]
    BATCHNORM_LAYERS_AE = [[layers_1_mRNA_batchnorm, layers_2_mRNA_batchnorm],[layers_1_DNA_batchnorm, layers_2_DNA_batchnorm],
                    [layers_1_microRNA_batchnorm,layers_2_microRNA_batchnorm]]

    # Second Integrated (Hierachical AE, Input into second AE)
    INTEGRATED_LAYERS_C_AE = [[layers_c_1,layers_c_2]]
    INTEGRATED_ACTIV_C_FUNCS_AE = [[layers_c_1_integrated_activfunc, layers_c_2_integrated_activfunc]]
    INTEGRATED_DROPOUT_C_LAYERS_AE = [[layers_c_1_integrated_dropout, layers_c_2_integrated_dropout]]
    INTEGRATED_BATCHNORM_C_LAYERS_AE = [[layers_c_1_integrated_batchnorm, layers_c_2_integrated_batchnorm]]

    # AE Hierarichal with 'none' setting as first model
    INTEGRATED_LAYERS_VIEWS = [[layers_1_mRNA_integrated, layers_2_mRNA_integrated],[layers_1_DNA_integrated, layers_2_DNA_integrated],
                             [layers_1_microRNA_integrated, layers_2_microRNA_integrated]]
    INTEGRATED_ACTIV_FUNCS_VIEWS = [[layers_1_mRNA_activfunc_integrated, layers_2_mRNA_activfunc_integrated],
                                    [layers_1_DNA_activfunc_integrated, layers_2_DNA_activfunc_integrated],
                                    [layers_1_microRNA_activfunc_integrated, layers_2_microRNA_activfunc_integrated]]
    INTEGRATED_DROPOUT_LAYERS_VIEWS = [[layers_1_mRNA_dropout_integrated, layers_2_RPPA_dropout_integrated],
                                       [layers_1_DNA_dropout_integrated, layers_2_DNA_dropout_integrated],
                                       [layers_1_microRNA_dropout_integrated, layers_2_microRNA_dropout_integrated]]
    INTEGRATED_BATCHNORM_LAYERS_VIEWS = [[layers_1_mRNA_batchnorm_integrated, layers_2_mRNA_batchnorm_integrated],
                                         [layers_1_DNA_batchnorm_integrated, layers_2_DNA_batchnorm_integrated],
                                         [layers_1_microRNA_batchnorm_integrated, layers_2_microRNA_batchnorm_integrated]]

    # GCN
    INTEGRATED_LAYERS_GCN = [layers_1,layers_2]
    GRAPHCONV_LAYERS = [layer_1_graphconv]
    GRAPHCONV_ACTIV_FUNCS = [layer_1_graphconv_activfunc]
    ratio = 0.2



    # Choose feature selection method (PCA,Variance,AE,Eigengenes)
    feature_select_method = 'pca'
    # Choose PCA components for each view (None : take all possible PC components for this view)
    components = [None,None,None,None]
    # Choose Variance thresholds for each view
    thresholds = [0.8,0.8,0.8,0.8]


    if feature_select_method.lower() == 'pca' or feature_select_method.lower() == 'variance' or\
            feature_select_method.lower() == 'eigengenes' or feature_select_method.lower() == 'ae':
        train_data, val_data, test_data, \
        train_duration, train_event, \
        val_duration, val_event, \
        test_duration, test_event = multimodule.feature_selection(method=feature_select_method,
                                                                  components= components,
                                                                  thresholds= thresholds,
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
                 train_event,val_event,test_event,
                 n_epochs= N_EPOCHS,
                 batch_size= BATCH_SIZE,
                 l2_regularization= L2_REGULARIZATION_BOOL,
                 l2_regularization_rate= L2_REGULARIZATION_RATE,
                 learning_rate= LEARNING_RATE,
                 prelu_rate = PRELU_RATE,
                 layers=LAYERS,
                 activation_layers= ACTIV_FUNCS,
                 dropout = DROPOUT_BOOL,
                 dropout_rate= DROPOUT_PROB,
                 dropout_layers= DROPOUT_LAYERS,
                 batchnorm= BATCHNORM_BOOL,
                 batchnorm_layers= BATCHNORM_LAYERS,
                 view_names = view_names_fix)


        print("######################## FULLY CONNECTED NEURAL NET FINISHED ####################################")


    elif method_train == 'GCN':
        print("######################## RUNNING GCN ####################################")


        GCN.train(train_data, val_data, test_data,
                  train_duration, val_duration, test_duration,
                  train_event,val_event,test_event,
                  n_epochs = N_EPOCHS,
                  batch_size= BATCH_SIZE,
                  l2_regularization= L2_REGULARIZATION_BOOL,
                  l2_regularization_rate= L2_REGULARIZATION_RATE,
                  learning_rate= LEARNING_RATE,
                  prelu_rate = PRELU_RATE,
                  layers=INTEGRATED_LAYERS_GCN,
                  activation_layers= INTEGRATED_ACTIV_FUNCS,
                  dropout = DROPOUT_BOOL,
                  dropout_rate= DROPOUT_PROB,
                  dropout_layers= INTEGRATED_DROPOUT_LAYERS,
                  batchnorm= BATCHNORM_BOOL,
                  batchnorm_layers= INTEGRATED_BATCHNORM_LAYERS,
                  processing_type= 'none',
                  edge_index= edge_index,
                  proteins_used= proteins_used,
                  ratio=ratio,
                  activation_layers_graphconv= GRAPHCONV_ACTIV_FUNCS,
                  layers_graphconv= GRAPHCONV_LAYERS)


        print("######################## GCN FINISHED ####################################")

    elif method_train == 'AE':
        print("######################## RUNNING AUTOENCODER ####################################")




        AE.train(train_data, val_data, test_data,
                 train_duration, val_duration, test_duration,
                 train_event, val_event, test_event,
          n_epochs = N_EPOCHS,
          batch_size= BATCH_SIZE,
          l2_regularization=L2_REGULARIZATION_BOOL,
          l2_regularization_rate=L2_REGULARIZATION_RATE,
          learning_rate= LEARNING_RATE,
          prelu_rate= PRELU_RATE,
          layers=LAYERS,
          activation_layers= ACTIV_FUNCS_AE,
          dropout = DROPOUT_BOOL,
          dropout_rate= DROPOUT_PROB,
          dropout_layers= DROPOUT_LAYERS_AE,
          batchnorm= BATCHNORM_BOOL,
          batchnorm_layers= BATCHNORM_LAYERS_AE,
          view_names = view_names_fix,
          cross_mutation= cross_mutation,
          model_types =  model_types,
          dropout_second = DROPOUT_C_BOOL_INTEGRATED,
          dropout_rate_second = DROPOUT_C_PROB_INTEGRATED,
          batchnorm_second = BATCHNORM_C_BOOL_INTEGRATED,
          layers_second = INTEGRATED_LAYERS_C_AE,
          activation_layers_second = INTEGRATED_ACTIV_C_FUNCS_AE,
          dropout_layers_second = INTEGRATED_DROPOUT_C_LAYERS_AE,
          batchnorm_layers_second = INTEGRATED_BATCHNORM_C_LAYERS_AE,
          dropout_third = DROPOUT_BOOL_INTEGRATED,
          dropout_rate_third = DROPOUT_PROB_INTEGRATED,
          batchnorm_third= BATCHNORM_BOOL_INTEGRATED,
          layers_third = INTEGRATED_LAYERS,
          activation_layers_third = INTEGRATED_ACTIV_FUNCS,
          dropout_layers_third = INTEGRATED_DROPOUT_LAYERS,
          batchnorm_layers_third = INTEGRATED_BATCHNORM_LAYERS)






        print("######################## AUTOENCODER FINISHED ####################################")

    
