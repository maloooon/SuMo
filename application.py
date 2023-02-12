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
    n_folds = 1
    # needed for R, cant read in cancer name directly for some reason
    with open('/Users/marlon/Desktop/Project/TCGAData/currentcancer.txt', 'w') as f:
        f.write(cancer_name)

    multimodule = DataInputNew.SurvMultiOmicsDataModule(data,
                                                        feature_offsets,
                                                        view_names,
                                                        cancer_name= cancer_name,
                                                        which_views = which_views,
                                                        n_folds = n_folds,
                                                        type_preprocess= 'none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup all the data
    n_train_samples, n_test_samples,n_val_samples, view_names_fix = multimodule.setup()


    # Write used views for input into NN to .txt
    # RPPA data is often dismissed, because nearly all values are NaN
    with open('/Users/marlon/Desktop/Project/TCGAData/cancerviews.txt', 'w') as fp:
        for item in view_names_fix:
            # write each item on a new line
            fp.write("%s\n" % item)



    ############################################ HYPERPARAMETER OPTIMIZATION ##########################################

    # Hyperparameter Optimization NN method
    method_tune = 'none'

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


    elif method_tune == 'FCNN' or method_tune == 'AE':

        # Choose feature selection method (PCA,Variance,AE,Eigengenes)
        feature_select_method = 'pca'
        # Choose PCA components for each view (None : take all possible PC components for this view)
        components = [100,100,100,100]
        # Choose Variance thresholds for each view
        thresholds = [0.8,0.8,0.8,0.8]



        train_data, val_data, test_data, \
        train_duration, train_event, \
        val_duration, val_event, \
        test_duration, test_event = multimodule.feature_selection(method=feature_select_method,
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


        if method_tune == 'FCNN':
            FCNN.optuna_optimization()
        elif method_tune == 'AE':
            AE.optuna_optimization()



    method_train = 'GCN'

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
    layers_1_mRNA = 112
    layers_2_mRNA = 453
    layers_1_DNA = 336
    layers_2_DNA = 338
    layers_1_microRNA = 10
    layers_2_microRNA = 10
    layers_1_RPPA = 200
    layers_2_RPPA = 200
    # LAYER SIZES INTEGRATED
    layers_1 = 200
    layers_2 = 200
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
    # DROPOUT
    DROPOUT_BOOL = True
    DROPOUT_PROB = 0.2
    layers_1_mRNA_dropout = 'no'
    layers_2_mRNA_dropout = 'no'
    layers_1_DNA_dropout = 'no'
    layers_2_DNA_dropout = 'yes'
    layers_1_microRNA_dropout = 'yes'
    layers_2_microRNA_dropout = 'no'
    # DROPOUT FUNCTIONS INTEGRATED
    layers_1_integrated_dropout = 'yes'
    layers_2_integrated_dropout = 'yes'
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
    layers_1_integrated_batchnorm = 'yes'
    layers_2_integrated_batchnorm = 'yes'
    # FINAL LAYER
    layer_final_activfunc = 'none'
    layer_final_dropout = 'yes'
    layer_final_batchnorm = 'yes'
    # GRAPHCONV LAYERS
    layer_1_graphconv = 2 # best to take the same amount as the views we are looking at (normally DNA & mRNA) because of float errors otherwise
    layer_2_graphconv = 5
    # GRAPHCONV ACTIVATION LAYERS
    layer_1_graphconv_activfunc = 'relu'
    layer_2_graphconv_activfunc = 'relu'


    LAYERS = [[layers_1_mRNA, layers_2_mRNA],[layers_1_DNA,layers_2_DNA],[layers_1_microRNA, layers_2_microRNA]]

    ACTIV_FUNCS = [[layers_1_mRNA_activfunc, layers_2_mRNA_activfunc], [layers_1_DNA_activfunc, layers_2_DNA_activfunc],
                   [layers_1_microRNA_activfunc, layers_2_microRNA_activfunc], [layer_final_activfunc]]
    DROPOUT_LAYERS = [[layers_1_mRNA_dropout, layers_2_mRNA_dropout], [layers_1_DNA_dropout, layers_2_DNA_dropout],
                      [layers_1_microRNA_dropout, layers_2_microRNA_dropout], [layer_final_dropout]]
    BATCHNORM_LAYERS = [[layers_1_mRNA_batchnorm, layers_2_mRNA_batchnorm],[layers_1_DNA_batchnorm, layers_2_DNA_batchnorm],
                        [layers_1_microRNA_batchnorm,layers_2_microRNA_batchnorm], [layer_final_batchnorm]]

    INTEGRATED_LAYERS = [layers_1,layers_2]
    INTEGRATED_ACTIV_FUNCS = [[layers_1_integrated_activfunc, layers_2_integrated_activfunc], [layer_final_activfunc]]
    INTEGRATED_DROPOUT_LAYERS = [[layers_1_integrated_dropout, layers_2_integrated_dropout], [layer_final_dropout]]
    INTEGRATED_BATCHNORM_LAYERS = [[layers_1_integrated_batchnorm, layers_2_integrated_batchnorm], [layer_final_batchnorm]]

    GRAPHCONV_LAYERS = [layer_1_graphconv]
    GRAPHCONV_ACTIV_FUNCS = [layer_1_graphconv_activfunc]



    # Choose feature selection method (PCA,Variance,AE,Eigengenes)
    feature_select_method = 'ppi'
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
                  layers=INTEGRATED_LAYERS,
                  activation_layers= INTEGRATED_ACTIV_FUNCS,
                  dropout = DROPOUT_BOOL,
                  dropout_rate= DROPOUT_PROB,
                  dropout_layers= INTEGRATED_DROPOUT_LAYERS,
                  batchnorm= BATCHNORM_BOOL,
                  batchnorm_layers= INTEGRATED_BATCHNORM_LAYERS,
                  view_names = view_names_fix,
                  feature_names=feature_names,
                  processing_type= 'none',
                  edge_index= edge_index,
                  proteins_used= proteins_used,
                  activation_layers_graphconv= GRAPHCONV_ACTIV_FUNCS,
                  layers_graphconv= GRAPHCONV_LAYERS)


        print("######################## GCN FINISHED ####################################")

    elif method_train == 'AE':
        print("######################## RUNNING AUTOENCODER ####################################")




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

    
