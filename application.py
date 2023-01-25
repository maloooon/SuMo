import pandas as pd
import NN
import AE
import GCN
import ReadInData
import DataInputNew
import torch



if __name__ == '__main__':
    cancer_data = ReadInData.readcancerdata('LAML')
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
                                                        onezeronorm_bool=False,
                                                        cancer_name= cancer_name,
                                                        which_views = which_views,
                                                        n_folds = n_folds,
                                                        preprocess_bool = True)



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
        "Layers_mRNA" : [[1024,512,256], [256, 128, 64], [64,32,16]],
        "Layers_DNA" : [[1024,512,256], [256, 128, 64], [64,32,16]],
        "Layers_microRNA" : [[1024,512,256], [256, 128, 64], [64,32,16]],
        "Layers_RPPA" : [[1024,512,256], [256, 128, 64], [64,32,16]],
        "BatchSize" : [200,150,128,100,75,64,50,32,16],
        "BatchSizeVal" : [100,75,64,50,40,32,20,16,8,4],
        "LearningRate" : [0.01,0.001,0.0008,0.0006,0.0004,0.0001,0.00005,0.00001],
        "DropoutBool" : ['yes','no'],
        "BatchNormBool" : ['yes','no'],
        "NNLayersConcat" : [[256,128,64], [64,32,16], [16,8,4,2]]

    }


    print("######################## RUNNING FULLY CONNECTED NEURAL NET ####################################")
    # FEATURE SELECTION SETTINGS
    selection_method_NN = 'pca'
    components_PCA_NN = [74,74,74,74]
    thresholds_VARIANCE_NN = [0.8,0.8,0.8,0.8]


    """
    #Select method for feature selection

    train_data, val_data, test_data, \
    train_duration, train_event, \
    val_duration, val_event, \
    test_duration, test_event = multimodule.feature_selection(method=selection_method_NN,
                                                              components= components_PCA_NN,
                                                              thresholds= thresholds_VARIANCE_NN,
                                                              feature_names= None)

    # for RayTune we store prepared data and will load data with a function later on for each fold
    # Feature offsets : Return them with feature selection methods
    
    for c_fold in range(n_folds):
        all_train_data = train_data[c_fold]
        all_train_data.append(train_duration[c_fold].unsqueeze(1))
        all_train_data.append(train_event[c_fold].unsqueeze(1))
        train_data = torch.cat(tuple(all_train_data), dim=1)
        train_data_df = pd.DataFrame(train_data)
        train_data_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/Fold" + str(c_fold + 1) +"_TrainData.csv")
        all_val_data = val_data[c_fold]
        all_val_data.append(val_duration[c_fold].unsqueeze(1))
        all_val_data.append(val_event[c_fold].unsqueeze(1))
        val_data = torch.cat(tuple(all_val_data), dim=1)
        val_data_df = pd.DataFrame(val_data)
        val_data_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/Fold" + str(c_fold + 1) +"_ValData.csv")
        all_test_data = test_data[c_fold]
        all_test_data.append(test_duration.unsqueeze(1))
        all_test_data.append(test_event.unsqueeze(1))
        test_data = torch.cat(tuple(all_test_data), dim=1)
        test_data_df = pd.DataFrame(test_data)
        test_data_df.to_csv("/Users/marlon/Desktop/Project/PreparedData/Fold" + str(c_fold + 1) +"_TestData.csv")
        
    """




 #   train,val,test = NN.load_data(n_fold=1)




    # FULLY CONNECTED NEURAL NET SETTINGS
    # LAYER SETTINGS
    layers_NN = [[8] for i in range(len(view_names_fix))]
    # ACTIVATION FUNCTIONS SETTINGS
    activations_layers_NN = [['relu'] for i in range(len(view_names_fix))]
    activation_last_layer_NN = ['none']
    activations_layers_NN.append(activation_last_layer_NN)
    activations_NN = activations_layers_NN
    # DROPOUT SETTINGS
    dropout_bool_NN = False
    dropout_rate_NN = 0.1
    dropout_layers_NN = [['yes' for _ in range(len(layers_NN[0]))] for i in range(len(view_names_fix))]
    # BATCH NORMALIZATION SETTINGS
    batchnorm_bool_NN = False
    batchnorm_layers_NN = [['yes' for _ in range(len(layers_NN[0]))] for i in range(len(view_names_fix))]
    # L2 REGULARIZATION SETTINGS
    l2_regularization_bool_NN = False
    l2_regularization_rate_NN = 0.000001
    # BATCH SIZE SETTINGS
    batch_size_NN = 64
    val_batch_size_NN = 16
    # EPOCH SETTINGS
    n_epochs_NN = 100
    # LEARNING RATE OPTIMIZER (ADAM)
    learning_rate = 0.005



 #   NN.train(module= multimodule,
 #         feature_select_method= selection_method_NN,
 #         components = components_PCA_NN,
 #         thresholds= thresholds_VARIANCE_NN,
 #         feature_names= None,
 #         batch_size=batch_size_NN,
 #         n_epochs=n_epochs_NN,
 #         learning_rate= learning_rate,
 #         l2_regularization=l2_regularization_bool_NN,
 #         l2_regularization_rate=l2_regularization_rate_NN,
 #         val_batch_size=val_batch_size_NN,
 #         dropout=dropout_bool_NN,
 #         dropout_rate=dropout_rate_NN,
 #         batchnorm = batchnorm_bool_NN,
 #         layers = layers_NN,
 #         activation_layers = activations_NN,
 #         batchnorm_layers = batchnorm_layers_NN,
 #         dropout_layers = dropout_layers_NN,
 #         view_names = view_names_fix,
 #         config=config,
 #         n_grid_search_iterations= 100)


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

    AE.train(module= multimodule,
          feature_select_method= selection_method_NN,
          components = components_PCA_NN,
          thresholds= thresholds_VARIANCE_NN,
          feature_names= None,
          batch_size=batch_size_NN,
          n_epochs=n_epochs_NN,
          learning_rate= learning_rate,
          l2_regularization=l2_regularization_bool_NN,
          l2_regularization_rate=l2_regularization_rate_NN,
          val_batch_size=val_batch_size_NN,
          dropout=dropout_bool_NN,
          dropout_rate=dropout_rate_NN,
          batchnorm = batchnorm_bool_NN,
          layers = layers_NN,
          activation_layers = activations_AE,
          batchnorm_layers = batchnorm_layers_NN,
          dropout_layers = dropout_layers_NN,
          view_names = view_names_fix,
          config=config,
          n_grid_search_iterations= 100)






    print("######################## AUTOENCODER FINISHED ####################################")

    
