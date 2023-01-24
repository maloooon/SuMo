import NN
import AE
import GCN
import ReadInData
import DataInputNew
import torch



if __name__ == '__main__':
    cancer_data = ReadInData.readcancerdata('PAAD')
    data = cancer_data[0][0]
    feature_offsets = cancer_data[0][1]
    view_names = cancer_data[0][2]
    feature_names = cancer_data[0][3]
    cancer_name = cancer_data[0][4][0]
    which_views = [] # no input to use all the given views

    # needed for R, cant read in cancer name directly for some weird reason...
    with open('/Users/marlon/Desktop/Project/TCGAData/currentcancer.txt', 'w') as f:
        f.write(cancer_name)

    if len(which_views) != 0:
        with open('/Users/marlon/Desktop/Project/TCGAData/cancerviews.txt', 'w') as fp:
            for item in which_views:
                # write each item on a new line
                fp.write("%s\n" % item)
    else:
        which_views = view_names
        with open('/Users/marlon/Desktop/Project/TCGAData/cancerviews.txt', 'w') as fp:
            for item in which_views:
                # write each item on a new line
                fp.write("%s\n" % item)





    # Possible Cancers : 'DNA', 'microRNA' , 'mRNA', 'RPPA'
    # Leaving which_views empty will take all views into consideration
    multimodule = DataInputNew.SurvMultiOmicsDataModule(data,
                                                        feature_offsets,
                                                        view_names,
                                                        onezeronorm_bool=False,
                                                        cancer_name= cancer_name,
                                                        which_views = which_views,
                                                        n_folds = 1,
                                                        preprocess_bool = True)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup all the data
    # view_names_fix : RPPA data often is all 0s, so we don't read in that data
    n_train_samples, n_test_samples,n_val_samples, view_names_fix = multimodule.setup()



    print("######################## RUNNING FULLY CONNECTED NEURAL NET ####################################")
    # FEATURE SELECTION SETTINGS
    selection_method_NN = 'eigengenes'
    components_PCA_NN = [100,100,20,20]
    thresholds_VARIANCE_NN = [0.8,0.8,0.8,0.8]


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



    NN.train(module= multimodule,
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
          activation_layers = activations_NN,
          batchnorm_layers = batchnorm_layers_NN,
          dropout_layers = dropout_layers_NN,
          view_names = view_names_fix)


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


 #   AE.train(module=multimodule,
 #            device=device,
 #            feature_select_method= 'pca',
 #            components = [100,100,20,20],
 #            thresholds= [0.8,0.8,0.8,0.8],
 #            feature_names= None,
 #            batch_size=128,
 #            n_epochs=100,
 #            l2_regularization=False,
  #           val_batch_size=32,
 #            number_folds=3,
 #            n_train_samples = n_train_samples,
 #            n_test_samples = n_test_samples,
  #           n_val_samples = n_val_samples,
  #           view_names = view_names)



    print("######################## AUTOENCODER FINISHED ####################################")

    
