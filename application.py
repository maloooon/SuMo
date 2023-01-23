import NN
import AE
import GCN
import ReadInData
import DataInputNew
import torch



if __name__ == '__main__':
    cancer_data = ReadInData.readcancerdata('LUAD')
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
                                                        which_views = which_views)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup all the data
    n_train_samples, n_test_samples,n_val_samples, view_names = multimodule.setup()

    print("######################## RUNNING FULLY CONNECTED NEURAL NET ####################################")

 #   NN.train(module= multimodule,
 #         device=device,
 #         feature_select_method= 'pca',
 #         components = [100,100,20,20],
 #         thresholds= [0.9,0.9,0.9,0.9],
 #         feature_names= None,
 #         batch_size=128,
 #         n_epochs=100,
 #         l2_regularization=False,
 #         val_batch_size=32,
 #         number_folds= 3,
 #         dropout=False,
 #         dropout_rate=0.1,
 #         activation_functions_per_view = [['relu'],['none']], #doesnt work here, needs to be in net call itself to work ?
 #         dropout_per_view = [['yes','no']],
 #         n_train_samples = n_train_samples,
 #         n_test_samples = n_test_samples,
 #         n_val_samples = n_val_samples,
 #         view_names = view_names)


    print("######################## FULLY CONNECTED NEURAL NET FINISHED ####################################")

    print("######################## RUNNING GCN ####################################")


    GCN.train(module= multimodule,
              device=device,
              batch_size=256,
              n_epochs=100,
              l2_regularization=False,
              val_batch_size=64,
              number_folds=3,
              feature_names=feature_names,
              n_train_samples = n_train_samples,
              n_test_samples = n_test_samples,
              n_val_samples = n_val_samples,
              view_names = view_names)



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

    
