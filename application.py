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
import time



def get_free_gpu_idx():
    """Get the index of the GPU with current lowest memory usage."""
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return np.argmin(memory_available)


def FeatureSelectFoldsAndSave(direc_set = 'SUMO', name_cancer ='KIRC', which_views = [], n_folds = 5,folds_folder_name='KIRP4VIEWS',feature_selection_type = 'PCA',
                              components_PCA = [100,100,100,100],preprocess_type = 'MaxAbs'):

    """
    Select Features and save data.
    Load prepared (preprocessed&feature selected) cancer.
    :param direc_set: Decide if on Desktop or GPU environment ; dtype : String
    :param name_cancer_folder: Folder to load cancer from ; type : String
    :param which_views: Name of the views to be analyzed ; dtype : List of strings
    :param n_folds: Number of folds ; dtype : Int
    :param preprocess_type: Type of preprocessing (MinMax/Robust/Standardize/MaxAbs) ; dtype : String
    :param feature_selection_type: Type of feature selection (PCA/Variance/PPI) ; dtype : String
    :param components_PCA: PCA components to choose ; dtype : List of int (one for each view)
    :return:
    """

    cancer_data = ReadInData.readcancerdata(name_cancer)
    data = cancer_data[0][0]
    feature_offsets = cancer_data[0][1]
    view_names = cancer_data[0][2]
    feature_names = cancer_data[0][3]
    cancer_name = cancer_data[0][4][0]




    # Call the module
    multimodule = DataProcessing.SurvMultiOmicsDataModule(data,
                                                          feature_offsets,
                                                          view_names,
                                                          cancer_name,
                                                          which_views =which_views,
                                                          n_folds = n_folds,
                                                          type_preprocess = None,
                                                          save_folds=False,
                                                          saved_folds_processing=True,
                                                          folds_folder_name = folds_folder_name,
                                                          direc_set= 'SUMO')

    dir = os.path.expanduser("~/{}/Project/ProcessedNotFeatSelectedData/{}/{}/ViewNames.txt".format(direc_set,preprocess_type,folds_folder_name))
    temp = open(dir,"r")
    view_names_fix = temp.read().split("\n")
    del view_names_fix[-1]
    # Load in remove columns from preprocessing so we can delete those from list of all given feature names for our cancers (needed for GCN)
    dir = os.path.expanduser("~/{}/Project/ProcessedNotFeatSelectedData/{}/{}/cols_remove.txt".format(direc_set,preprocess_type,folds_folder_name))
    temp = open(dir,"r")
    cols_remove = temp.read().split("\n")
    del cols_remove[-1]
    main_dir = '~/{}/Project/PreparedData/{}/{}/{}'.format(direc_set,folds_folder_name,feature_selection_type,preprocess_type)

    if feature_selection_type.lower() == 'ppi':
        edge_index, proteins_used, train_data, val_data, test_data, \
        train_duration, train_event, \
        val_duration, val_event, \
        test_duration, test_event = multimodule.feature_selection(feature_selection_type.lower(), feature_names,saved_data_loading = True,
                                                                  saved_data_preprocessing=preprocess_type,saved_data_folder_name=folds_folder_name,
                                                                  columns_removed = cols_remove)


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
            dir = os.path.expanduser('{}/TrainData_{}.csv'.format(main_dir,c_fold))
            train_data_df.to_csv(dir)
            dir = os.path.expanduser('{}/TrainDataFeatOffs_{}.csv'.format(main_dir,c_fold))
            feat_offs_train_df.to_csv(dir)

            all_val_data = []
            all_val_data.append(val_data[c_fold])
            all_val_data.append(val_duration[c_fold].unsqueeze(1))
            all_val_data.append(val_event[c_fold].unsqueeze(1))
            for idx, _ in enumerate(all_val_data):
                feat_offs_val.append(all_val_data[idx].size(1))
            val_data_c = torch.cat(tuple(all_val_data), dim=1)
            val_data_df = pd.DataFrame(val_data_c)
            dir = os.path.expanduser('{}/ValData_{}.csv'.format(main_dir,c_fold))
            val_data_df.to_csv(dir)



            all_test_data = []
            all_test_data.append(test_data[c_fold])
            all_test_data.append(test_duration.unsqueeze(1))
            all_test_data.append(test_event.unsqueeze(1))
            for idx, _ in enumerate(all_test_data):
                feat_offs_test.append(all_test_data[idx].size(1))
            test_data_c = torch.cat(tuple(all_test_data), dim=1)
            test_data_df = pd.DataFrame(test_data_c)
            dir = os.path.expanduser('{}/TestData_{}.csv'.format(main_dir,c_fold))
            test_data_df.to_csv(dir)



            dir = os.path.expanduser('{}/num_features.txt'.format(main_dir))
            with open(dir, 'w') as f:
                f.write(str(num_features))
            dir = os.path.expanduser('{}/num_nodes.txt'.format(main_dir))
            with open(dir, 'w') as f:
                f.write(str(num_nodes))
            dir = os.path.expanduser('{}/edge_index_1.txt'.format(main_dir))
            with open(dir, 'w') as f:
                f.write(','.join(str(i) for i in edge_index[0]))
            dir = os.path.expanduser('{}/edge_index_2.txt'.format(main_dir))
            with open(dir, 'w') as f:
                f.write(','.join(str(i) for i in edge_index[1]))



    else:
        train_data, val_data, test_data, \
        train_duration, train_event, \
        val_duration, val_event, \
        test_duration, test_event = multimodule.feature_selection(method=feature_selection_type.lower(),
                                                                  components= components_PCA,
                                                                  feature_names= feature_names,
                                                                  saved_data_loading = True,
                                                                  saved_data_preprocessing= preprocess_type,
                                                                  saved_data_folder_name=folds_folder_name)



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

            dir = os.path.expanduser('{}/TrainData_{}.csv'.format(main_dir,c_fold))
            train_data_df.to_csv(dir)
            dir = os.path.expanduser('{}/TrainDataFeatOffs_{}.csv'.format(main_dir,c_fold))
            feat_offs_train_df.to_csv(dir)


            all_val_data = copy.deepcopy(val_data[c_fold])
            all_val_data.append(val_duration[c_fold].unsqueeze(1))
            all_val_data.append(val_event[c_fold].unsqueeze(1))
            for idx, _ in enumerate(all_val_data):
                feat_offs_val.append(all_val_data[idx].size(1))
            val_data_c = torch.cat(tuple(all_val_data), dim=1)
            val_data_df = pd.DataFrame(val_data_c)

            dir = os.path.expanduser('{}/ValData_{}.csv'.format(main_dir,c_fold))
            val_data_df.to_csv(dir)




            all_test_data = copy.deepcopy(test_data[c_fold])
            all_test_data.append(test_duration.unsqueeze(1))
            all_test_data.append(test_event.unsqueeze(1))
            for idx, _ in enumerate(all_test_data):
                feat_offs_test.append(all_test_data[idx].size(1))
            test_data_c = torch.cat(tuple(all_test_data), dim=1)
            test_data_df = pd.DataFrame(test_data_c)

            dir = os.path.expanduser('{}/TestData_{}.csv'.format(main_dir,c_fold))
            test_data_df.to_csv(dir)

        dir = os.path.expanduser(r'{}/ViewNames.txt'.format(main_dir))
        with open(dir, 'w') as fp:
            for item in view_names_fix:
                # write each item on a new line
                fp.write("%s\n" % item)

def PreprocessFoldsAndSave(direc_set = 'SUMO', name_cancer ='KIRC', which_views = [], n_folds = 5,folds_folder_name='KIRP4VIEWS', preprocess_type = 'MaxAbs'):
    """
    Preprocess data and save it.
    Preprocess the folds,and save data.
    :param direc_set: Decide if on Desktop or GPU environment ; dtype : String
    :param name_cancer : Name of the cancer ; dtype : String
    :param which_views: Name of the views to be analyzed ; dtype : List of strings
    :param n_folds: Number of folds ; dtype : Int
    :param folds_folder_name: Folder to save data in ; type : String
    :param preprocess_type: Type of preprocessing (MinMax/Robust/Standardize/MaxAbs) ; dtype : String
    """

    cancer_data = ReadInData.readcancerdata(name_cancer)
    data = cancer_data[0][0]
    feature_offsets = cancer_data[0][1]
    view_names = cancer_data[0][2]
    feature_names = cancer_data[0][3]
    cancer_name = cancer_data[0][4][0]




    # Call the module
    multimodule = DataProcessing.SurvMultiOmicsDataModule(data,
                                                          feature_offsets,
                                                          view_names,
                                                          cancer_name,
                                                          which_views =which_views,
                                                          n_folds = n_folds,
                                                          type_preprocess = preprocess_type,
                                                          save_folds=False,
                                                          saved_folds_processing=True,
                                                          folds_folder_name = folds_folder_name,
                                                          direc_set= 'SUMO')


    n_train_samples, n_test_samples,n_val_samples, view_names_fix = multimodule.setup()

    dir = os.path.expanduser('~/{}/Project/TCGAData/cancerviews.txt'.format(direc_set))
    with open(dir, 'w') as fp:
        for item in view_names_fix:
            # write each item on a new line
            fp.write("%s\n" % item)


def SaveFolds(direc_set = 'SUMO', name_cancer = 'KIRC', which_views=[],n_folds=5,folds_folder_name='KIRP4VIEWS'):
    """
    Save cross validated folds.
    :param direct_set: Decide if on Desktop or GPU environment ; dtype : String
    :param name_cancer: Name of the cancer which we are currently analyzing ; dtype : String
    :param which_views: Decide which views to look at (use [] if you want to use all possible ones) ; dtype : List of Strings
    :param n_folds: Number of folds ; dtype : Int
    :param folds_folder_name: Folder to save data in ; type : String
    """

    # 4 views STAD,LUAD,LAML,KIRC,KIRP,LIHC,LUSC,MESO,UVM
    # 2 views ACC, BRCA, CHOL,DLBC, ESCA, GBM, KICH, KIRC, PaAAD, PCPG, READ, TGCT, THYM, UCEC, UCS (miRNA,RPPA)
    # 1 view BLCA, CESC, COAD, HNSC, LGG, SARC, SKCM, THCA (RPPA)
    cancer_data = ReadInData.readcancerdata(name_cancer)
    data = cancer_data[0][0]
    feature_offsets = cancer_data[0][1]
    view_names = cancer_data[0][2]
    feature_names = cancer_data[0][3]
    cancer_name = cancer_data[0][4][0]


    # Call the module
    multimodule = DataProcessing.SurvMultiOmicsDataModule(data,
                                                          feature_offsets,
                                                          view_names,
                                                          cancer_name= cancer_name,
                                                          which_views = which_views,
                                                          n_folds = n_folds,
                                                          type_preprocess= None,
                                                          save_folds=True,
                                                          saved_folds_processing= False,
                                                          folds_folder_name = folds_folder_name,
                                                          direc_set= 'SUMO')


    n_train_samples, n_test_samples,n_val_samples, view_names_fix = multimodule.setup()


def LoadPreparedCancer(direc_set = 'SUMO', name_cancer_folder = 'KIRC2', n_folds=5,
                       preprocess_type='Standardize',feature_selection_type='PCA',nn_type='FCNN',nn_setting=['concat'],
                       mode='tune'):
    """
    Load prepared (preprocessed&feature selected) cancer.
    :param direc_set: Decide if on Desktop or GPU environment ; dtype : String
    :param name_cancer_folder: Folder to load cancer from ; type : String
    :param which_views: Name of the views to be analyzed ; dtype : List of strings
    :param n_folds: Number of folds ; dtype : Int
    :param preprocess_type: Type of preprocessing (MinMax/Robust/Standardize/MaxAbs) ; dtype : String
    :param feature_selection_type: Type of feature selection (PCA/Variance/PPI) ; dtype : String
    :param nn_type: Type of neural network (FCNN/AE/GCN) ; dtype : String
    :param nn_setting: Setting of neural network (AE & GCN have different settings) ; dtype : List of strings
    :param mode: Decide whether to train and optimize (Train/Tune) ; dtype : String
    :param components_PCA: PCA components to choose ; dtype : List of int (one for each view)
    :return:
    """

    if preprocess_type.lower() == 'all':
        preprocessing_types = ['Standardize','MaxAbs','Robust']
    else:
        preprocessing_types = [preprocess_type]





    if nn_type.lower() == 'fcnn':

        time_list = []
        # For benchmarking, we iterate through each preprocessing type
        for p_type in preprocessing_types:
            for c_fold in range(5):
                print("Tuning FCNN on cancer {}, feature selection {} and preprocessing {}".format(name_cancer_folder,
                                                                                                   feature_selection_type,
                                                                                                   p_type))
                start_time = time.time()
                FCNN.optuna_optimization(n_fold=c_fold,t_preprocess=p_type,
                                         feature_selection_type=feature_selection_type,cancer=name_cancer_folder, mode='prepared_data')
                #JUMPER1
                # Time in minutes
                time_variable = (time.time() - start_time) / 60
                time_list.append(time_variable)
            # Define where you want to save the time it took for each fold
            dir = os.path.expanduser(r'~/{}/Project/Trial/FCNN_{}_{}_{}_TIME_VAR2L.txt'.format(direc_set,
                                                                                               feature_selection_type,
                                                                                               name_cancer_folder,p_type))
            with open(dir, 'w') as fp:
                for item in time_list:
                    # write each item on a new line
                    fp.write("%s\n" % item)

    elif nn_type.lower() == 'ae':

        decoder_bool = False
        if decoder_bool == True:
            decoder = "decoder"
        else:
            decoder = ""

        for p_type in preprocessing_types:
            time_list = []
            for c_fold in range(5):
                start_time = time.time()
                AE.optuna_optimization(n_fold=c_fold,t_preprocess=p_type,feature_selection_type = feature_selection_type,
                                       model_type = nn_setting, decoder_bool=decoder_bool, cancer=name_cancer_folder, mode='prepared_data')
                #JUMPER1
                # Time in minutes
                time_variable = (time.time() - start_time) / 60
                time_list.append(time_variable)
            # Define where you want to save the time it took for each fold
            dir = os.path.expanduser(r'~/{}/Project/Trial/{}/{}/AE_{}_{}_{}_{}_{}_TIME.txt'.format(direc_set,
                                                                                                   nn_setting[0],

                                                                                                   p_type,
                                                                                                   feature_selection_type,
                                                                                                   name_cancer_folder,
                                                                                                   p_type,nn_setting[0],
                                                                                                   decoder))
            with open(dir, 'w') as fp:
                for item in time_list:
                    # write each item on a new line
                    fp.write("%s\n" % item)


    elif nn_type.lower() == 'gcn':
        layer_amount = '2layer'
        for p_type in preprocessing_types:
            time_list = []
            for c_fold in range(5):
                start_time = time.time()
                GCN.optuna_optimization(n_fold=c_fold,cancer=name_cancer_folder,t_preprocess=p_type,layer_amount=layer_amount)
                time_variable = (time.time() - start_time) / 60
                time_list.append(time_variable)
            # Define where you want to save the time it took for each fold
            dir = os.path.expanduser(r'~/{}/Project/Trial/GCN/{}/{}/GCN_TIME.txt'.format(direc_set,layer_amount,p_type))
            with open(dir, 'w') as fp:
                for item in time_list:
                    # write each item on a new line
                    fp.write("%s\n" % item)








#################### OLD FUNCTION, used to load in brand new cancer and directly preprocess, feature select etc. // use above functions and save preprocessed data, feature selected etc.
#################### as this is needed for benchmark purposes
"""
def LoadNewCancer(direc_set = 'SUMO',name_cancer='KIRC',which_views=[],n_folds=5,preprocess_type='Standardize',
                  feature_selection_type='PCA',nn_type='FCNN',nn_setting=['concat'],mode='train',components_PCA=[100,100,100,100],folds_folder_name = 'KIRC2'):

    
  #  Load in a new cancer and choose a preprocessing, feature selection and NN type. Decide whether to optimize with
  #  optuna or simply train the NN on own inputs.
  #  :param name_cancer: Name of the cancer ; dtype : String
  #  :param which_views: Name of the views to be analyzed ; dtype : List of strings
  #  :param n_folds: Number of folds ; dtype : Int
  #  :param preprocess_type: Type of preprocessing (MinMax/Robust/Standardize/MaxAbs) ; dtype : String
  #  :param feature_selection_type: Type of feature selection (PCA/Variance/PPI) ; dtype : String
  #  :param nn_type : Type of neural network (FCNN/AE/GCN) ; dtype : String
  #  :param nn_setting : Setting of neural network (AE & GCN have different settings) ; dtype : List of strings
  #  :param mode: Decide whether to train and optimize (Train/Tune) ; dtype : String
  #  :param components_PCA: PCA components to choose ; dtype : List of int (one for each view)
  #  :param folds_folder_name : Folder to save folds into ; dtype : String


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on device : ", device)
    # Define Cancer to load
    # Possible Cancers are :
    cancer_data = ReadInData.readcancerdata(name_cancer)
    data = cancer_data[0][0]
    feature_offsets = cancer_data[0][1]
    view_names = cancer_data[0][2]
    feature_names = cancer_data[0][3]
    cancer_name = cancer_data[0][4][0]

    print("Which views : ", which_views)
    print("Number of folds : ", n_folds)
    print("Preprocessing Type : ", preprocess_type)
    print("Neural Network : ", nn_type)
    print("Feature Selection Method : " ,feature_selection_type)
    if nn_type.lower() == 'GCN' or nn_type.lower() == 'AE':
        print("NN setting :", nn_setting)
    if feature_selection_type.lower() == 'pca':
        print("PCA components : ", components_PCA)

    print("Mode :", mode)

    dir = os.path.expanduser('~/{}/Project/TCGAData/currentcancer.txt'.format(direc_set))
    with open(dir, 'w') as f:
        f.write(cancer_name)

    # Call the module
    multimodule = DataProcessing.SurvMultiOmicsDataModule(data,
                                                          feature_offsets,
                                                          view_names,
                                                          cancer_name= cancer_name,
                                                          which_views = which_views,
                                                          n_folds = n_folds,
                                                          type_preprocess= preprocess_type,
                                                          save_folds= False,
                                                          saved_folds_processing=False,
                                                          folds_folder_name=folds_folder_name)


    n_train_samples, n_test_samples,n_val_samples, view_names_fix = multimodule.setup()

    # After preprocessing, views might have been deleted due to missing data 
    dir = os.path.expanduser('~/{}/Project/TCGAData/cancerviews.txt'.format(direc_set))
    with open(dir, 'w') as fp:
        for item in view_names_fix:
            # write each item on a new line
            fp.write("%s\n" % item)

    if mode.lower() == 'tune':
        if nn_type.lower() == 'gcn':
            assert(feature_selection_type.lower() == 'ppi'), "for GCN, the feature selection needs to be ppi."
            edge_index, proteins_used, train_data, val_data, test_data, \
            train_duration, train_event, \
            val_duration, val_event, \
            test_duration, test_event = multimodule.feature_selection(feature_selection_type.lower(), feature_names)

            # For each fold 
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
                train_data_df.to_csv(dir)
                dir = os.path.expanduser('~/{}/Project/PreparedData/TrainDataFeatOffs_{}.csv'.format(direc_set,c_fold))
                feat_offs_train_df.to_csv(dir)

                all_val_data = []
                all_val_data.append(val_data[c_fold])
                all_val_data.append(val_duration[c_fold].unsqueeze(1))
                all_val_data.append(val_event[c_fold].unsqueeze(1))
                for idx, _ in enumerate(all_val_data):
                    feat_offs_val.append(all_val_data[idx].size(1))
                val_data_c = torch.cat(tuple(all_val_data), dim=1)
                val_data_df = pd.DataFrame(val_data_c)
                dir = os.path.expanduser('~/{}/Project/PreparedData/ValData_{}.csv'.format(direc_set,c_fold))
                val_data_df.to_csv(dir)



                all_test_data = []
                all_test_data.append(test_data[c_fold])
                all_test_data.append(test_duration.unsqueeze(1))
                all_test_data.append(test_event.unsqueeze(1))
                for idx, _ in enumerate(all_test_data):
                    feat_offs_test.append(all_test_data[idx].size(1))
                test_data_c = torch.cat(tuple(all_test_data), dim=1)
                test_data_df = pd.DataFrame(test_data_c)
                dir = os.path.expanduser('~/{}/Project/PreparedData/TestData_{}.csv'.format(direc_set,c_fold))
                test_data_df.to_csv(dir)



                dir = os.path.expanduser('~/{}/Project/PreparedData/num_features.txt'.format(direc_set))
                with open(dir, 'w') as f:
                    f.write(str(num_features))
                dir = os.path.expanduser('~/{}/Project/PreparedData/num_nodes.txt'.format(direc_set))
                with open(dir, 'w') as f:
                    f.write(str(num_nodes))
                dir = os.path.expanduser('~/{}/Project/PreparedData/edge_index_1.txt'.format(direc_set))
                with open(dir, 'w') as f:
                    f.write(','.join(str(i) for i in edge_index[0]))
                dir = os.path.expanduser('~/{}/Project/PreparedData/edge_index_2.txt'.format(direc_set))
                with open(dir, 'w') as f:
                    f.write(','.join(str(i) for i in edge_index[1]))


            GCN.optuna_optimization()

        if nn_type.lower() == 'ae' or nn_type.lower() == 'fcnn':
            assert(feature_selection_type.lower() != 'ppi'), "For AE or FCNN, the feature selection can't be ppi."
            train_data, val_data, test_data, \
            train_duration, train_event, \
            val_duration, val_event, \
            test_duration, test_event = multimodule.feature_selection(method=feature_selection_type.lower(),
                                                                      components= components_PCA,
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
                train_data_df.to_csv(dir)
                dir = os.path.expanduser('~/{}/Project/PreparedData/TrainDataFeatOffs_{}.csv'.format(direc_set,c_fold))
                feat_offs_train_df.to_csv(dir)


                all_val_data = copy.deepcopy(val_data[c_fold])
                all_val_data.append(val_duration[c_fold].unsqueeze(1))
                all_val_data.append(val_event[c_fold].unsqueeze(1))
                for idx, _ in enumerate(all_val_data):
                    feat_offs_val.append(all_val_data[idx].size(1))
                val_data_c = torch.cat(tuple(all_val_data), dim=1)
                val_data_df = pd.DataFrame(val_data_c)

                dir = os.path.expanduser('~/{}/Project/PreparedData/ValData_{}.csv'.format(direc_set,c_fold))
                val_data_df.to_csv(dir)




                all_test_data = copy.deepcopy(test_data[c_fold])
                all_test_data.append(test_duration.unsqueeze(1))
                all_test_data.append(test_event.unsqueeze(1))
                for idx, _ in enumerate(all_test_data):
                    feat_offs_test.append(all_test_data[idx].size(1))
                test_data_c = torch.cat(tuple(all_test_data), dim=1)
                test_data_df = pd.DataFrame(test_data_c)

                dir = os.path.expanduser('~/{}/Project/PreparedData/TestData_{}.csv'.format(direc_set,c_fold))
                test_data_df.to_csv(dir)





            if nn_type.lower() == 'fcnn':
                FCNN.optuna_optimization()
            elif nn_type.lower() == 'ae':
                AE.optuna_optimization()

    if mode.lower() == 'train':
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
                'layers_2_RPPA_batchnorm_hierarichcal': 'no', 'cross_decoders_2_views': (1, 0),
                'layers_1_FCNN': 37, 'layers_2_FCNN': 30, 'layers_1_FCNN_activfunc': 'prelu',
                'layers_2_FCNN_activfunc': 'prelu', 'FCNN_dropout_prob': 0.4, 'FCNN_dropout_bool': False,
                'FCNN_batchnorm_bool': True, 'layers_1_FCNN_dropout': 'no', 'layers_2_FCNN_dropout': 'yes',
                'layers_1_FCNN_batchnorm': 'no', 'layers_2_FCNN_batchnorm': 'no'}
        if nn_type.lower() == 'fcnn':
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
        if nn_type.lower() == 'gcn':
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
        if nn_type.lower() == 'ae':
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



"""

if __name__ == '__main__':
    try:
        gpu_idx = get_free_gpu_idx()
        print("Using GPU #%s" % gpu_idx)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    except Exception as e:
        print(e)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on device : ", device)
    # Folds saved in folder FoldsNew 
  #  SaveFolds(name_cancer='KIRC',n_folds=5,folds_folder_name='KIRC4VIEWS',which_views=[])
    # Saved in folder : ProcessedNotFeatSelectedData
  
   # PreprocessFoldsAndSave(name_cancer='KIRC',n_folds=5,folds_folder_name='KIRC4VIEWS',preprocess_type='Robust')
    # Saved in folder : Prepared Data
   # FeatureSelectFoldsAndSave(name_cancer='LIHC',n_folds=5,folds_folder_name='LIHC4VIEWS',preprocess_type='MinMax', feature_selection_type = 'PCA')
   # FeatureSelectFoldsAndSave(name_cancer='STAD',n_folds=5,folds_folder_name='STAD4VIEWS',preprocess_type='MinMax', feature_selection_type = 'Variance_2')
  #  FeatureSelectFoldsAndSave(name_cancer='STAD',n_folds=5,folds_folder_name='STAD4VIEWS',preprocess_type='Robust', feature_selection_type = 'PCA')
   # FeatureSelectFoldsAndSave(name_cancer='KIRC',n_folds=5,folds_folder_name='KIRC4VIEWS',preprocess_type='Standardize', feature_selection_type = 'PCA')
  #

    # NOTE : For AE setting, everything can be set here, but if you wish to only use 1 Layer in FCNN when the last AE has overall as setting
    # NOTE : this needs to be set in AE code itself.
  #  LoadPreparedCancer(name_cancer_folder = 'LUSC4VIEWS',feature_selection_type='PCA',nn_type='FCNN',nn_setting= ['cross_elementwisemax','overallavg'],preprocess_type='MaxAbs')
   # LoadPreparedCancer(name_cancer_folder = 'LUAD4VIEWS',feature_selection_type='PCA',nn_type='AE',nn_setting= ['concat','overallmax'],preprocess_type='MinMax')
  #  LoadPreparedCancer(name_cancer_folder = 'KIRC4VIEWS',feature_selection_type='PCA',nn_type='AE',nn_setting= ['overallavg'],preprocess_type='MinMax')

    #   FCNN.test_model(n_fold=0,t_preprocess='MinMax',feature_selection_type='PCA',cancer='KIRC4VIEWS')
    #   AE.test_model(n_fold=3,t_preprocess='MinMax',feature_selection_type='PCA',model_types=['elementwisemax'],cancer='KIRC4VIEWS',decoder_bool=False)
 #   LoadNewCancer(name_cancer='KIRC',feature_selection_type='PPI', mode='tune', nn_type='GCN')







