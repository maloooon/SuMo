from statistics import mean

def flatten(l):
    """
    Function to flatten a list (e.g. turn a list of lists into a list)
    :param l: List input ; dtype : List
    :return: Flattened List (removal of one inner list layer)
    """
    return [item for sublist in l for item in sublist]



def averagingdictionaries(dicts):
    """
    Function to average int/float occurences of a list of dictionaries
     (used for testing on averaged optimized hyperparameter settings)
    :param dicts: List of dictionaries ; dtype : List
    :return: averaged dictionary
    """
    # Every dict in dicts has same parameters
    final_dict = {}

    for key,value in dicts[0].items():
        final_dict[key] = []

    for dictionary in dicts:
        for key,value in dictionary.items():
            final_dict[key].append(value)

    # Now that dict is full
    for key,value in final_dict.items():
        if type(value[0]) is int or type(value[0]) is float:
            if 'batch' in key:
                print(key)
                my_dict = {i:value.count(i) for i in value}
                ele = max(my_dict, key=my_dict.get)
                final_dict[key] = ele
            elif 'layers' in key:
                final_dict[key] = round(mean(value))
            else:
                final_dict[key] = mean(value)
        # For str/booleans, we take what occurs the most throughout all folds
        if type(value[0]) is str or type(value[0]) is bool:
            my_dict = {i:value.count(i) for i in value}
            ele = max(my_dict, key=my_dict.get)
            final_dict[key] = ele


    return final_dict


if __name__ == '__main__':
    # Example of averagingdictionaries()
    list_of_params = []
    params={'l2_regularization_bool': True, 'learning_rate': 5.34911649101302e-05, 'l2_regularization_rate': 3.0432897911163468e-05, 'batch_size': 128, 'dropout_prob': 0.5, 'dropout_bool': True, 'batchnorm_bool': True, 'prelu_rate': 0.75, 'layers_1_mRNA': 75, 'layers_2_mRNA': 26, 'layers_1_mRNA_activfunc': 'prelu', 'layers_2_mRNA_activfunc': 'prelu', 'layers_1_mRNA_dropout': 'yes', 'layers_2_mRNA_dropout': 'yes', 'layers_1_mRNA_batchnorm': 'no', 'layers_2_mRNA_batchnorm': 'yes', 'layers_1_microRNA': 41, 'layers_2_microRNA': 29, 'layers_1_microRNA_activfunc': 'relu', 'layers_2_microRNA_activfunc': 'prelu', 'layers_1_microRNA_dropout': 'yes', 'layers_2_microRNA_dropout': 'yes', 'layers_1_microRNA_batchnorm': 'no', 'layers_2_microRNA_batchnorm': 'no', 'layers_1_RPPA': 96, 'layers_2_RPPA': 18, 'layers_1_RPPA_activfunc': 'prelu', 'layers_2_RPPA_activfunc': 'relu', 'layers_1_RPPA_dropout': 'yes', 'layers_2_RPPA_dropout': 'yes', 'layers_1_RPPA_batchnorm': 'yes', 'layers_2_RPPA_batchnorm': 'yes', 'layers_final_activfunc': 'prelu', 'layer_final_dropout': 'yes', 'layer_final_batchnorm': 'yes'}
    params2={'l2_regularization_bool': True, 'learning_rate': 0.005854362632155896, 'l2_regularization_rate': 9.254662852565781e-05, 'batch_size': 64, 'dropout_prob': 0.1, 'dropout_bool': True, 'batchnorm_bool': False, 'prelu_rate': 0.75, 'layers_1_mRNA': 61, 'layers_2_mRNA': 17, 'layers_1_mRNA_activfunc': 'prelu', 'layers_2_mRNA_activfunc': 'sigmoid', 'layers_1_mRNA_dropout': 'yes', 'layers_2_mRNA_dropout': 'yes', 'layers_1_mRNA_batchnorm': 'yes', 'layers_2_mRNA_batchnorm': 'no', 'layers_1_microRNA': 48, 'layers_2_microRNA': 20, 'layers_1_microRNA_activfunc': 'sigmoid', 'layers_2_microRNA_activfunc': 'relu', 'layers_1_microRNA_dropout': 'no', 'layers_2_microRNA_dropout': 'yes', 'layers_1_microRNA_batchnorm': 'no', 'layers_2_microRNA_batchnorm': 'yes', 'layers_1_RPPA': 43, 'layers_2_RPPA': 8, 'layers_1_RPPA_activfunc': 'relu', 'layers_2_RPPA_activfunc': 'prelu', 'layers_1_RPPA_dropout': 'no', 'layers_2_RPPA_dropout': 'no', 'layers_1_RPPA_batchnorm': 'no', 'layers_2_RPPA_batchnorm': 'yes', 'layers_final_activfunc': 'sigmoid', 'layer_final_dropout': 'yes', 'layer_final_batchnorm': 'yes'}
    params3={'l2_regularization_bool': False, 'learning_rate': 1.614743919381929e-05, 'l2_regularization_rate': 1.7331266099143365e-05, 'batch_size': 64, 'dropout_prob': 0.1, 'dropout_bool': False, 'batchnorm_bool': True, 'prelu_rate': 0.55, 'layers_1_mRNA': 32, 'layers_2_mRNA': 31, 'layers_1_mRNA_activfunc': 'prelu', 'layers_2_mRNA_activfunc': 'sigmoid', 'layers_1_mRNA_dropout': 'yes', 'layers_2_mRNA_dropout': 'no', 'layers_1_mRNA_batchnorm': 'yes', 'layers_2_mRNA_batchnorm': 'no', 'layers_1_microRNA': 34, 'layers_2_microRNA': 19, 'layers_1_microRNA_activfunc': 'prelu', 'layers_2_microRNA_activfunc': 'sigmoid', 'layers_1_microRNA_dropout': 'no', 'layers_2_microRNA_dropout': 'no', 'layers_1_microRNA_batchnorm': 'yes', 'layers_2_microRNA_batchnorm': 'yes', 'layers_1_RPPA': 84, 'layers_2_RPPA': 16, 'layers_1_RPPA_activfunc': 'sigmoid', 'layers_2_RPPA_activfunc': 'prelu', 'layers_1_RPPA_dropout': 'yes', 'layers_2_RPPA_dropout': 'yes', 'layers_1_RPPA_batchnorm': 'no', 'layers_2_RPPA_batchnorm': 'no', 'layers_final_activfunc': 'prelu', 'layer_final_dropout': 'yes', 'layer_final_batchnorm': 'yes'}
    params4={'l2_regularization_bool': True, 'learning_rate': 9.698380620026325e-05, 'l2_regularization_rate': 0.00045565686931944074, 'batch_size': 64, 'dropout_prob': 0.1, 'dropout_bool': True, 'batchnorm_bool': True, 'prelu_rate': 0.8500000000000001, 'layers_1_mRNA': 46, 'layers_2_mRNA': 13, 'layers_1_mRNA_activfunc': 'relu', 'layers_2_mRNA_activfunc': 'prelu', 'layers_1_mRNA_dropout': 'no', 'layers_2_mRNA_dropout': 'no', 'layers_1_mRNA_batchnorm': 'no', 'layers_2_mRNA_batchnorm': 'yes', 'layers_1_microRNA': 72, 'layers_2_microRNA': 14, 'layers_1_microRNA_activfunc': 'relu', 'layers_2_microRNA_activfunc': 'prelu', 'layers_1_microRNA_dropout': 'yes', 'layers_2_microRNA_dropout': 'no', 'layers_1_microRNA_batchnorm': 'no', 'layers_2_microRNA_batchnorm': 'yes', 'layers_1_RPPA': 90, 'layers_2_RPPA': 22, 'layers_1_RPPA_activfunc': 'prelu', 'layers_2_RPPA_activfunc': 'prelu', 'layers_1_RPPA_dropout': 'no', 'layers_2_RPPA_dropout': 'yes', 'layers_1_RPPA_batchnorm': 'no', 'layers_2_RPPA_batchnorm': 'yes', 'layers_final_activfunc': 'none', 'layer_final_dropout': 'no', 'layer_final_batchnorm': 'no'}
    params5={'l2_regularization_bool': True, 'learning_rate': 0.0008428890111697013, 'l2_regularization_rate': 1.345416014099565e-06, 'batch_size': 32, 'dropout_prob': 0.30000000000000004, 'dropout_bool': False, 'batchnorm_bool': False, 'prelu_rate': 0.25, 'layers_1_mRNA': 79, 'layers_2_mRNA': 27, 'layers_1_mRNA_activfunc': 'sigmoid', 'layers_2_mRNA_activfunc': 'relu', 'layers_1_mRNA_dropout': 'yes', 'layers_2_mRNA_dropout': 'yes', 'layers_1_mRNA_batchnorm': 'no', 'layers_2_mRNA_batchnorm': 'no', 'layers_1_microRNA': 93, 'layers_2_microRNA': 32, 'layers_1_microRNA_activfunc': 'sigmoid', 'layers_2_microRNA_activfunc': 'relu', 'layers_1_microRNA_dropout': 'yes', 'layers_2_microRNA_dropout': 'yes', 'layers_1_microRNA_batchnorm': 'yes', 'layers_2_microRNA_batchnorm': 'no', 'layers_1_RPPA': 49, 'layers_2_RPPA': 23, 'layers_1_RPPA_activfunc': 'sigmoid', 'layers_2_RPPA_activfunc': 'relu', 'layers_1_RPPA_dropout': 'yes', 'layers_2_RPPA_dropout': 'no', 'layers_1_RPPA_batchnorm': 'yes', 'layers_2_RPPA_batchnorm': 'no', 'layers_final_activfunc': 'prelu', 'layer_final_dropout': 'yes', 'layer_final_batchnorm': 'yes'}
    list_of_params = [params,params5,params4,params3,params2]
    final_dict = averagingdictionaries(list_of_params)
    print(final_dict)
