import NN
import AE
import GCN
import ReadInData
import DataInputNew
import torch

# GOOD RESULTS : Feature Selection AE, First AE cross, second AE concat

if __name__ == '__main__':
    cancer_data = ReadInData.readcancerdata('LUAD')
    data = cancer_data[0][0]
    feature_offsets = cancer_data[0][1]
    view_names = cancer_data[0][2]
    feature_names = cancer_data[0][3]
    multimodule = DataInputNew.SurvMultiOmicsDataModule(data, feature_offsets, view_names,onezeronorm_bool=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    NN.train(module= multimodule,
          device=device,
          batch_size=256,
          n_epochs=100,
          l2_regularization=False,
          val_batch_size=64,
          number_folds= 3)


    GCN.train(module= multimodule,
              device=device,
              batch_size=256,
              n_epochs=100,
              l2_regularization=False,
              val_batch_size=64,
              number_folds=3,
              feature_names=feature_names)


    AE.train(module=multimodule,
             device=device,
             batch_size=256,
             n_epochs=100,
             l2_regularization=False,
             val_batch_size=64,
             number_folds=3)

