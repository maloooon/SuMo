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
    multimodule = DataInputNew.SurvMultiOmicsDataModule(data, feature_offsets, view_names,onezeronorm_bool=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    NN.train(module= multimodule,
          device=device,
          batch_size=256,
          n_epochs=100,
          lr_scheduler_type='lambda',
          l2_regularization='yes',
          val_batch_size=64,
          number_folds= 5)

