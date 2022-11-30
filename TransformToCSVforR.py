import DataInputNew
import pandas as pd
import torch


if __name__ == '__main__':

    train_loader = DataInputNew.multimodule.train_dataloader(batch_size= 10)
    for data,mask, duration, event in train_loader:
        data_for_r_mRNA = pd.DataFrame(data[0].numpy(), columns= DataInputNew.features[0])
        data_for_r_DNA = pd.DataFrame(data[1].numpy(), columns= DataInputNew.features[1])
        data_for_r_microRNA = pd.DataFrame(data[2].numpy(), columns= DataInputNew.features[2])
        data_for_r_RPPA = pd.DataFrame(data[3].numpy(), columns= DataInputNew.features[3])
        break

    print(len(data_for_r_RPPA.columns))
    # TODO : nochmal neu csv files laden ; batch size 10 bei RPPA, bei Rest alle 389 (alle train cases)


  #  data_for_r_RPPA.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/RPPA_for_r.csv")
  #  data_for_r_mRNA.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/mRNA_for_r.csv")
  #  data_for_r_DNA.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/DNA_for_r.csv")
  #  data_for_r_microRNA.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/microRNA_for_r.csv")