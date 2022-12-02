import DataInputNew
import pandas as pd
import torch


if __name__ == '__main__':

    train_loader = DataInputNew.multimodule.train_dataloader(batch_size= 10)
    for data,mask, duration, event in train_loader:
        data_for_r_mRNA = pd.DataFrame(data[0].numpy()) #columns= DataInputNew.features[0])
        data_for_r_DNA = pd.DataFrame(data[1].numpy(), columns= DataInputNew.features[1])
        data_for_r_microRNA = pd.DataFrame(data[2].numpy(), columns= DataInputNew.features[2])
        data_for_r_RPPA = pd.DataFrame(data[3].numpy(), columns= DataInputNew.features[3])
        data_RPPA = data[3]
        mask_RPPA = mask[3]
        data_mRNA = data[0]
        mask_mRNA = mask[0]
        data_DNA = data[1]
        mask_DNA = data[1]
        data_microRNA = data[2]
        mask_microRNA = mask[2]

        break




    # drop columns with only null values
    to_drop_columns_RPPA = []
    to_drop_columns_mRNA = []
    to_drop_columns_DNA = []
    to_drop_columns_microRNA = []

    for x in range(len(data_for_r_RPPA.columns)):
        if (data_for_r_RPPA.iloc[:, x] == data_for_r_RPPA.iloc[:, x][0]).all():

            to_drop_columns_RPPA.append(x)

    data_for_r_RPPA = data_for_r_RPPA.drop(data_for_r_RPPA.columns[to_drop_columns_RPPA], axis = 1)

    for x in range(len(data_for_r_mRNA.columns)):
        if (data_for_r_mRNA.iloc[:, x] == data_for_r_mRNA.iloc[:, x][0]).all():

            to_drop_columns_mRNA.append(x)

    data_for_r_mRNA = data_for_r_mRNA.drop(data_for_r_mRNA.columns[to_drop_columns_mRNA], axis = 1)


    for x in range(len(data_for_r_DNA.columns)):
        if (data_for_r_DNA.iloc[:, x] == data_for_r_DNA.iloc[:, x][0]).all():

            to_drop_columns_DNA.append(x)

    data_for_r_DNA = data_for_r_DNA.drop(data_for_r_DNA.columns[to_drop_columns_DNA], axis = 1)


    for x in range(len(data_for_r_microRNA.columns)):
        if (data_for_r_microRNA.iloc[:, x] == data_for_r_microRNA.iloc[:, x][0]).all():

            to_drop_columns_microRNA.append(x)

    data_for_r_microRNA = data_for_r_microRNA.drop(data_for_r_microRNA.columns[to_drop_columns_microRNA], axis = 1)



    # drop rows with only null values
    to_drop_index_RPPA = []
    to_drop_index_mRNA = []
    to_drop_index_DNA = []
    to_drop_index_microRNA = []

    for x in range(len(data_for_r_RPPA.index)):
        if torch.all(mask_RPPA[x] == True):

            to_drop_index_RPPA.append(x)

    data_for_r_RPPA = data_for_r_RPPA.drop(to_drop_index_RPPA, axis = 0)


    for x in range(len(data_for_r_mRNA.index)):
        if torch.all(mask_mRNA[x] == True):

            to_drop_index_mRNA.append(x)

    data_for_r_mRNA = data_for_r_mRNA.drop(to_drop_index_mRNA, axis = 0)

    for x in range(len(data_for_r_DNA.index)):
        if torch.all(mask_DNA[x] == True):

            to_drop_index_DNA.append(x)

    data_for_r_DNA = data_for_r_DNA.drop(to_drop_index_DNA, axis = 0)


    for x in range(len(data_for_r_microRNA.index)):
        if torch.all(mask_microRNA[x] == True):

            to_drop_index_microRNA.append(x)

    data_for_r_microRNA = data_for_r_microRNA.drop(to_drop_index_microRNA, axis = 0)



   # print(len(data_for_r_RPPA.columns))
    # TODO : nochmal neu csv files laden ; batch size 10 bei RPPA, bei Rest alle 389 (alle train cases)


    data_for_r_RPPA.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/RPPA_for_r.csv")
    data_for_r_mRNA.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/mRNA_for_r.csv")
    data_for_r_DNA.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/DNA_for_r.csv")
    data_for_r_microRNA.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/microRNA_for_r.csv")

