import pandas as pd
import os
import torch
import numpy as np

if __name__ == '__main__':


    data_transform = pd.read_csv(
        os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                     "1.tsv"), index_col=0, sep='\t'
    )

    names_mRNA = (data_transform["Entry Name"])
    #names_mRNA.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/mRNA_transf.csv")

    #names_mRNA_tensor = torch.tensor(names_mRNA.values)




    #if __name__ == '__main__':

    data_mRNA = pd.read_csv(
        os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                     "TCGA_PRAD_1_mrna.csv"), index_col=0
    )
    data_DNA = pd.read_csv(
        os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                     "TCGA_PRAD_2_dna.csv"), index_col=0
    )
    data_microRNA = pd.read_csv(
        os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                     "TCGA_PRAD_3_microrna.csv"), index_col=0
    )
    data_RPPA = pd.read_csv(
        os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                     "TCGA_PRAD_4_rppa.csv"), index_col=0
    )
    data_survival = pd.read_csv(
        os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                     "TCGA_PRAD_meta.csv"), index_col=0
    )


    data_survival.rename(
        {"vital_status": "event", "duration": "duration"}, axis="columns", inplace=True
    )
    data_survival.drop(labels= 'cancer_type', axis='columns', inplace=True)

    data_event = data_survival.iloc[:, 0]
    data_duration = data_survival.iloc[:, 1]


    n_samples = data_survival.shape[0]
    n_features = [len(data_mRNA.columns),
                  len(data_DNA.columns),
                  len(data_microRNA.columns),
                  len(data_RPPA.columns),
                  1,                    # event (no feature)
                  1]                    # duration (no feature)
    #print(n_features) # 6000, 6000, 336, 148

    # cumulative sum of features in list
    feature_offsets = [0] + np.cumsum(n_features).tolist()
    #print("foff", feature_offsets) # 0,6000,12000,12336,12484,12485,12486

    feat_offset_df = pd.DataFrame(feature_offsets)

    # TODO : feature selection in module

    df_all = pd.concat([data_mRNA, data_DNA, data_microRNA, data_RPPA, data_survival], axis=1)


    # drop unnecessary samples
    df_all.drop(df_all[df_all["duration"] <= 0].index, axis = 0, inplace= True)



    # Load into csv

    df_all.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/PRADData.csv")
    feat_offset_df.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/PRADDataFeatOffsets.csv")

