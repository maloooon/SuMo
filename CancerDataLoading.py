import pandas as pd
import os
import torch
import numpy as np

"""

# DataSpellProjectsForSAMO/SAMO/TCGAData/PRAD
    data_transform = pd.read_csv(
        os.path.join("/Users", "marlon", "DataspellProjectsForSAMO", "SAMO", "TCGAData", "PRAD",
                     "1.tsv"), index_col=0, sep='\t'
    )

    names_mRNA = (data_transform["Entry Name"])
    #names_mRNA.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/mRNA_transf.csv")

    #names_mRNA_tensor = torch.tensor(names_mRNA.values)


"""

if __name__ == '__main__':


    # Count amount of cancer types we have


    dir_path = '/Users/marlon/Desktop/Project/TCGAData'
    cancer_types = 0

    for base, dirs, files in os.walk(dir_path):
        for directories in dirs:
            cancer_types += 1


    # different types ; TODO : deleted OV as it has only meta data
    cancers = ['PRAD', 'ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH',
               'KIRC','KIRP','LAML','LGG','LIHC','LUAD','LUSC','MESO','PAAD','PCPG','READ','SARC','SKCM',
               'STAD','TGCT','THCA','THYM','UCEC' ,'UCS', 'UVM']
    # store all cancer data (each in a sublist)
    cancer_data = [[] for x in range(len(cancers))]
    # store feature sizes per view
    n_features = [[] for x in range(len(cancers))]
    # store which views we have for the data (at most mRNA, DNA, microRNA and RPPA
    views = [[] for x in range(len(cancers))]
    # amount of samples
    samples = [[] for x in range(len(cancers))]


    # Track non dropped indices, which are these that have atleast 100 not NaN feature values in a sample for each view
    not_dropped_indices = [set() for x in range(len(cancers))]

    for c, cancer_name in enumerate(cancers):
        if os.path.exists(os.path.join("/Users", "marlon", "Desktop", "Project", "TCGAData", cancer_name,
                                       "TCGA_" + cancer_name + "_1_mRNA.csv")):
            data_mRNA = pd.read_csv(
                os.path.join("/Users", "marlon", "Desktop", "Project", "TCGAData", cancer_name,
                             "TCGA_" + cancer_name + "_1_mRNA.csv"), index_col=0)
            samples[c].append(len(data_mRNA))
            # We need to reset indices bc meta data and view data have different index names
            data_mRNA.reset_index(drop=True,inplace=True)

            temp = data_mRNA.dropna(thresh=100)
            indices = list(temp.index.values)

            not_dropped_indices[c].update(indices)

            cancer_data[c].append(data_mRNA)
            n_features[c].append(len(data_mRNA.columns))
            views[c].append('mRNA')


        if os.path.exists(os.path.join("/Users", "marlon", "Desktop", "Project", "TCGAData", cancer_name,
                                       "TCGA_" + cancer_name + "_2_DNA.csv")):
            data_DNA = pd.read_csv(
                os.path.join("/Users", "marlon", "Desktop", "Project", "TCGAData", cancer_name,
                             "TCGA_" + cancer_name + "_2_DNA.csv"), index_col=0)

            samples[c].append(len(data_DNA))
            data_DNA.reset_index(drop=True, inplace=True)

            temp = data_DNA.dropna(thresh=100)
            indices = list(temp.index.values)

            not_dropped_indices[c].update(indices)

            cancer_data[c].append(data_DNA)
            n_features[c].append(len(data_DNA.columns))
            views[c].append('DNA')


        if os.path.exists(os.path.join("/Users", "marlon", "Desktop", "Project", "TCGAData", cancer_name,
                                       "TCGA_" + cancer_name + "_3_miRNA.csv")):
            data_miRNA = pd.read_csv(
                os.path.join("/Users", "marlon", "Desktop", "Project", "TCGAData", cancer_name,
                             "TCGA_" + cancer_name + "_3_miRNA.csv"), index_col=0)
            samples[c].append(len(data_miRNA))
            data_miRNA.reset_index(drop=True, inplace=True)

            temp = data_miRNA.dropna(thresh=10)
            indices = list(temp.index.values)

            not_dropped_indices[c].update(indices)

            cancer_data[c].append(data_miRNA)
            n_features[c].append(len(data_miRNA.columns))
            views[c].append('microRNA')


        if os.path.exists(os.path.join("/Users", "marlon", "Desktop", "Project", "TCGAData", cancer_name,
                                       "TCGA_" + cancer_name + "_4_RPPA.csv")):
            data_RPPA = pd.read_csv(
                os.path.join("/Users", "marlon", "Desktop", "Project", "TCGAData", cancer_name,
                             "TCGA_" + cancer_name + "_4_RPPA.csv"), index_col=0)



            samples[c].append(len(data_RPPA))
            data_RPPA.reset_index(drop=True, inplace=True)


            temp = data_RPPA.dropna(thresh=10)
            indices = list(temp.index.values)

            not_dropped_indices[c].update(indices)

            cancer_data[c].append(data_RPPA)
            n_features[c].append(len(data_RPPA.columns))
            views[c].append('RPPA')


        # Meta data needs to be available for each type, otherwise we can't do survival analysis
        meta_data = pd.read_csv(
            os.path.join("/Users", "marlon", "Desktop", "Project", "TCGAData", cancer_name,
                         "TCGA_" + cancer_name + "_meta.csv"), index_col=0, keep_default_na=False)
        meta_data.rename(
            {"clin:vital_status": "event"}, axis="columns", inplace=True
        )

        meta_data.reset_index(drop=True, inplace=True)

        for co,x in enumerate(meta_data['clin:days_to_last_followup'].values):
            if x == '':
                meta_data['clin:days_to_last_followup'].iloc[co] = meta_data['clin:days_to_death'].iloc[co]




        meta_data.rename(columns = {'clin:days_to_last_followup':'duration'}, inplace = True)



        # TODO : how to include other meta data like smoking time ?

        survival = meta_data[["event","duration"]]






        cancer_data[c].append(survival)
        n_features[c].append(1) # for event
        n_features[c].append(1) # for duration


    # Feature offsets for each cancer
    feature_offsets = []


    for c,_ in enumerate(n_features):
        feature_offsets.append([0] + np.cumsum(_).tolist())




    # Dataframes, so we can send to csv files and not load all cancer data each time

    feature_offsets_dfs = []
    cancer_data_dfs = []
    view_names_dfs = []
    # Dataframe for each cancers feature offsets
    for _ in feature_offsets:
        feature_offsets_dfs.append(pd.DataFrame(_))


    # Dataframe for each cancer data
    for _ in cancer_data:
        cancer_data_dfs.append(pd.concat(_, axis=1))


    # Dataframe for each cancer views
    for _ in views:
        view_names_dfs.append(pd.DataFrame(_))


    drop_indices = [[] for x in range(len(cancers))]
    # Indices to be dropped bc. of too many missing values :
    for c,_ in enumerate(not_dropped_indices):
        # Go through all samples for the cancer type
        for x in range(samples[c][0]):
            if x not in _:
                drop_indices[c].append(x)



    # Drop samples with missing duration and unnecessary samples (duration < 0)
    for c,x in enumerate(cancer_data_dfs):
        x.drop(x.index[drop_indices[c]],inplace=True)
        x.drop(x[x["duration"] == ''].index, axis = 0, inplace= True) # drop samples with missing duration values
        x.drop(x[x["duration"].isnull().values].index, axis = 0, inplace= True)
        x.drop(x[x["duration"] <= str(0)].index, axis = 0, inplace= True) # drop samples with duration values smaller than or 0

        # In the end, reset indices
        x.reset_index(inplace=True, drop=True)


        # Drop every row which doesn't have atleast 100 real values (not NaN values)
        # TODO : best thresh ?
     #   x.dropna(thresh=100, inplace=True)




    # Storing all feature names without duplicates (needed for PPI-Network preparation)
    features = set()

    for cancer in cancer_data_dfs:
        features.update(list(cancer.columns))

    features_df = pd.DataFrame(features)

    # Load all to csv's
    for c,_ in enumerate(cancers):
        cancer_data_dfs[c].to_csv("/Users/marlon/Desktop/Project/TCGAData/" + _ + "/" + _ + "Data.csv")
        feature_offsets_dfs[c].to_csv("/Users/marlon/Desktop/Project/TCGAData/" + _ +"/" + _ + "DataFeatOffsets.csv")
        view_names_dfs[c].to_csv("/Users/marlon/Desktop/Project/TCGAData/" + _ + "/" + _ + "Views.csv")

    features_df.to_csv("/Users/marlon/Desktop/Project/TCGAData/AllFeatures.csv")




