import pandas as pd
import os
import numpy as np



if __name__ == '__main__':
    """
    Load in all Cancer Types and concatenate their views, duration and event respectively by samples.
    """

    direc_set = "SUMO" # Dir is Desktop for own CPU or SUMO for GPU

    # Path where initial cancer data is stored
    dir_path = os.path.expanduser('~/{}/Project/TCGAData'.format(dir))
    cancer_types = 0

    for base, dirs, files in os.walk(dir_path):
        for directories in dirs:
            cancer_types += 1


    # Different Cancer Types
    cancers = ['PRAD', 'ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH',
               'KIRC','KIRP','LAML','LGG','LIHC','LUAD','LUSC','MESO','PAAD','PCPG','READ','SARC','SKCM',
               'STAD','TGCT','THCA','THYM','UCEC' ,'UCS', 'UVM']
    # Store all cancer data (each in a sublist)
    cancer_data = [[] for x in range(len(cancers))]
    # Store Feature sizes per view
    n_features = [[] for x in range(len(cancers))]
    # Store which iews we have for the data (at most mRNA, DNA, microRNA and RPPA)
    views = [[] for x in range(len(cancers))]
    # Amount of samples per cancer
    samples = [[] for x in range(len(cancers))]
    # Features per cancer per view (needed for PPI network construction)
    features_per_cancer_per_view = [[] for x in range(len(cancers))]


    # Track non dropped indices (In case we want to delete some, not implemented currently)
    not_dropped_indices = [set() for x in range(len(cancers))]




    for c, cancer_name in enumerate(cancers):
        # Read mRNA data if it exists
        directory = os.path.expanduser('~/{}/Project/TCGAData/{}/TCGA_{}_1_mRNA.csv'.format(dir,cancer_name,cancer_name))
        if os.path.exists(os.path.join(directory)):
            data_mRNA = pd.read_csv(
                os.path.join(directory), index_col=0)
            samples[c].append(len(data_mRNA))
            # We need to reset indices bc meta data and view data have different index names
            data_mRNA.reset_index(drop=True,inplace=True)

            indices = list(data_mRNA.index.values)

            not_dropped_indices[c].update(indices)

            cancer_data[c].append(data_mRNA)
            n_features[c].append(len(data_mRNA.columns))
            features_per_cancer_per_view[c].append(list(data_mRNA.columns.values))
            views[c].append('mRNA')

        # Read DNA data if it exists
        directory = os.path.expanduser('~/{}/Project/TCGAData/{}/TCGA_{}_2_DNA.csv'.format(dir,cancer_name,cancer_name))
        if os.path.exists(os.path.join(directory)):
            data_DNA = pd.read_csv(
                os.path.join(directory), index_col=0)

            samples[c].append(len(data_DNA))
            data_DNA.reset_index(drop=True, inplace=True)

            indices = list(data_DNA.index.values)

            not_dropped_indices[c].update(indices)

            cancer_data[c].append(data_DNA)
            n_features[c].append(len(data_DNA.columns))
            features_per_cancer_per_view[c].append(list(data_DNA.columns.values))
            views[c].append('DNA')

        # Read microRNA data if it exists
        directory = os.path.expanduser('~/{}/Project/TCGAData/{}/TCGA_{}_3_miRNA.csv'.format(dir,cancer_name,cancer_name))
        if os.path.exists(os.path.join(directory)):
            data_miRNA = pd.read_csv(
                os.path.join(directory), index_col=0)
            samples[c].append(len(data_miRNA))
            data_miRNA.reset_index(drop=True, inplace=True)

            indices = list(data_miRNA.index.values)

            not_dropped_indices[c].update(indices)

            cancer_data[c].append(data_miRNA)
            n_features[c].append(len(data_miRNA.columns))
            features_per_cancer_per_view[c].append(list(data_miRNA.columns.values))
            views[c].append('microRNA')

        # Read RPPA data if it exists
        directory = os.path.expanduser('~/{}/Project/TCGAData/{}/TCGA_{}_4_RPPA.csv'.format(dir,cancer_name,cancer_name))
        if os.path.exists(os.path.join(directory)):
            data_RPPA = pd.read_csv(
                os.path.join(directory), index_col=0)



            samples[c].append(len(data_RPPA))
            data_RPPA.reset_index(drop=True, inplace=True)

            indices = list(data_RPPA.index.values)

            not_dropped_indices[c].update(indices)

            cancer_data[c].append(data_RPPA)
            n_features[c].append(len(data_RPPA.columns))
            features_per_cancer_per_view[c].append(list(data_RPPA.columns.values))
            views[c].append('RPPA')


        # Duration & Event data, which is stored in meta file,
        # needs to be available for each type, otherwise we can't do survival analysis
        directory = os.path.expanduser('~/{}/Project/TCGAData/{}/TCGA_{}_meta.csv'.format(dir,cancer_name,cancer_name))
        meta_data = pd.read_csv(
            os.path.join(directory), index_col=0, keep_default_na=False)
        # Rename vital status to event
        meta_data.rename(
            {"clin:vital_status": "event"}, axis="columns", inplace=True
        )

        meta_data.reset_index(drop=True, inplace=True)

        # For event = 0, we have days to last followup, for event = 1 days to death ; we store this data in a single
        # column called duration
        for co,x in enumerate(meta_data['clin:days_to_last_followup'].values):
            if x == '':
                meta_data['clin:days_to_last_followup'].iloc[co] = meta_data['clin:days_to_death'].iloc[co]




        meta_data.rename(columns = {'clin:days_to_last_followup':'duration'}, inplace = True)


        survival = meta_data[["event","duration"]]


        cancer_data[c].append(survival)
        n_features[c].append(1) # for event
        n_features[c].append(1) # for duration


    # Feature offsets for each cancer
    feature_offsets = []


    for c,_ in enumerate(n_features):
        feature_offsets.append([0] + np.cumsum(_).tolist())




    # Create dataframes, so we can create csv files for each cancer
    feature_offsets_dfs = []
    cancer_data_dfs = []
    view_names_dfs = []
    features_per_cancer_per_view_dfs = [[] for x in range(len(cancers))]
    # Dataframe for each cancers feature offsets
    for _ in feature_offsets:
        feature_offsets_dfs.append(pd.DataFrame(_))


    # Dataframe for each cancer data
    for _ in cancer_data:
        cancer_data_dfs.append(pd.concat(_, axis=1))


    # Dataframe for each cancer views
    for _ in views:
        view_names_dfs.append(pd.DataFrame(_))


    for c,cancer in enumerate(features_per_cancer_per_view):
        for view in cancer:
            features_per_cancer_per_view_dfs[c].append(pd.DataFrame(view))


    drop_indices = [[] for x in range(len(cancers))]
    # Indices to be dropped (currently we keep all data):
    for c,_ in enumerate(not_dropped_indices):
        # Go through all samples for the cancer type
        for x in range(samples[c][0]):
            if x not in _:
                drop_indices[c].append(x)



    # Drop samples with missing duration and unnecessary samples (duration < 0)
    for c,x in enumerate(cancer_data_dfs):
        x.drop(x.index[drop_indices[c]],inplace=True)
        x.drop(x[x["duration"] == ''].index, axis = 0, inplace= True) # Drop samples with missing duration values
        x.drop(x[x["duration"].isnull().values].index, axis = 0, inplace= True)
        x.drop(x[x["duration"] <= str(0)].index, axis = 0, inplace= True) # Drop samples with duration values smaller than or 0

        # In the end, reset indices
        x.reset_index(inplace=True, drop=True)





    # Storing all feature names without duplicates (needed for PPI-Network preparation)
    features = set()

    for cancer in cancer_data_dfs:
        features.update(list(cancer.columns))

    features_df = pd.DataFrame(features)


    # Create csv files
    for c,_ in enumerate(cancers):
        cancer_data_dfs[c].to_csv(("~/{}/Project/TCGAData/" + _ + "/" + _ + "Data.csv").format(dir))
        feature_offsets_dfs[c].to_csv(("~/{}/Project/TCGAData/" + _ +"/" + _ + "DataFeatOffsets.csv").format(dir))
        view_names_dfs[c].to_csv(("~/{}/Project/TCGAData/" + _ + "/" + _ + "Views.csv").format(dir))

        for c2,x in enumerate(views[c]):
            features_per_cancer_per_view_dfs[c][c2].to_csv(("~/{}/Project/TCGAData/" + _ + "/" + x + "FeatureNames.csv").format(dir))

    features_df.to_csv(("~/{}/Project/TCGAData/AllFeatures.csv").format(dir))






