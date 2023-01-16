import os
import pandas as pd
import HelperFunctions as HF
#def flatten(l):
#    """
#    :param l: list input
#    :return: flattened list (removal of one inner list layer)
#    """
#    return [item for sublist in l for item in sublist]



def readcancerdata(cancer_name):
    # Load in cancer data, its feature offsets and views used
    # different types # TODO : removed OV
    cancer_names = ['PRAD', 'ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH',
                    'KIRC','KIRP','LAML','LGG','LIHC','LUAD','LUSC','MESO','PAAD','PCPG','READ','SARC','SKCM',
                    'STAD','TGCT','THCA','THYM','UCEC' ,'UCS', 'UVM']

    # Testing purposes :
    cancer_names = ['STAD']

    # With input variant
    cancer_names = [cancer_name]


    cancer_data = [[] for x in range(len(cancer_names))]

    for c,_ in enumerate(cancer_names):
        print("Reading in " + str(_) + " data...")
        data = pd.read_csv(
            os.path.join("/Users", "marlon", "Desktop", "Project", "TCGAData", str(_),
                         str(_) + "Data.csv"), index_col=0)
        print("Reading in " + str(_) + " offsets...")
        feat_offset = pd.read_csv(
            os.path.join("/Users", "marlon", "Desktop", "Project", "TCGAData", str(_),
                         str(_) + "DataFeatOffsets.csv"), index_col=0)
        print("Reading in " + str(_) + " view names...")
        view_names = pd.read_csv(
            os.path.join("/Users", "marlon", "Desktop", "Project", "TCGAData", str(_),
                         str(_) + "Views.csv"), index_col=0)

        views = HF.flatten(view_names.values.tolist())
        print("Reading in " + str(_) + " feature names...")
        names = []
        for view in views:
            feat_names = pd.read_csv(
                os.path.join("/Users", "marlon", "Desktop", "Project", "TCGAData", str(_),
                             str(view) + "FeatureNames.csv"), index_col=0)
            names.append(HF.flatten(feat_names.values.tolist()))

        cancer_data[c].append(data)
        cancer_data[c].append(HF.flatten(feat_offset.values.tolist()))
        cancer_data[c].append(views)
        cancer_data[c].append(names)


    return cancer_data



        # Call module for each cancer type
    #  multimodules = []

    #  for c,_ in enumerate(cancer_names):
    #      multimodules.append(SurvMultiOmicsDataModule(cancer_data[c][0],cancer_data[c][1],cancer_data[c][2]))

    #  multimodules[0].setup()