import os
import pandas as pd
import HelperFunctions as HF


def readcancerdata(cancer_name):
    """
    Read in cancer data of a particular cancer for analysis.
    :param cancer_name: abbreviation of the cancer name ; dtype : String
    :return: Cancer data, which contains feature values for each view, duration, event,
     data offsets, view names, feature names and the cancer name to be analyzed
     ; dtype : List of (Dataframe, Dataframe, List, List, List)
    """

    direc_set = "SUMO" # Dir is Desktop for own CPU or SUMO for GPU

    cancer_names = [cancer_name]


    cancer_data = [[] for x in range(len(cancer_names))]

    for c,_ in enumerate(cancer_names):
        print("Reading in " + str(_) + " data...")
        data = pd.read_csv(
            os.path.join("~", direc_set, "Project", "TCGAData", str(_),
                         str(_) + "Data.csv"), index_col=0)

        print("Reading in " + str(_) + " offsets...")
        feat_offset = pd.read_csv(
            os.path.join("~", direc_set, "Project", "TCGAData", str(_),
                         str(_) + "DataFeatOffsets.csv"), index_col=0)

        print("Reading in " + str(_) + " view names...")
        view_names = pd.read_csv(
            os.path.join("~", direc_set, "Project", "TCGAData", str(_),
                         str(_) + "Views.csv"), index_col=0)

        views = HF.flatten(view_names.values.tolist())
        print("Reading in " + str(_) + " feature names...")
        names = []
        for c2,view in enumerate(views):
            feat_names = pd.read_csv(
                os.path.join("~", direc_set, "Project", "TCGAData", str(_),
                             str(view) + "FeatureNames.csv"), index_col=0)
            names.append((views[c2],HF.flatten(feat_names.values.tolist())))


        cancer_data[c].append(data)
        cancer_data[c].append(HF.flatten(feat_offset.values.tolist()))
        cancer_data[c].append(views)
        cancer_data[c].append(names)
        cancer_data[c].append(cancer_names)


    return cancer_data

