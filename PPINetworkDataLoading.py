import pandas as pd
import os
import torch
import DataInputNew



if __name__ == '__main__':
    # Get features without numbers (needed for conversion) ; also remove | and - from strings

    features_mRNA_no_numbers = []
    for x in DataInputNew.features[0]:
        index = x.find('|') # delete characters starting at | as the online tool won't work with these
        features_mRNA_no_numbers.append(x[:index])

    features_DNA_no_numbers = []
    for x in DataInputNew.features[1]:
        index = x.find('|')
        features_DNA_no_numbers.append(x[:index])



    # Index features (all features, no matter if mapping or not) so we can later on combine feature values of samples to proteins
    feature_mRNA_indexed = {}
    for index in range(len(features_mRNA_no_numbers)):
        feature_mRNA_indexed[index] = features_mRNA_no_numbers[index]



    # features to csv file
    df_mRNA_features_no_numbers = pd.DataFrame(features_mRNA_no_numbers)
    df_DNA_features_no_number = pd.DataFrame(features_DNA_no_numbers)

    df_mRNA_features_no_numbers.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/mRNA_feat_for_transf.csv",
                                       index=True)
    df_DNA_features_no_number.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/DNA_feat_for_transf.csv",
                                     index=True)




    # Load proteins and interactions

    ppi_data = pd.read_csv(os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                                        "pp.txt"), sep=" ", header=0)





    # drop scores below 700 (like in DeepMOCCA)
    ppi_data.drop(ppi_data[(ppi_data["combined_score"]) < 700].index, axis = 0, inplace= True)

    #reset indices
    ppi_data = ppi_data.reset_index(drop=True)

    #set so we can remove duplicates
    proteins_data = list(set(ppi_data["protein1"].values))   # list of all proteins we look at

    #check if protein 2 has proteins that are not in protein1 and add them if needed
    for x in set(ppi_data["protein2"].values):
        if x not in proteins_data:
            proteins_data.append(x)



    #create dictionary so we can use the index more easily

    dict_proteins_data = {}
    for index in range(len(proteins_data)):
        dict_proteins_data[index] = proteins_data[index]

    #switch key and values (to get same data structure as in DeepMOCCA) (protein name: index)
    dict_proteins_data = {y: x for x, y in dict_proteins_data.items()}

    df_dict_proteins_data = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in dict_proteins_data.items()]))
    df_dict_proteins_data.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/proteins.csv"
                                 ,index=True)
    # Edges in PPI network and according score

    node1 = []
    node2 = []
    score = []



    for index, row in ppi_data.iterrows():
        node1.append(dict_proteins_data[row["protein1"]])
        node2.append(dict_proteins_data[row["protein2"]])
        score.append(row["combined_score"])

    temp = [node1,node2,score]

    # tensor with rows : node 1, node 2, score
    ppi_edges_scores = torch.tensor(temp)

    # Convert to dataframe and create .csv-file, as we will need this for the creation of the adjacency matrix for
    # the PPI-network
    ppi_edges_scores_dataframe = pd.DataFrame(ppi_edges_scores.numpy())
    ppi_edges_scores_dataframe.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/ppi_edges_score.csv",
                                      index=True)






    # Mapping HGNC to uniprot and uniprot to proteins : https://www.uniprot.org/id-mapping
    # Mapping feature names to HGNC : https://www.syngoportal.org/convert



    # feature to HGNC mapping
    mRNA_HGNC_full = pd.read_excel(
        os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                     "mRNA_to_HGNC.xlsx"), index_col=0
    )

    mRNA_HGNC_full["features"] = mRNA_HGNC_full.index

    mRNA_features_only = mRNA_HGNC_full["features"]

    #Needed for conversion online
    mRNA_HGNC_only = mRNA_HGNC_full["HGNC"]

    mRNA_HGNC_only.to_csv(
        os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                     "HGNC_only"),index= False
    )


    mRNA_HGNC_dict = {}

    # Dictionary ordering features to HGNC
    for feature in mRNA_features_only:
        mRNA_HGNC_dict[feature] = mRNA_HGNC_full.loc[mRNA_HGNC_full["features"] == feature]["HGNC"].values

    # turn arrays into lists in dict
    for key in mRNA_HGNC_dict:
        mRNA_HGNC_dict[key] = mRNA_HGNC_dict[key].tolist()

    counter = 0

    # Remove features with no HGNC value
    for key in list(mRNA_HGNC_dict.keys()):
        if type(mRNA_HGNC_dict[key][0]) is not str: # no HGNC value : nan as value to key, so not a string
            counter += 1
            del mRNA_HGNC_dict[key]


    print("We have {} successful mappings from features to HGNC. {} could not be mapped.".format(len(mRNA_HGNC_dict), counter))









    #HGNC to uniprot mapping

    uniprot_HGNC_mRNA_full = pd.read_csv(
        os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                     "HGNC_mRNA_to_uniprot.tsv"), index_col=0, sep='\t'
    )

    uniprot_HGNC_mRNA_HGNC_only = uniprot_HGNC_mRNA_full.index
    uniprot_HGNC_mRNA_HGNC_only = list(uniprot_HGNC_mRNA_HGNC_only)

    uniprot_HGNC_mRNA_dict = {}

    for HGNC in uniprot_HGNC_mRNA_HGNC_only:
        uniprot_HGNC_mRNA_dict[HGNC] = uniprot_HGNC_mRNA_full.loc[uniprot_HGNC_mRNA_full.index == HGNC]["Entry"].values


    # turn arrays into lists in dict
    for key in uniprot_HGNC_mRNA_dict:
        uniprot_HGNC_mRNA_dict[key] = uniprot_HGNC_mRNA_dict[key].tolist()

    # Needed for further conversion
    uniprot_HGNC_mRNA_full_entries = uniprot_HGNC_mRNA_full["Entry"] # Entry columns has uniprot values

    uniprot_HGNC_mRNA_full_entries.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/uniprot_mRNA_transf.csv", index=False)

    #counter = 0
    # Remove HGNC-values which couldn't be mapped to uniprot
    #for key in list(uniprot_HGNC_mRNA_dict.keys()):
    #    if len(uniprot_HGNC_mRNA_dict[key]) == 0:
    #        del uniprot_HGNC_mRNA_dict[key]
    #        counter += 1

    print("We have {} successful mappings from HGNC to uniprots.".format(len(uniprot_HGNC_mRNA_dict)))













    #Finally, uniprot values to proteins
    uniprot_to_proteins_mRNA_full = pd.read_csv(
        os.path.join("/Users", "marlon", "DataspellProjects", "MuVAEProject", "MuVAE", "TCGAData",
                     "uniprot_to_protein_mRNA.tsv"), index_col=0, sep="\t"
    )

    uniprot_to_proteins_mRNA_dict = {}


    uniprot_to_proteins_mRNA_uniprot_only = uniprot_to_proteins_mRNA_full.index
    uniprot_to_proteins_mRNA_uniprot_only = list(uniprot_to_proteins_mRNA_uniprot_only)

    proteins = list(uniprot_to_proteins_mRNA_full["To"].values)
    #save proteins and give them indices
    dict_index_proteins = {}

    for x in range(len(proteins)):
        dict_index_proteins[x] = proteins[x]



    for uniprot in uniprot_to_proteins_mRNA_uniprot_only:
        uniprot_to_proteins_mRNA_dict[uniprot] = uniprot_to_proteins_mRNA_full.loc[uniprot_to_proteins_mRNA_full.index == uniprot]["To"].values

    # turn arrays into lists in dict
    for key in uniprot_to_proteins_mRNA_dict:
        uniprot_to_proteins_mRNA_dict[key] = uniprot_to_proteins_mRNA_dict[key].tolist()

    #counter = 0
    # Remove HGNC-values which couldn't be mapped to uniprot        #TODO : diese funktionen könnten key errors bei dicts zusammenfügen später erzeugen!
    #for key in list(uniprot_to_proteins_mRNA_dict.keys()):
    #    if len(uniprot_to_proteins_mRNA_dict[key]) == 0:
    #        del uniprot_to_proteins_mRNA_dict[key]
    #        counter += 1


    print("We have {} successful mappings from uniprots to proteins.".format(len(uniprot_to_proteins_mRNA_dict)))






    # Map our features from the beginning to the proteins (the ones that could be mapped : only these will be a part of
    # the GCN

    #first from dict(HGNC: uniprot) and dict(uniprot : proteins) to dict(HGNC : proteins)

    dict_HGNC_proteins = {}

    # Intialize empty dictionary for each HGNC value as key

    for HGNC in uniprot_HGNC_mRNA_HGNC_only:
        dict_HGNC_proteins[HGNC] = []

    # fill dictionary

    for key in uniprot_to_proteins_mRNA_dict:
        for key2 in uniprot_HGNC_mRNA_dict:
            if key in uniprot_HGNC_mRNA_dict[key2]:
                dict_HGNC_proteins[key2].append(uniprot_to_proteins_mRNA_dict[key])


    print("We have {} mappings from HGNC to proteins".format(len(dict_HGNC_proteins)))


    # now from dict(HGNC:proteins) and dict(feature:HGNC) to dict(feature:proteins) (by applying the same logic)

    dict_features_proteins_mRNA = {}

    for features in mRNA_features_only:
        dict_features_proteins_mRNA[features] = []


    def flatten(l):
        return [item for sublist in l for item in sublist]



    for key in dict_HGNC_proteins:
        for key2 in mRNA_HGNC_dict:
            if key in mRNA_HGNC_dict[key2]:

                dict_features_proteins_mRNA[key2] = (flatten(dict_HGNC_proteins[key]))


    # todo : pandas rename dict



    counter = 0
    # Remove features with no protein connection
    for key in list(dict_features_proteins_mRNA.keys()):
        if len(dict_features_proteins_mRNA[key]) == 0:
            counter += 1
            del dict_features_proteins_mRNA[key]




    # Note that some features are connected to multiple proteins !
    print("We have {} mappings from features to proteins. {} could not be mapped".format(len(dict_features_proteins_mRNA),counter))




    # Create a dataframe so we can save this file as a .csv and use it later on
    features_proteins_dataframe = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in dict_features_proteins_mRNA.items()]))
    features_proteins_dataframe.to_csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/mRNA_feat_proteins.csv",
                                       index=True)








    # set feature values to proteins, sample wise
    train_loader = DataInputNew.multimodule.train_dataloader(batch_size = 20)


    sample_to_protein = []
    # For each sample, make a dictionary with protein : feature value ; save list of dictionaries
    features_proteins_mRNA_values_all = []
    #print("train ", data)
    #print("train 0", data[0])
    #print("sample 0", data[0][0])
    #print("sample0 first value", data[0][0][0])
    #mRNA

    print(features_mRNA_no_numbers)
    print(dict_features_proteins_mRNA)


    for data,mask, duration, event in train_loader:


        # Go through each mRNA sample
        for sample in data[0]:
            # for each sample dictionary
            dict_features_proteins_mRNA_values = {}
            # Through each feature (for mRNA)
            for feature_idx in range(len(sample)): # len = 6000 features
                temp = []
                # If the feature has a mapping in feature to protein
                # features_mRNA_no_numbers[feature_idx] : feature name at index
                if features_mRNA_no_numbers[feature_idx] in dict_features_proteins_mRNA:


                    # take the proteins at the current index which correspond to the feature we are looking at
                    protein_list = dict_features_proteins_mRNA[features_mRNA_no_numbers[feature_idx]]

                    # create dictionary entries for it
                    for protein in protein_list:
                        #if key already exists, just append values to protein
                        if protein in dict_features_proteins_mRNA_values:
                            dict_features_proteins_mRNA_values[protein].append(sample[feature_idx])
                        #else create a key entry
                        else:

                            dict_features_proteins_mRNA_values[protein] = [sample[feature_idx]]


            # for each sample, append to list
            features_proteins_mRNA_values_all.append(dict_features_proteins_mRNA_values)









    # Create graph representation

    # feature matrix X |A| x |B| A : number of nodes (proteins), B : number of features per node (value features diff. views)
    # adjacency matrix Z : |A| x |A|

    #adjacency matrix for protein-protein-network
    adjacency_matrix_ppi = torch.zeros(len(proteins_data), len(proteins_data))


    for protein_idx in range(len(proteins_data)):
        # for each edge between two nodes, fill in a 1 in adjacency matrix
        adjacency_matrix_ppi[int((ppi_edges_scores[0, protein_idx]).item()), int((ppi_edges_scores[1, protein_idx]).item())] = torch.tensor(1)

    # todo : adjazenz mit gewichten ?


    # feature matrix for each sample
    #torch tensor with A x B : rows x columns
    # Initialize
    feature_matrices_mRNA = []
    counter = 0
    #for each sample (we could also use smth diff then len(features...) to access all samples
    for sample_idx in range(len(features_proteins_mRNA_values_all)):

        # As column size for the tensor, we use the protein (node) which has the most features (values from diff views)
        # We use the largest as the torch tensor is basically a matrix and we need a fixed size
        # to test, just implemented for mRNA
        most_feat_for_protein = len(max(features_proteins_mRNA_values_all[sample_idx].values(), key=len))

        # As rows, we just use all the proteins ; matrix for one sample
        feature_matrix_mRNA = torch.zeros(len(proteins_data), most_feat_for_protein)

        # fill with data

        # go through each protein
        for protein in proteins_data:
            # check if protein in protein :feature values dict
            if protein in features_proteins_mRNA_values_all[sample_idx]:

                # fill row with according data ; find right row by accesing dict protein data (find the right index)

                # Find index by accessing dict which saves pairs of protein : index
                index = dict_proteins_data[protein]
                feature_matrix_mRNA[index, 0:len(features_proteins_mRNA_values_all[sample_idx][protein])] = \
                    torch.tensor(features_proteins_mRNA_values_all[sample_idx][protein])



            #else : we leave it as it is (filled with 0s)

        feature_matrices_mRNA.append(feature_matrix_mRNA)
    #   Todo : normalization before adding into matrices (see DeepMOCCA)



    #print(feature_matrix_mRNA[0])
    #print(feature_matrix_mRNA.shape) # 16814,3







    # each protein node has features from different views (features of that node, in deepmocca sind das die 1-8 ?
    # bei mir dann mRNA und DNA ?
    # input gcn ist dann das PPI : jeder Node hat best. features und wir kennen die edges


#%%
