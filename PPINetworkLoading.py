import pandas as pd
import torch
import os
import numpy
import gzip


proteins = {}
interactions1 = []
interactions2 = []
with gzip.open('/Users/marlon/Desktop/Project/9606.protein.links.v11.5.txt.gz', 'rt') as f:
    next(f) # Ignore the header
    for line in f:
        protein1, protein2, score = line.strip().split()
        score = int(score)
        if score >= 700: # Filter interactions with more confidence
            protein1 = protein1.split('.')[1]
            protein2 = protein2.split('.')[1]
            if protein1 not in proteins:
                proteins[protein1] = len(proteins) # Assign an ID to protein
            if protein2 not in proteins:
                proteins[protein2] = len(proteins)
#            interactions1.append(proteins[protein1])
#            interactions2.append(proteins[protein2])

#edge_index = [interactions1, interactions2]


# Save proteins and interactions for future use
#proteins_df = pd.DataFrame({'proteins': list(proteins.keys()), 'ids': proteins.values()})
#proteins_df.to_csv('/Users/marlon/Desktop/Project/proteins.csv')
#interactions_df = pd.DataFrame({'protein1': interactions1, 'protein2': interactions2})
#interactions_df.to_csv('/Users/marlon/Desktop/Project/interactions.csv')


# Read ensemble data
proteins_features = {}

# we are only interested in certain fields
fields = ['converted_alias', 'name']

data = pd.read_csv(
    os.path.join("/Users", "marlon", "Desktop", "Project", "gProfiler_hsapiens_09-01-2023_11-01-49.csv"), usecols=fields ,index_col=0)


# Drop None values
data = data.mask(data.eq('None')).dropna()
data = data.reset_index()


# Remove proteins for which we have no interaction information and

# remove features (and therefore proteins) that are not included in any of the cancer types views

features = pd.read_csv(
    os.path.join("/Users", "marlon", "Desktop", "Project", "TCGAData", "AllFeatures.csv"), index_col=0)

features = list(features.iloc[:,0].values)

# First, get feature names into the right structure
features_fix = []

for idx,feat in enumerate(features):

    index_1 = feat.find(':')
    index_2 = feat.find('|')


    if index_1 != -1:
        if index_2 != -1:
            features_fix.append(feat[index_1+1:index_2].upper())
        else:
            features_fix.append(feat[index_1+1:].upper())
    else:
        if index_2 != -1:
            features_fix.append(feat[0:index_2].upper())
        else:
            features_fix.append(feat.upper())

    features_fix[-1] = features_fix[-1].replace('-','')




# Start deleting ...

to_delete_ = []
for index, row in data.iterrows():

    curr_protein = row['converted_alias']
    curr_feature = row['name']
    # TODO : nearly all duplicates of features (mapped to multiple proteins) get deleted bc not in proteins (interactions)
    if str(curr_protein) not in proteins:
        to_delete_.append(index)
    if str(curr_feature) not in features_fix:
        if index not in to_delete_:
            to_delete_.append(index)


data = data.drop(index=to_delete_)




# Based on this data, we can create matrices

data_df = pd.DataFrame(data)

data_df.to_csv("/Users/marlon/Desktop/Project/ProteinToFeature.csv")











