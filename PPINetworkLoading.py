import pandas as pd
import os
import gzip



# Read in Protein pairs with interactions
proteins = {}
interactions1 = []
interactions2 = []

# Replace /Users/marlon/Desktop with dir = os.path.expanduser('~/SUMO ..')

dir = os.path.expanduser('/Users/marlon/Desktop/Project/9606.protein.links.v11.5.txt.gz')
#dir = os.path.expanduser('~/SUMO/Project/9606.protein.links.v11.5.txt.gz')
with gzip.open(dir, 'rt') as f:
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

# Read ensemble data
proteins_features = {}

# we are only interested in certain fields
fields = ['converted_alias', 'name']

# Replace with "~", "SUMO"
data = pd.read_csv(
    os.path.join("/Users", "marlon", "Desktop", "Project", "gProfiler_hsapiens_09-01-2023_11-01-49.csv"),
    usecols=fields ,index_col=0)


# Drop None values
data = data.mask(data.eq('None')).dropna()
data = data.reset_index()



# Replace with "~", "SUMO"
# Read in all features across all cancer types
features = pd.read_csv(
    os.path.join("/Users", "marlon", "Desktop", "Project", "TCGAData", "AllFeatures.csv"), index_col=0)

features = list(features.iloc[:,0].values)

# Get feature names into the right structure
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



# Remove proteins for which we have no interaction information and
# remove features (and therefore proteins) that are not included in any of the cancer types views
# Start deleting ...

to_delete_ = []
for index, row in data.iterrows():

    curr_protein = row['converted_alias']
    curr_feature = row['name']
    # no interaction proteins...
    if str(curr_protein) not in proteins:
        to_delete_.append(index)
    # feature not included in any cancer type ...
    if str(curr_feature) not in features_fix:
        if index not in to_delete_:
            to_delete_.append(index)


data = data.drop(index=to_delete_)



# Based on this data, we can create matrices
data_df = pd.DataFrame(data)


dir = os.path.expanduser("/Users/marlon/Desktop/Project/ProteinToFeature.csv")
#dir = os.path.expanduser("~/SUMO/Project/ProteinToFeature.csv")
data_df.to_csv(dir)











