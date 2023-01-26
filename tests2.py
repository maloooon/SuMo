
import numpy as np
testset = []
testset_feat = []
test_data= []
for c,feat in enumerate(testset_feat):
    if c < len(testset_feat) - 3: # train data views
        test_data.append(np.array((testset.iloc[:, testset_feat[c]: testset_feat[c + 1]]).values))
    elif c == len(testset_feat) - 3: # duration
        train_duration = (np.array((testset.iloc[:, testset_feat[c]: testset_feat[c + 1]]).values)).squeeze(axis=1)
    elif c == len(testset_feat) -2: # event
        train_event = (np.array((testset.iloc[:, testset_feat[c]: testset_feat[c + 1]]).values)).squeeze(axis=1)