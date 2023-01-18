import torch
import numpy as np
import pickle


if __name__ == '__main__':


    # SALMON
    objects = []
    with (open("/Users/marlon/Desktop/Project/datasets_5folds.pickle", "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break


    print(objects[0]['1']['train']['x'].shape)
    print(objects[0]['1']['test']['x'].shape)

    # Train/Test direkt gleiche Größe ?? Feature Selection auf allen Daten durchgeführt und dann erst gesplittet ?