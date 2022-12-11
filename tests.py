import torch
import numpy as np
import os
import pandas as pd


def flatten(l):
    return [item for sublist in l for item in sublist]

def create_tensor(tensor_list):
    x = len(tensor_list)
    z, y = tensor_list[0].shape

    tensor_new = torch.zeros((x, z, y))
    for i, tensor in enumerate(tensor_list):
        tensor_new[i, :, :] = tensor

    return tensor_new



if __name__ == '__main__':
    print(torch.empty(4))
