import torch
import numpy as np
import os
import pandas as pd


def flatten(l):
    return [item for sublist in l for item in sublist]

if __name__ == '__main__':
    x = torch.rand(10,5)
    x[3,4] = 10
    print(x[3,4].item())

    print(x)