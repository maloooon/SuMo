import torch
import numpy as np
import pickle


if __name__ == '__main__':

    x  = torch.randn(4)
    print(x.shape)
    x = x.unsqueeze(1)
    print(x.shape)


