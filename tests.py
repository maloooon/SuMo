import torch
import numpy as np
import os
import pandas as pd

if __name__ == '__main__':
    x = torch.randn(489,6000)

    for a in x:
        print(a)
        print(a[0:2])
        break
