import torch
import numpy as np
import os
import pandas as pd

if __name__ == '__main__':
    x = torch.randn(10,10)

    df = pd.DataFrame(x)

    print(df)
    if (df.iloc[:,0][0] == df.iloc[:,0]).all():
        print("fuego")

