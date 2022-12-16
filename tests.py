import torch
import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.DataFrame({'bool': [1, 0, 1, None],
                       'floats': [1.2, 3.1, 4.4, 5.5],
                       'ints': [1, 2, 3, 4],
                       'str': ['a', 'b', 'c', 'd']})

    bool_cols = [col for col in df
                 if np.isin(df[col].dropna().unique(), [0, 1]).all()]
