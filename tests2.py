import torch
import numpy as np


if __name__ == '__main__':
    a = torch.randn(361,6148,2)
    b = torch.randn(20,6148,2)


    x = torch.cat(tuple([a,b]),dim=0)


    f = torch.stack(tuple([x[0],x[1]]))

    a = np.empty(5)

    np.put(a,2,33)

    print(a)