import torch
import numpy as np



if __name__ == '__main__':
    a = [[4,3,2],[3,2,1]]


    for c,view in enumerate(a):
        temp = a[c].copy()
        temp.reverse()
        a[c] += temp[1:]
        for x in view:
            print(x)

