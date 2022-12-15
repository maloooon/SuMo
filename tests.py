import torch

if __name__ == '__main__':
    a = torch.rand(4,1)
    b = torch.rand(4)
    a = a.reshape((4))
    print(b)
    print(a.shape)
    print(b.shape)