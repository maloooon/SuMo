import torch


if __name__ == '__main__':
    b = torch.tensor([[0, 1], [2, 3]])
    print(torch.reshape(b, (2,-1)))

