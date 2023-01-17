import torch



if __name__ == '__main__':
    x = torch.randn(4,2)
    b = torch.randn(4,2)

    c = [x,b]

    print(torch.cat(tuple(c), dim=0))