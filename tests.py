import torch



if __name__ == '__main__':
    x = torch.tensor(0.425)

    y = torch.tensor(0.150)


    z = [x,y]

    a = torch.stack(tuple(z))




