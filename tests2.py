import torch



if __name__ == '__main__':
    a =torch.rand(2,10)
    b = torch.rand(2,1)
    c= torch.rand(2,1)
    print(torch.cat((a,b,c), dim=1))


    print(a)
    b = torch.rand(5)

    b = torch.unsqueeze(b, dim=1)
    print(b.shape)

   # f = torch.cat((a,b),dim=1)
   # print(f)