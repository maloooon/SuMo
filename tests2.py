import torch



if __name__ == '__main__':
    a = [['b','c'],['h']]

    for c,x in enumerate(a):
        for c2,z in enumerate(x):
            if z.lower() == 'b':
                a[c][c2] = 'dfu'


    print(a)


