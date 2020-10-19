import numpy as np


def dataset_Circles(n=1000, radius=0.7):
    X = np.zeros((2, n))
    Y = np.zeros((1, n))

    for currentN in range(n):
        i, j = 2*np.random.rand(2)-1
        r = np.sqrt( i**2 + j**2 )
        #r += np.random.rand()*0.05
        if (r < radius): 
            l=0 
        else:
            l = 1

        X[0, currentN] = i
        X[1, currentN] = j
        Y[0, currentN] = float(l)

    return X, Y
