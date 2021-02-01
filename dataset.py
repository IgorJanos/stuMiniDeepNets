import numpy as np


def dataset_Circles(n=1000, radius=0.7, noise=0.0):
    X = np.zeros((2, n))
    Y = np.zeros((1, n))

    for currentN in range(n):
        i, j = 2*np.random.rand(2)-1
        
        r = np.sqrt( i**2 + j**2 )
        if (noise > 0.0):
            r += np.random.rand()*noise

        if (r < radius): 
            l = 0 
        else:
            l = 1

        X[0, currentN] = i
        X[1, currentN] = j
        Y[0, currentN] = float(l)

    return X, Y


def dataset_Flower(n=1000, noise=0.0):
    N = int(n/2) 
    D = 2
    X = np.zeros((n,D)) 
    Y = np.zeros((n,1), dtype='uint8') 
    a = 4 

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*noise
        r = a*np.sin(4*t) + np.random.randn(N)*noise
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T
    return X, Y