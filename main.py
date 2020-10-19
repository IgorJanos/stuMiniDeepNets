import numpy as np
import matplotlib.pyplot as plt

from dataset import dataset_Circles
from layer import DenseLayer, InputLayer, ActivationLayer, BatchNormLayer
from model import Model, PlotModel


#------------------------------------------------------------------------------
#   Entry Point - Ucenie modelu
#------------------------------------------------------------------------------

def trainModel(X, Y, devX, devY):
    m = Model()

    # Vstupne a vystupne rozmery
    n_x, _ = X.shape
    n_y, _ = Y.shape

    # Input -> Dense(3) -> Dense(1)
    m.addLayer(InputLayer(n_x))
    m.addLayer(DenseLayer(15, act="linear"))
    m.addLayer(BatchNormLayer())
    m.addLayer(ActivationLayer(act="relu"))
    m.addLayer(DenseLayer(8, act="linear"))
    m.addLayer(BatchNormLayer())
    m.addLayer(ActivationLayer(act="relu"))
    m.addLayer(DenseLayer(n_y, act="sigmoid"))
    
    # Prepare all layer internals
    params = {
        
        # Optimizer parameters
        "learning_rate": 0.01,
        "beta": 0.9,               # Momentum, RMSProp
        "beta1": 0.9,               # Adam
        "beta2": 0.999,

        # MiniBatch
        "batch_size": 0            # 0 = Gradient descent. No minibatches
    }
    m.initialize(params, lossName="bce", optName="adam")

    # Train the shit
    data = m.train(X, Y, 10000, devX, devY)  
    PlotModel(m, data, devX, devY)


def doShit():       
    np.random.seed(1)

    # Loadneme dataset a ucime
    train_X, train_Y = dataset_Circles(n=20000)
    dev_X, dev_Y = dataset_Circles(n=2500)

    # Chod sa ucit, vole!
    trainModel(train_X, train_Y, dev_X, dev_Y)   




if (__name__ == "__main__"):
    doShit()