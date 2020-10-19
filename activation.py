import numpy as np


#------------------------------------------------------------------------------
#   ActivationFunction class
#------------------------------------------------------------------------------
class ActivationFunction:
    def __init__(self):
        pass

    def __call__(self, Z):
        pass

    def derivate(self, A):
        pass


#------------------------------------------------------------------------------
#   LinearActivationFunction class
#------------------------------------------------------------------------------
class LinearActivationFunction(ActivationFunction):
    def __call__(self, Z):
        return Z

    def derivate(self, A):
        return np.ones(A.shape)

#------------------------------------------------------------------------------
#   RELUActivationFunction class
#------------------------------------------------------------------------------
class RELUActivationFunction(ActivationFunction):
    def __call__(self, Z):
        return np.maximum(Z, 0)

    def derivate(self, A):
        return (A > 0)


#------------------------------------------------------------------------------
#   SigmoidActivationFunction class
#------------------------------------------------------------------------------
class SigmoidActivationFunction(ActivationFunction):
    def __call__(self, Z):
        return 1.0/(1.0+np.exp(-Z))

    def derivate(self, A):
        return np.multiply(A, 1-A)


MAP_ACTIVATION_FUCTIONS = {
    "linear": LinearActivationFunction,
    "relu": RELUActivationFunction,
    "sigmoid": SigmoidActivationFunction
}

def CreateActivationFunction(kind):
    if (kind in MAP_ACTIVATION_FUCTIONS):
        return MAP_ACTIVATION_FUCTIONS[kind]()
    raise ValueError(kind, "Unknown activation function {0}".format(kind))


