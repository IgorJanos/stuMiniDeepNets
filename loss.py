import numpy as np

#------------------------------------------------------------------------------
#   LossFunction class
#------------------------------------------------------------------------------
class LossFunction:
    def __init__(self):
        pass

    def __call__(self, A, Y):
        pass

    def derivate(self, A, Y):
        pass


#------------------------------------------------------------------------------
#   BinaryCrossEntropyLossFunction class
#------------------------------------------------------------------------------
class BinaryCrossEntropyLossFunction(LossFunction):
    def __call__(self, A, Y):
        A = np.clip(A, 1e-7, 1.0-1e-7)
        return np.multiply(-np.log(A), Y) + np.multiply(-np.log(1-A), (1-Y))

    def derivate(self, A, Y):
        A = np.clip(A, 1e-7, 1.0-1e-7)
        return -np.divide(Y, A) + np.divide(1-Y, 1-A)



MAP_LOSS_FUNCTIONS = {
    "bce": BinaryCrossEntropyLossFunction
}

def CreateLossFunction(kind):
    if (kind in MAP_LOSS_FUNCTIONS):
        return MAP_LOSS_FUNCTIONS[kind]()
    raise ValueError(kind, "Unknown loss function {0}".format(kind))

