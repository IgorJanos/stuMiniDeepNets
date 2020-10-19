import numpy as np


#------------------------------------------------------------------------------
#   Optimizer class
#------------------------------------------------------------------------------
class Optimizer:
    def __init__(self):
        pass

    def initialize(self, params):
        pass

    def backward(self, optdata, dW, db):
        return None

    def update(self, optdata, W, b):
        return W, b     # NULL - nic nenauci !


#------------------------------------------------------------------------------
#   GradientDescentOptimizer class
#------------------------------------------------------------------------------
class GradientDescentOptimizer(Optimizer):
    def __init__(self):
        self.learning_rate = 0.01

    def initialize(self, params):
        # toto nas zatial zaujima
        self.learning_rate = params["learning_rate"]

    def backward(self, optdata, dW, db):
        # Pracujeme len nad aktualnou verziou
        return (dW, db)

    def update(self, optdata, W, b):
        # Rozbalime co sme si poslali
        dW, db = optdata

        # Priamy gradient descent
        W = W - self.learning_rate*dW
        b = b - self.learning_rate*db
        return W, b


#------------------------------------------------------------------------------
#   MomentumOptimizer class
#------------------------------------------------------------------------------
class MomentumOptimizer(Optimizer):
    def __init__(self):
        self.learning_rate = 0.01
        self.beta = 0.9

    def initialize(self, params):
        # toto nas zatial zaujima
        self.learning_rate = params["learning_rate"]
        self.beta = params["beta"]

    def backward(self, optdata, dW, db):
        
        # Startujeme z dW, db - alebo z historickych dat
        if (optdata == None):
            vdW = dW
            vdb = db
        else:
            vdW, vdb = optdata

        # Spocitame novy running average
        vdW = self.beta*vdW + (1-self.beta)*dW
        vdb = self.beta*vdb + (1-self.beta)*db

        # Vraciame vdW, vdb
        return (vdW, vdb)

    def update(self, optdata, W, b):
        # Rozbalime co sme si poslali
        vdW, vdb = optdata

        # Momentum gradient descent
        W = W - self.learning_rate*vdW
        b = b - self.learning_rate*vdb
        return W, b

#------------------------------------------------------------------------------
#   RMSPropOptimizer class
#------------------------------------------------------------------------------
class RMSPropOptimizer(Optimizer):
    def __init__(self):
        self.learning_rate = 0.01
        self.beta = 0.9
        self.epsilon = 1e-8

    def initialize(self, params):
        # toto nas zatial zaujima
        self.learning_rate = params["learning_rate"]
        self.beta = params["beta"]

    def backward(self, optdata, dW, db):
        
        # Startujeme z dW, db - alebo z historickych dat
        if (optdata == None):
            sdW = dW**2
            sdb = db**2
        else:
            sdW, sdb, _, _ = optdata
            sdW = self.beta * sdW + (1-self.beta)*(dW**2)
            sdb = self.beta * sdb + (1-self.beta)*(db**2)

        # update term
        udW = np.divide(dW, np.sqrt(sdW)+self.epsilon)
        udb = np.divide(db, np.sqrt(sdb)+self.epsilon)

        # Vraciame kompletku
        return (sdW, sdb, udW, udb)

    def update(self, optdata, W, b):
        # Rozbalime co sme si poslali
        sdW, sdb, udW, udb = optdata

        # Momentum gradient descent
        W = W - self.learning_rate*udW
        b = b - self.learning_rate*udb
        return W, b


#------------------------------------------------------------------------------
#   AdamOptimizer class
#------------------------------------------------------------------------------
class AdamOptimizer(Optimizer):
    def __init__(self):
        self.learning_rate = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def initialize(self, params):
        # toto nas zatial zaujima
        self.learning_rate = params["learning_rate"]
        self.beta1 = params["beta1"]
        self.beta2 = params["beta2"]

    def backward(self, optdata, dW, db):
        # Restart?
        if (optdata == None):
            vdW = dW
            vdb = db
            sdW = dW**2
            sdb = db**2
        else:
            vdW, vdb, sdW, sdb, _, _ = optdata

            # nove momentum
            vdW = self.beta1 * vdW + (1-self.beta1)*dW
            vdb = self.beta1 * vdb + (1-self.beta1)*db

            # nove square momentum
            sdW = self.beta2 * sdW + (1-self.beta2)*(dW**2)
            sdb = self.beta2 * sdb + (1-self.beta2)*(db**2)

        # update term
        udW = np.divide(vdW, np.sqrt(sdW)+self.epsilon)
        udb = np.divide(vdb, np.sqrt(sdb)+self.epsilon)

        # Vraciame kompletku
        return (vdW, vdb, sdW, sdb, udW, udb)

    def update(self, optdata, W, b):
        # Rozbalime co sme si poslali
        _, _, _, _, udW, udb = optdata

        # Momentum gradient descent
        W = W - self.learning_rate*udW
        b = b - self.learning_rate*udb
        return W, b






MAP_OPTIMIZER_FUNCTIONS = {
    "gd": GradientDescentOptimizer,
    "momentum": MomentumOptimizer,
    "rmsprop": RMSPropOptimizer,
    "adam": AdamOptimizer
}

def CreateOptimizer(kind):
    if (kind in MAP_OPTIMIZER_FUNCTIONS):
        return MAP_OPTIMIZER_FUNCTIONS[kind]()
    raise ValueError(kind, "Unknown optimizer {0}".format(kind))