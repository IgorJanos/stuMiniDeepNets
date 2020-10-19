import numpy as np

from activation import CreateActivationFunction

#------------------------------------------------------------------------------
#   Layer class
#------------------------------------------------------------------------------
class Layer:
    def __init__(self, act="linear"):
        self.shape = (0, 0)
        self.activation = CreateActivationFunction(act)
        pass

    def initialize(self, prevLayer):
        pass

    def forward(self, x, isTraining):
        pass

    def backward(self, optimizer, da, aprev, cache):
        #   da      =   dL/da of current layer
        #   aprev   =   activation of previous layer needed for weights
        #   cache   =   cached data from forward pass for current layer
        pass

    def update(self, optimizer):
        pass

#------------------------------------------------------------------------------
#   InputLayer class
#------------------------------------------------------------------------------
class InputLayer(Layer):
    def __init__(self, nX):
        super().__init__(act="linear")

        # Vstupny tvar
        self.nX = nX

    def initialize(self, prevLayer):
        self.shape = (self.nX, 1)

    def forward(self, x, isTraining):
        return x, None

    def backward(self, optimizer, da, aprev, cache):
        return None, None

    def update(self, optimizer):
        pass


#------------------------------------------------------------------------------
#   ActivationLayer class
#------------------------------------------------------------------------------
class ActivationLayer(Layer):
    def __init__(self, act="linear"):
        super().__init__(act)

    def initialize(self, prevLayer):
        self.shape = prevLayer.shape

    def forward(self, x, isTraining):
        a = self.activation(x)     # a = activation(z)
        return a, a

    def backward(self, optimizer, da, aprev, cache):
        # Rozbalime data, ktore sme si nacachovali
        dz = np.multiply(da, self.activation.derivate(cache))
        return dz

    def update(self, optimizer):
        pass


#------------------------------------------------------------------------------
#   BatchNormLayer class
#------------------------------------------------------------------------------
class BatchNormLayer(Layer):
    def __init__(self):
        super().__init__(act="linear")
        self.epsilon = 1e-7
        self.movingMean = 0.0
        self.movingVariance = 1.0
        self.momentum = 0.99

    def initialize(self, prevLayer):
        self.shape = prevLayer.shape
        self.optimizerData = None

        # Gamme and Beta
        self.gamma = np.ones(shape=(self.shape[0], 1))
        self.beta = np.zeros(shape=(self.shape[0], 1))
        

    def forward(self, x, isTraining):
        
        # Shape of input data
        n_x, m = x.shape

        # Compute Mean and Variance
        _mu = 1.0/m * np.sum(x, axis=1).reshape(n_x, 1)
        self.movingMean = self.momentum*self.movingMean + (1-self.momentum)*_mu
        if (not isTraining):
            _mu = self.movingMean

        _x = x - _mu

        _sigSquared = 1.0/m * np.sum(_x**2, axis=1).reshape(n_x, 1)
        self.movingVariance = self.momentum*self.movingVariance + (1-self.momentum)*_sigSquared
        if (not isTraining):
            _sigSquared = self.movingVariance

        _xNorm = _x / np.sqrt(_sigSquared + self.epsilon)

        # Compute zTilda
        _zTilda = np.multiply(self.gamma, _xNorm) + self.beta
        return _zTilda, (x, _xNorm, _mu, _sigSquared, self.gamma)

    def backward(self, optimizer, da, aprev, cache):
        # Rozbalime data, ktore sme si nacachovali
        x, _xNorm, _mu, _sigSquared, gamma = cache
        m = x.shape[1]

        # Spocitame gradienty na gamma, beta
        dxnorm = gamma * da        
        dGamma = np.sum(_xNorm*da, axis=1, keepdims=True)
        dBeta = np.sum(da, axis=1, keepdims=True)

        # A treba este spocitat gradient vstupnej aktivacie
        dSigSquared = np.sum(
            dxnorm*(x-_mu)*(-1.0/2.0)*np.power(_sigSquared+self.epsilon, -3.0/2.0),
            axis=1, keepdims=True
            )
        dMu = np.sum(
            dxnorm*(-1.0/np.sqrt(_sigSquared+self.epsilon)),
            axis=1, keepdims=True
        )
        dx = dxnorm*(1.0/np.sqrt(_sigSquared+self.epsilon)) + dSigSquared*((2.0/m)*(x-_mu)) + dMu*(1.0/m)

        # Optimizer sa moze pohrat s dGamma a dBeta
        self.optimizerData = optimizer.backward(self.optimizerData, dGamma, dBeta)
        return dx

    def update(self, optimizer):
        # Aktualizujeme cez optimizer
        self.gamma, self.beta = optimizer.update(self.optimizerData, self.gamma, self.beta)


#------------------------------------------------------------------------------
#   DenseLayer class
#------------------------------------------------------------------------------
class DenseLayer(Layer):
    def __init__(self, nUnits, act="linear"):
        super().__init__(act)

        # Pocet neuronov
        self.nUnits = nUnits
        self.W = None
        self.b = None
        self.optimizerData = None               # Persistant optimizer data

    def initialize(self, prevLayer):
        self.shape = (self.nUnits, 1)
        self.optimizerData = None

        # Inicializcia weights and bias
        pnx, _ = prevLayer.shape
        nx, _ = self.shape
        self.W = np.random.randn(nx, pnx)
        self.b = np.zeros((nx, 1), dtype=float)

    def forward(self, x, isTraining):
        z = np.matmul(self.W, x) + self.b           # z = W*x + b
        a = self.activation(z)                      # a = activation(z)

        # vraciame medzivypocty
        return a, (self.W, self.b, z, a)
        
    def backward(self, optimizer, da, aprev, cache):
        #   da      =   dL/da of current layer
        #   aprev   =   activation of previous layer needed for weights
        #   cache   =   cached data from forward pass for current layer

        # Rozbalime data, ktore sme si nacachovali
        W, b, z, a = cache

        # spocitame dz = da * activation.derivate
        dz = np.multiply(da, self.activation.derivate(a))
        dW = np.matmul(dz, aprev.T)
        db = np.sum(dz, axis=1, keepdims=True)

        # Vysledok - da pre predchadzajucu vrstvu a data pre update parameters
        daPrev = np.matmul(W.T, dz)        

        # Optimizer sa moze pohrat s dW a db
        self.optimizerData = optimizer.backward(self.optimizerData, dW, db)
        return daPrev

    def update(self, optimizer):
        # Aktualizujeme parametre cez optimizer
        self.W, self.b = optimizer.update(self.optimizerData, self.W, self.b)




