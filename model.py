import numpy as np
import matplotlib.pyplot as plt

from loss import CreateLossFunction
from optimizer import CreateOptimizer

#------------------------------------------------------------------------------
#   Model class
#------------------------------------------------------------------------------
class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.batch_size = 0

    def addLayer(self,  layer):
        self.layers.append(layer)

    def initialize(self, params, lossName, optName):

        # Este potahame loss
        self.loss = CreateLossFunction(lossName)

        # Nainicializujeme optimizer
        self.optimizer = CreateOptimizer(optName)
        self.optimizer.initialize(params)

        # Parametre
        self.params = params
        self.batch_size = params["batch_size"]

        # Zavolame inicializaciu postupne na vsetkych vrstvach
        prevLayer = None
        for l in self.layers:
            l.initialize(prevLayer)
            prevLayer = l                

    def train(self, X, Y, epochs, devX, devY):

        # Parsneme si rozmery datasetu
        nX, m = X.shape
        nY, mY = Y.shape

        doValidation = True
        if (devX is None or devY is None):
            doValidation = False

        # Bud ideme po batchoch, alebo po celom datasete
        batchSize = self.batch_size
        if (batchSize <= 0): batchSize = m
        steps = int(np.ceil(m / batchSize))

        # Zbierame vysledky
        data = {
            "epochs": [],
            "loss": [],
            "val_loss": []
        }

        # Ideme prechadzat cez epochy a cez batch kroky
        loss = 0.0
        for _ep in range(epochs):

            # Akumulator pre training loss
            _trainLoss = 0.0

            for _step in range(steps):
                mStart = _step * batchSize
                mEnd = min(mStart + batchSize, m)

                # Vyberame data pre aktualny batch
                batch_X = X[:,mStart:mEnd]
                batch_Y = Y[:,mStart:mEnd]

                # Ucime jeden krok
                loss = self._train_SingleStep(batch_X, batch_Y)

                # Zapocitame prirastok training lossu
                _batchM = batch_Y.shape[1]
                _trainLoss += loss*_batchM

            # Pridame data novej epochy
            data["epochs"].append(_ep)
            data["loss"].append(_trainLoss / m)

            # Robime validaciu ?
            if (doValidation):
                _A = self.predict(devX)
                l, _ = self._compute_loss(_A, devY)
                data["val_loss"].append(l)


            if (_ep%1000 == 0):
                if (doValidation):
                    print('Epoch {0}:    Loss = {1:.5f}   Val_Loss: {2:.5f}'.format(_ep, data["loss"][_ep], data["val_loss"][_ep]))
                else:
                    print('Epoch {0}:    Loss = {1:.5f}'.format(_ep, data["loss"][_ep]))

        return data


    def predict(self, X):
        # Jednoduchy priamy prechod dopredu
        a = X
        for l in self.layers:
            a, _ = l.forward(a, False)

        # Len 2 triedy nas zaujimaju
        a = (a > 0.5)
        return a  

    def _train_SingleStep(self,X,Y):

        #----------------------------------------------------------------------
        # 1. Forward pass
        a = X
        forwardCache = []
        for l in self.layers:
            aNext, layerCache = l.forward(a, True)
            
            # Zozbierame data potrebne pre backward pass
            forwardCache.append((l, layerCache, a))

            # Pokracujeme
            a = aNext

        #----------------------------------------------------------------------
        # 2. Compute loss
        l, da = self._compute_loss(a, Y)

        #----------------------------------------------------------------------
        # 3. Backward pass
        for layer, layerCache, aPrev in reversed(forwardCache):
            # Zbehneme backward vypocet
            daPrev = layer.backward(self.optimizer, da, aPrev, layerCache)
            da = daPrev

        #----------------------------------------------------------------------
        # 4. Update parameters pass
        for layer, _, _ in reversed(forwardCache):
            layer.update(self.optimizer)

        # Vraciame loss
        return l


    def _compute_loss(self, A, Y):
        m = Y.shape[1]
        l = self.loss(A, Y)
        dl = 1.0/m * self.loss.derivate(A, Y)        
        l = 1.0/m * np.nansum(l)
        return l, dl



def PlotModel(model, data, X, Y):
    pad = 0.5

    x_min, x_max = X[0,:].min()-pad, X[0,:].max()+pad
    y_min, y_max = X[1,:].min()-pad, X[1,:].max()+pad
    h = 0.01

    # spravime grid
    xx, yy = np.meshgrid(
                    np.arange(x_min, x_max, h), 
                    np.arange(y_min, y_max, h)
                    )

    # spocitame model pre cely grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)

    # nakreslime kontury a training examples
    plt.figure(figsize=(20,9))
    plt.subplot(121)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.yscale('linear')
    plt.scatter(X[0,:], X[1,:], c=Y, cmap=plt.cm.RdBu)

    plt.subplot(122)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.plot(data["epochs"], data["loss"], 'b', label='train')
    plt.plot(data["epochs"], data["val_loss"], 'r', label='val')
    plt.legend()

    plt.show()
    plt.close()
