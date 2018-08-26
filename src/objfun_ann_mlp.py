from objfun import ObjFun
import heur_aux
import numpy as np
import math
import matplotlib.pyplot as plt

class ANNMLPClassifier(ObjFun):

    def __init__(self, par_numOfNeuronsInHiddenLayers, par_epochsNum,
                 par_batchSize, par_learningRate, par_lambda, par_lossFunction, par_activationFunction,
                 par_data, par_labels, par_dataSplit):
        """
        Initialization
        :param par_numOfNeuronsInHiddenLayers: number of neurons in hidden layers (vector of ints)
        :param par_epochsNum: max num of epochs (int)
        :param par_batchSize: number samples per bin (int)
        :param par_learningRate: learning rate (float)
        :param par_lambda: rate coeficient for regularisation (float)
        :param par_lossFunction: Loss function type (string)
        :param par_activationFunction: Activation function type (string)
        :param par_data: data used for training and testing ANN MLP (matrix, dim: records x features, elementsType: np.float)
        :param par_labels: labels for data (array like, elementsType: int, start labels from 0 with class-wise increased by 1 i.e.:[0, 0, 1, 1, 2, 2])
        :param par_dataSplit: split percentage for each strata (float)

        """

        #PREPARE DATASETS
        featuresCount, classesList, trainingLabelsVectors, trainingLabels, \
        trainingData, testingLabelsVectors, testingLabels, testingData = self.prepareDatasets(par_data, par_labels, par_dataSplit)
        self.classes = classesList
        self.trainingLabels = trainingLabels
        self.trainingLabelsVectors = trainingLabelsVectors
        self.trainingData = trainingData
        self.testingLabels = testingLabels
        self.testingLabelsVectors = testingLabelsVectors
        self.testingData = testingData
        self.trainingDataSize = trainingLabels.shape[0]
        self.testingDataSize = testingLabels.shape[0]
        self.featuresCount = featuresCount

        #DEFINE ANN WEIGHTS
        self.neuronsNums = np.concatenate([par_numOfNeuronsInHiddenLayers, [len(classesList)]]).astype(np.int)
        self.weightsTensor = [np.zeros((self.neuronsNums[0], featuresCount + 1), dtype=np.float)]
        dim = self.neuronsNums[0] * (featuresCount + 1)
        for layerInd in range(1, len(self.neuronsNums)):
            layer = np.zeros((self.neuronsNums[layerInd], self.neuronsNums[layerInd-1]+1), dtype=np.float)
            self.weightsTensor.append(layer)
            dim += self.neuronsNums[layerInd] * (self.neuronsNums[layerInd-1]+1)
        self.weightsDim = dim

        #INITIALIZE ANN WEIGHTS
        for layerInd in range(0, len(self.weightsTensor)):
            layerShape = self.weightsTensor[layerInd].shape
            self.weightsTensor[layerInd][:,0] = self.initializeBias(layerShape[0], layerShape[1]-1)
            self.weightsTensor[layerInd][:, 1:] = self.initializeWeights(layerShape[0], layerShape[1]-1)

        #TRAIN NETWORK
        self.batchSize = par_batchSize
        self.lossFunction = self.resolveLossFunc(par_lossFunction)
        self.activationFunction = self.resolveActivationFunc(par_activationFunction)
        self.maxNumberOfEpochs = par_epochsNum
        self.learningRate = par_learningRate
        self.lambdaPar = par_lambda
        self.trainingStatusInfo = []
        self.trainNetwork()
        self.neighborhoodStep = 0.05

    def trainNetwork(self):
        activationVectorizerFunction = np.vectorize(self.activationFunction.getValue, otypes=[np.float])
        trainingSetSize = len(self.trainingLabels)
        batchesNum = math.ceil(trainingSetSize / float(self.batchSize))
        epoch = 1

        #FIGURES STATISTICS
        epochArray = []
        lossArray = []
        precisionArray = []
        precisionTestingArray = []
        lossTestingArray = []

        while epoch <= self.maxNumberOfEpochs:
            for binNum in range(1, batchesNum+1):
                deltas = self.initializeAuxDeltas()

                lowerBound = (binNum-1)*self.batchSize
                upperBound = min(binNum*self.batchSize, trainingSetSize)
                batchData = self.trainingData[lowerBound:upperBound,:]
                batchLabels = self.trainingLabelsVectors[lowerBound:upperBound]
                samplesNum = upperBound - lowerBound

                #FEED FORWARD PART
                inputs = [np.transpose(batchData)] # inputs to each layer
                outputs = [np.transpose(batchData)]
                for hiddenLayer in self.weightsTensor:
                    inputAux = np.add(np.matmul(hiddenLayer[:,1:], outputs[-1]), np.transpose(np.tile(hiddenLayer[:,0], (samplesNum, 1))))
                    inputs.append(inputAux)
                    outputAux = activationVectorizerFunction(inputAux)
                    outputs.append(outputAux)

                #GET LOSS DERIVATIVE VALUES\
                lossDerivative = self.lossFunction.getLossDerivative(np.transpose(outputs[-1]), batchLabels)
                errors = [None] * len(self.weightsTensor)
                errors[-1] = np.transpose(lossDerivative)

                #BACKPROPAGATE THE ERROR
                for layerInd in reversed(range(0, len(self.weightsTensor)-1)):
                    feedForwardOut = outputs[layerInd+1]
                    propagatedErrMult1 = np.matmul(np.transpose(self.weightsTensor[layerInd + 1])[1:,:], errors[layerInd + 1])
                    activationFunctionDerivative = self.activationFunction.getDerivative(feedForwardOut)
                    propagatedErr = np.multiply(propagatedErrMult1, activationFunctionDerivative)
                    errors[layerInd] = propagatedErr

                #COMPUTE DESCENT AUX SUM
                deltas[0][:, 0] = np.sum(errors[0], axis=1)
                deltas[0][:, 1:] = np.matmul(errors[0], np.transpose(outputs[0]))
                for layerInd in range(1, len(self.weightsTensor)):
                    deltas[layerInd][:,0] = np.sum(errors[layerInd], axis=1)
                    deltas[layerInd][:, 1:] = np.matmul(errors[layerInd], np.transpose(outputs[layerInd]))

                #COMPUTE GRADIENT DESCENT
                for layerInd in range(0, len(self.weightsTensor)):
                    self.weightsTensor[layerInd][:,0] = self.lossFunction.getBiasUpdate(self.weightsTensor[layerInd][:, 0], deltas[layerInd][:, 0],
                                                        self.learningRate, self.lambdaPar, samplesNum)
                    self.weightsTensor[layerInd][:, 1:] = \
                        self.lossFunction.getWeightUpdate(self.weightsTensor[layerInd][:, 1:], deltas[layerInd][:, 1:],
                                                        self.learningRate, self.lambdaPar, samplesNum)

            #COMPUTE STATISTICS
            input = self.trainingData

            for layer in self.weightsTensor:
                input = activationVectorizerFunction(np.add(np.matmul(input, np.transpose(layer[:,1:])), np.transpose(layer[:,0])))
            predictions = input

            loss = self.lossFunction.getLoss(predictions, self.trainingLabelsVectors, self.weightsTensor, self.lambdaPar, self.trainingDataSize)
            predictionsLabels = np.argmax(predictions, axis=1)
            precision = np.sum(predictionsLabels == self.trainingLabels, dtype=float) / self.trainingDataSize

            input = self.testingData

            for layer in self.weightsTensor:
                input = activationVectorizerFunction(
                    np.add(np.matmul(input, np.transpose(layer[:, 1:])), np.transpose(layer[:, 0])))
            predictions = input

            lossTesting = self.lossFunction.getLoss(predictions, self.testingLabelsVectors, self.weightsTensor,
                                             self.lambdaPar, self.trainingDataSize)
            predictionsLabels = np.argmax(predictions, axis=1)
            precisionTesting = np.sum(predictionsLabels == self.testingLabels, dtype=float) / self.testingDataSize

            self.trainingStatusInfo.append({"precision_training": precision, "loss_training": loss,
                                            "precision_testing": precisionTesting, "loss_testing": lossTesting,
                                            "epoch": epoch})
            epochArray.append(epoch)
            lossArray.append(loss)
            precisionArray.append(precision)
            lossTestingArray.append(lossTesting)
            precisionTestingArray.append(precisionTesting)

            epoch += 1

        plt.rcParams['figure.figsize'] = [10, 8]

        plt.subplot(2,2,1)
        plt.plot(epochArray, lossArray)
        plt.title("MLP Training Statistics")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Training Data")

        plt.subplot(2,2, 2)
        plt.plot(epochArray, precisionArray)
        plt.xlabel("Epoch")
        plt.ylabel("Training Data Precision")

        plt.subplot(2, 2, 3)
        plt.plot(epochArray, lossTestingArray)
        plt.xlabel("Epoch")
        plt.ylabel("Loss Testing Data")

        plt.subplot(2, 2, 4)
        plt.plot(epochArray, precisionTestingArray)
        plt.xlabel("Epoch")
        plt.ylabel("Testing Data Precision")

        weightsFlatten = np.array([0])
        for layer in self.weightsTensor:
            weightsFlatten = np.append(weightsFlatten, layer.flatten(), axis=0)
        minVal = math.floor(np.min(weightsFlatten))
        maxVal = math.ceil(np.max(weightsFlatten))

        self.fstar = lossArray[-1]
        self.a = minVal
        self.b = maxVal

    def initializeAuxDeltas(self):
        auxDeltasTensor = []
        for layerWeights in self.weightsTensor:
            auxDeltasTensor.append(np.zeros(layerWeights.shape))
        return auxDeltasTensor

    def resolveLossFunc(self, lossfuncPar):
        if(lossfuncPar == "MSR"):
            return heur_aux.MSRLoss()
        if (lossfuncPar == "cross-entropy"):
            return heur_aux.CrossEntropyLoss()
        assert True, "Unresolved loss function: " + lossfuncPar

    def resolveActivationFunc(self, activationfuncPar):
        if(activationfuncPar == "sigmoid"):
            return heur_aux.SigmoidFunction()
        assert True, "Unresolved activation function: " + activationfuncPar

    def initializeBias(self, numberNeuronsInLayer, numberNeuronsInPreviousLayer):
        randomBias = np.random.rand(numberNeuronsInLayer)
        #randomBias = np.ones(numberNeuronsInLayer)
        return randomBias

    def initializeWeights(self, numberNeuronsInLayer, numberNeuronsInPreviousLayer):
        randomWeights = np.random.normal(0, 1/math.sqrt(numberNeuronsInPreviousLayer),(numberNeuronsInLayer, numberNeuronsInPreviousLayer))
        #randomWeights = np.ones((numberNeuronsInLayer, numberNeuronsInPreviousLayer))
        return randomWeights

    def prepareDatasets(self, data, labels, strataSplit):
        classesList = np.unique(labels)
        featuresCount = np.shape(data)[1]

        trainingIndices = np.array([], dtype=np.int)
        testingIndices = np.array([], dtype=np.int)

        for classInd in classesList:
            classIndices = np.argwhere(labels == classInd)
            classDataNum = np.shape(classIndices)[0]
            splitNum = math.ceil(strataSplit*classDataNum)

            splits = np.split(classIndices, [splitNum])
            trainingIndices = np.concatenate([trainingIndices, splits[0].reshape(-1)])
            testingIndices = np.concatenate([testingIndices, splits[1].reshape(-1)])

        np.random.shuffle(trainingIndices)
        np.random.shuffle(testingIndices)
        trainingLabelsIndices = labels[trainingIndices]
        trainingLabels = np.zeros((len(trainingLabelsIndices), len(classesList)), dtype=np.int)
        indices = np.ones(len(trainingLabelsIndices), dtype=np.int)
        indices[0] = 0
        indices = np.cumsum(indices)
        trainingLabels[indices, trainingLabelsIndices] = 1
        trainingData = data[trainingIndices]
        testingLabelsIndices = labels[testingIndices]
        testingLabels = np.zeros((len(testingLabelsIndices), len(classesList)), dtype=np.int)
        indices = np.ones(len(testingLabelsIndices), dtype=np.int)
        indices[0] = 0
        indices = np.cumsum(indices)
        testingLabels[indices, testingLabelsIndices] = 1
        testingData = data[testingIndices]

        return featuresCount, classesList, trainingLabels, labels[trainingIndices], trainingData, \
               testingLabels, labels[testingIndices], testingData

    def generate_point(self):
        return np.random.uniform(self.a, self.b, size=self.weightsDim)

    def get_neighborhood(self, x, d):
        nd = []
        for i, xi in enumerate(x):
            # x-lower
            if x[i] > self.a:  # (!) mutation correction .. will be discussed later
                xl = x.copy()
                xl[i] = x[i] - self.neighborhoodStep
                nd.append(xl)

            # x-upper
            if x[i] < self.b:  # (!) mutation correction ..  -- // --
                xu = x.copy()
                xu[i] = x[i] + self.neighborhoodStep
                nd.append(xu)

        return nd

    def evaluate(self, x):
        activationVectorizerFunction = np.vectorize(self.activationFunction.getValue, otypes=[np.float])
        tensor = []
        neuronsNum = self.neuronsNums[0] * (self.featuresCount + 1)
        auxInd = neuronsNum
        tensor.append(np.reshape(x[0:neuronsNum], newshape=(self.neuronsNums[0], self.featuresCount + 1)))
        for layerInd in range(1, len(self.neuronsNums)):
            neuronsNum = self.neuronsNums[layerInd] * (self.neuronsNums[layerInd - 1] + 1)
            tensor.append(np.reshape(x[auxInd:auxInd + neuronsNum], newshape=( self.neuronsNums[layerInd], self.neuronsNums[layerInd - 1] + 1)))
            auxInd += neuronsNum

        input = self.trainingData

        for layer in tensor:
            input = activationVectorizerFunction(
                np.add(np.matmul(input, np.transpose(layer[:, 1:])), np.transpose(layer[:, 0])))
        predictions = input

        loss = self.lossFunction.getLoss(predictions, self.trainingLabelsVectors, tensor, self.lambdaPar,
                                         self.trainingDataSize)
        return loss

    def getTrainingDataPrecision(self, weights):
        activationVectorizerFunction = np.vectorize(self.activationFunction.getValue, otypes=[np.float])

        tensor = []
        neuronsNum = self.neuronsNums[0] * (self.featuresCount + 1)
        auxInd = neuronsNum
        tensor.append(np.reshape(weights[0:neuronsNum], newshape=(self.neuronsNums[0], self.featuresCount + 1)))
        for layerInd in range(1, len(self.neuronsNums)):
            neuronsNum = self.neuronsNums[layerInd] * (self.neuronsNums[layerInd - 1] + 1)
            tensor.append(np.reshape(weights[auxInd:auxInd + neuronsNum],
                                     newshape=(self.neuronsNums[layerInd], self.neuronsNums[layerInd - 1] + 1)))
            auxInd += neuronsNum

        input = self.trainingData
        for layer in tensor:
            input = activationVectorizerFunction(
                np.add(np.matmul(input, np.transpose(layer[:, 1:])), np.transpose(layer[:, 0])))
        predictions = input
        predictionsLabels = np.argmax(predictions, axis=1)
        precision = np.sum(predictionsLabels == self.trainingLabels, dtype=float) / self.trainingDataSize
        return precision

    def getTestingDataPrecision(self, weights):
        activationVectorizerFunction = np.vectorize(self.activationFunction.getValue, otypes=[np.float])

        tensor = []
        neuronsNum = self.neuronsNums[0] * (self.featuresCount + 1)
        auxInd = neuronsNum
        tensor.append(np.reshape(weights[0:neuronsNum], newshape=(self.neuronsNums[0], self.featuresCount + 1)))
        for layerInd in range(1, len(self.neuronsNums)):
            neuronsNum = self.neuronsNums[layerInd] * (self.neuronsNums[layerInd - 1] + 1)
            tensor.append(np.reshape(weights[auxInd:auxInd + neuronsNum],
                                     newshape=(self.neuronsNums[layerInd], self.neuronsNums[layerInd - 1] + 1)))
            auxInd += neuronsNum

        input = self.testingData
        for layer in tensor:
            input = activationVectorizerFunction(
                np.add(np.matmul(input, np.transpose(layer[:, 1:])), np.transpose(layer[:, 0])))
        predictions = input
        predictionsLabels = np.argmax(predictions, axis=1)
        precision = np.sum(predictionsLabels == self.testingLabels, dtype=float) / self.testingDataSize
        return precision