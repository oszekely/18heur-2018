import numpy as np
import math


def is_integer(a):
    """
    Tests if `a` is integer
    """
    dt = a.dtype
    return dt == np.int16 or dt == np.int32 or dt == np.int64


class Correction:

    """
    Baseline mutation correction strategy - "sticks" the solution to domain boundaries
    """

    def __init__(self, of):
        self.of = of

    def correct(self, x):
        return np.minimum(np.maximum(x, self.of.a), self.of.b)


class MirrorCorrection(Correction):
    """
    Mutation correction via mirroring
    """

    def __init__(self, of):
        Correction.__init__(self, of)

    def correct(self, x):
        n = np.size(x)
        d = self.of.b - self.of.a
        for k in range(n):
            if d[k] == 0:
                x[k] = self.of.a[k]
            else:
                de = np.mod(x[k] - self.of.a[k], 2*d[k])
                de = np.amin([de, 2*d[k] - de])
                x[k] = self.of.a[k] + de
        return x


class ExtensionCorrection(Correction):
    """
    Mutation correction via periodic domain extension
    """

    def __init__(self, of):
        Correction.__init__(self, of)

    def correct(self, x):
        d = self.of.b - self.of.a
        x = self.of.a + np.mod(x - self.of.a, d + (1 if is_integer(x) else 0))
        return x


class Mutation:

    """
    Generic mutation super-class
    """

    def __init__(self, correction):
        self.correction = correction


class CauchyMutation(Mutation):

    """
    Cauchy mutation
    """

    def __init__(self, r, correction):
        Mutation.__init__(self, correction)
        self.r = r

    def mutate(self, x):
        n = np.size(x)
        u = np.random.uniform(low=0.0, high=1.0, size=n)
        r = self.r
        x_new = x + r * np.tan(np.pi * (u - 1 / 2))
        if is_integer(x):
            x_new = np.array(np.round(x_new), dtype=int)  # optional rounding
        x_new_corrected = self.correction.correct(x_new)
        return x_new_corrected

class SigmoidFunction:

    def getValue(self, x):
        return 1.0/(1.0 + math.exp(-x))

    def getDerivative(self, input):
        return np.multiply(input, np.subtract(np.ones(input.shape), input))

class MSRLoss:

    def getLossDerivative(self, predicted, exact):
        auxMultiplication = np.multiply(np.subtract(predicted, exact), predicted)
        auxSubtraction = np.subtract(np.ones(exact.shape), predicted)
        return np.multiply(auxMultiplication, auxSubtraction)


    def getLoss(self, predictionsMatrix, exactMatrix, weightsTensor, lambdaPar, recordsNum):
        diff = np.subtract(predictionsMatrix, exactMatrix)
        cost = 0
        for differenceOutput in diff:
            cost += np.dot(differenceOutput, differenceOutput)
        for layer in weightsTensor:
            for neuronWeights in layer:
                cost += lambdaPar * np.dot(neuronWeights, neuronWeights)
        return cost * 1.0/(2*recordsNum)

    def getWeightUpdate(self, weight, auxDeltaWeight, learningRate, lambdaPar, batchSamplesNum):
        regularizedWeight = (1- learningRate * lambdaPar / batchSamplesNum) * weight
        return np.subtract(regularizedWeight, learningRate / batchSamplesNum * auxDeltaWeight)

    def getBiasUpdate(self, bias, auxDeltaBias, learningRate, lambdaPar, batchSamplesNum):
        return bias - learningRate / batchSamplesNum * auxDeltaBias

class CrossEntropyLoss:

    def getLossDerivative(self, predicted, exact):
        return np.subtract(predicted, exact)

    def getLoss(self, predictionsMatrix, exactMatrix, weightsTensor, lambdaPar, recordsNum):
        cost = 0
        for ind in range(predictionsMatrix.shape[0]):
            cost -= np.matmul(exactMatrix[ind,:], np.log(np.transpose(predictionsMatrix[ind,:])))
            cost -= np.matmul(1-exactMatrix[ind,:], np.transpose(1-predictionsMatrix[ind,:]))

        for layer in weightsTensor:
            for neuronWeights in layer:
                cost += lambdaPar / 2.0 * np.dot(neuronWeights, neuronWeights)
        return 1/recordsNum * cost

    def getWeightUpdate(self, weight, auxDeltaWeight, learningRate, lambdaPar, batchSamplesNum):
        regularizedWeight = (1 - learningRate * lambdaPar / batchSamplesNum) * weight
        return np.subtract(regularizedWeight, learningRate / batchSamplesNum * auxDeltaWeight)

    def getBiasUpdate(self, bias, auxDeltaBias, learningRate, lambdaPar, batchSamplesNum):
        return bias - learningRate / batchSamplesNum * auxDeltaBias