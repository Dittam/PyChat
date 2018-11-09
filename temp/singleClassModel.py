import numpy as np
import trainingGraph


class Rnn():

    def __init__(self, L1Neurons, L2Neurons, vocabSize):
        self.L1Neurons = L1Neurons
        self.L2Neurons = L2Neurons
        self.vocabSize = vocabSize

        # ---Model Definition---
        self.weightL1 = np.random.randn(L1Neurons, vocabSize) * 0.01
        self.weightL2 = np.random.randn(L2Neurons, L1Neurons) * 0.1
        self.weightRecur = np.random.randn(L2Neurons, L2Neurons) * 0.01
        self.weightOut = np.random.randn(vocabSize, L2Neurons) * 0.01
        self.biasL1 = np.zeros((L2Neurons, 1))
        self.biasHid = np.zeros((L2Neurons, 1))
        self.biasOut = np.zeros((vocabSize, 1))

        # Memory vars to keep track of states at each timestep
        self.charVec, self.hState, self.yState, self.pState, self.L1State = {}, {}, {}, {}, {}

    def getTotalWeights(self):
        return (self.L1Neurons * self.vocabSize) + (self.L2Neurons * self.L1Neurons) +\
            (self.L2Neurons * self.L2Neurons) + (self.vocabSize * self.L2Neurons) + \
            self.L1Neurons + self.L2Neurons + self.vocabSize

    def forwardPropagate(self, inputs, t):

        self.charVec[t] = np.zeros((self.vocabSize, 1))
        self.charVec[t][inputs[t]] = 1

        # get matrix state formed by passing input through L1
        self.L1State[t] = np.dot(self.weightL1, self.charVec[t]) + self.biasL1

        # get matrix state formed by passing L1State throught L2
        temp = np.dot(self.weightL2, self.L1State[t])
        self.hState[t] = np.tanh(
            temp + np.dot(self.weightRecur, self.hState[t - 1]) + self.biasHid)

        # get raw output vector from nerual net
        self.yState[t] = np.dot(
            self.weightOut, self.hState[t]) + self.biasOut

        # squash values into a probability vector
        self.pState[t] = np.exp(self.yState[t]) / \
            np.sum(np.exp(self.yState[t]))

    def lossFun(self, inputs, targets, prevHState):

        self.hState[-1] = np.copy(prevHState)

        loss = 0
        for t in range(len(inputs)):
            self.forwardPropagate(inputs, t)
            loss += -np.log(self.pState[t][targets[t], 0])

        return loss, self.hState[len(inputs) - 1]

    def backpropagate(self, inputs, targets):

        dwL1, dwL2, dwRec, dwOut, dbL1, dbHid, dbOut = np.zeros_like(
            self.weightL1), np.zeros_like(self.weightL2), np.zeros_like(
            self.weightRecur), np.zeros_like(self.weightOut), np.zeros_like(
            self.biasL1), np.zeros_like(self.biasHid), np.zeros_like(
            self.biasOut)
        dhNext = np.zeros_like(self.hState[0])

        for t in reversed(range(len(inputs))):
            deltaOut = np.copy(self.pState[t])

            deltaOut[targets[t]] -= 1
            dwOut += np.dot(deltaOut, self.hState[t].T)
            dbOut += deltaOut

            deltaHid = np.dot(self.weightOut.T, deltaOut) + dhNext
            deltaTan = (1 - self.hState[t] * self.hState[t]) * deltaHid
            dbHid += deltaTan

            dwL2 += np.dot(deltaTan, self.L1State[t].T)
            deltaL2 = np.dot(self.weightL2.T, deltaTan)

            dwL1 += np.dot(deltaL2, self.charVec[t].T)
            dbL1 += deltaL2

            dwRec += np.dot(deltaTan, self.hState[t - 1].T)

            dhNext = np.dot(self.weightRecur.T, deltaTan)

        for delta in [dwL1, dwL2, dwRec, dwOut, dbL1, dbHid, dbOut]:
            np.clip(delta, -5, 5, out=delta)

        return [dwL1, dwL2, dwRec, dwOut, dbL1, dbHid, dbOut]

    def adaGrad(self, deltaMatrices, lrMemMatrices, learnRate, epsilon):

        for param, delta, mem in zip([self.weightL1, self.weightL2,
                                      self.weightRecur, self.weightOut,
                                      self.biasL1, self.biasHid, self.biasOut],
                                     deltaMatrices, lrMemMatrices):
            mem += delta * delta
            param += -learnRate * delta / np.sqrt(mem + epsilon)

    def generateHiddenState(self, wordsIdx):
        # set initial prevHState
        prevHState = np.zeros((self.L2Neurons, 1))
        # generate new hstate at each timestep based on prevHState and
        # current char in words
        for t in range(len(wordsIdx)):
            x = np.zeros((self.vocabSize, 1))
            x[wordsIdx[t]] = 1
            L1State = np.dot(self.weightL1, x) + self.biasL1
            temp = np.dot(self.weightL2, L1State)
            prevHState = np.tanh(
                temp + np.dot(self.weightRecur, prevHState) + self.biasHid)

        return prevHState

    def generateWords(self, words, numChars, sampling=False, prevHStat=None):
        # convert words into idx vector
        wordsIdx = [char2Idx[ch] for ch in words]

        if not sampling:
            prevHState = self.generateHiddenState(wordsIdx)
        if sampling:
            prevHState = np.copy(prevHStat)
        # one hot encode last char in words (this is the intial input)
        x = np.zeros((self.vocabSize, 1))
        x[wordsIdx[-1]] = 1

        charIdx = []
        for t in range(numChars):
            # forward propagate
            L1State = np.dot(self.weightL1, x) + self.biasL1
            temp = np.dot(self.weightL2, L1State)
            prevHState = np.tanh(
                temp + np.dot(self.weightRecur, prevHState) + self.biasHid)
            yState = np.dot(self.weightOut, prevHState) + self.biasOut
            pState = np.exp(yState) / np.sum(np.exp(yState))

            # get idx of next char with highest probabilty
            idx = np.random.choice(range(self.vocabSize), p=pState.ravel())
            # set next char as input for next forward propagation
            x = np.zeros((self.vocabSize, 1))
            x[idx] = 1

            charIdx.append(idx)

        return ''.join(idx2Char[idx] for idx in charIdx)

    def saveModel(self):
        # export the weight matrices as a npz file
        saveArray = [self.weightL1, self.weightL2, self.weightRecur, self.weightOut,
                     self.biasL1, self.biasHid, self.biasOut, char2Idx, self.L1Neurons,
                     self.L2Neurons, self.vocabSize]
        np.save('TrainedModels/modelExport.npy', saveArray)

    def loadModel(self, fileName):
        # load weight matrices from npz file
        saveArray = np.load('TrainedModels/' + fileName)
        self.weightL1 = saveArray[0]
        self.weightL2 = saveArray[1]
        self.weightRecur = saveArray[2]
        self.weightOut = saveArray[3]
        self.biasL1 = saveArray[4]
        self.biasHid = saveArray[5]
        self.biasOut = saveArray[6]
        self.L1Neurons = saveArray[8]
        self.L2Neurons = saveArray[9]
        self.vocabSize = saveArray[10]

    def train(self, data, iterations, charBatch, learnRate, epsilon):
        lrMemMatrices = np.zeros_like(self.weightL1), np.zeros_like(
            self.weightL2), np.zeros_like(self.weightRecur), np.zeros_like(
            self.weightOut), np.zeros_like(self.biasL1), np.zeros_like(
            self.biasHid), np.zeros_like(self.biasOut)

        smoothLoss = -np.log(1.0 / self.vocabSize) * charBatch
        lossData = []
        pos = 0
        for n in range(iterations + 1):
            if pos + charBatch + 1 >= len(data) or n == 0:
                # set initial conditions
                prevHState = np.zeros((self.L2Neurons, 1))
                pos = 0

            inputs = [char2Idx[ch] for ch in data[pos:pos + charBatch]]
            targets = [char2Idx[ch]
                       for ch in data[pos + 1:pos + charBatch + 1]]

            loss, prevHState = self.lossFun(inputs, targets, prevHState)

            deltaMatrices = self.backpropagate(inputs, targets)

            self.adaGrad(deltaMatrices, lrMemMatrices, learnRate, epsilon)

            smoothLoss = smoothLoss * 0.999 + loss * 0.001

            if n % 1000 == 0:
                print('iter %d, loss: %f' % (n, smoothLoss))
                print(self.generateWords(
                    idx2Char[inputs[0]], 20, True, prevHState) + '\n')
                print(rnn.generateWords('a', 20))

            if n % 100 == 0:
                lossData.append((n / 100, smoothLoss))

            pos += charBatch

        self.saveModel()
        with open('lossData.txt', 'w') as file:
            for entry in lossData:
                file.write(str('{}, {}\n'.format(int(entry[0]), entry[1])))


if __name__ == '__main__':

    data = open('data.txt', 'r').read()

    dataSize, vocabSize = len(data), len(set(data))
    char2Idx = {char: idx for idx, char in enumerate(set(data))}

    idx2Char = inv = {val: key for key, val in char2Idx.items()}

    L1Neurons = 100
    L2Neurons = 100
    charBatch = 25
    learnRate = 0.1
    epsilon = 1e-8

    rnn = Rnn(L1Neurons, L2Neurons, vocabSize)
    print(rnn.train(data, 500, charBatch, learnRate, epsilon))
    # trainingGraph.plotgraph()

    rnn.loadModel('modelExport.npy')
    prevHState = np.zeros((L2Neurons, 1))
    print(rnn.generateWords('hello world', 20))
    print(rnn.generateWords('hello world', 20, True, prevHState))
