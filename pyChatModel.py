# July ‎31, ‎2018
import numpy as np


class Rnn():
    '''Class representing the recurrent neural network'''

    def __init__(self, L1Neurons=0, L2Neurons=0, vocabSize=0):
        self.L1Neurons = L1Neurons  # layer 1 neurons
        self.L2Neurons = L2Neurons  # layer 2 neurons
        self.vocabSize = vocabSize  # number of unique chars in training data

        # ---Model Definition---
        # weights scaled by 0.01
        self.weightL1 = np.random.randn(L1Neurons, vocabSize) * 0.01
        self.weightL2 = np.random.randn(L2Neurons, L1Neurons) * 0.1
        self.weightRecur = np.random.randn(L2Neurons, L2Neurons) * 0.01
        self.weightOut = np.random.randn(vocabSize, L2Neurons) * 0.01
        self.biasL1 = np.zeros((L2Neurons, 1))
        self.biasHid = np.zeros((L2Neurons, 1))
        self.biasOut = np.zeros((vocabSize, 1))

        # Memory vars to keep track of states at each timestep
        # charVec: 1-hot vectors of len vocabSize representing each char
        # L1State: state of layer one matrix at each timestep
        # hState: hidden state matrix after at timestep
        # yState: raw vector output
        # pState: converted probablity distribution vector
        self.charVec, self.hState, self.yState, self.pState, self.L1State = {}, {}, {}, {}, {}

    def getTotalWeights(self):
        ''' return total weight parameters in the model
        '''
        return (self.L1Neurons * self.vocabSize) + (self.L2Neurons * self.L1Neurons) +\
            (self.L2Neurons * self.L2Neurons) + (self.vocabSize * self.L2Neurons) + \
            self.L1Neurons + self.L2Neurons + self.vocabSize

    def forwardPropagate(self, inputs, t):
        '''(array, int) -> None
        given char2int array (int rep of a seqnc of chars) and the current
        timestep will perform forward propagation of 1 timestep, updating
        hState, yState, pState, L1State
        '''
        # convert input char at timestep t into a 1-hot encoded vector
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

    def lossFunction(self, inputs, targets, prevHState):
        '''(array,array,array) -> int, array
        Negative Log Likelihood(Cross Entropy Loss)
        given char2int array (e.g. 'abc'->[1,2,3]), char2int array of
        target chars and previous hidden state matrix will perform
        forward propagation and return the loss and hidden state of last
        timestep
        '''
        # prevents mutation of the input hidden state matrix
        self.hState[-1] = np.copy(prevHState)

        loss = 0
        for t in range(len(inputs)):  # num timesteps = num chars in input
            self.forwardPropagate(inputs, t)
            # neg log likelihood
            loss += -np.log(self.pState[t][targets[t], 0])

        return loss, self.hState[len(inputs) - 1]

    def backpropagate(self, inputs, targets):
        '''(array, array) -> array of arrays
        given char2int array (e.g. 'abc'->[1,2,3]) and char2int array of
        target chars will perform backpropagation and return the delta matrices
        (how much each weight should change) at each training step
        '''
        # initialize delta matrices corresponding to each layer
        dwL1, dwL2, dwRec, dwOut, dbL1, dbHid, dbOut = np.zeros_like(
            self.weightL1), np.zeros_like(self.weightL2), np.zeros_like(
            self.weightRecur), np.zeros_like(self.weightOut), np.zeros_like(
            self.biasL1), np.zeros_like(self.biasHid), np.zeros_like(
            self.biasOut)
        dhNext = np.zeros_like(self.hState[0])

        for t in reversed(range(len(inputs))):
            # pass the error thorugh each layer started from output layer
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

        # clip each weight to avoid exploding gradient problem
        for delta in [dwL1, dwL2, dwRec, dwOut, dbL1, dbHid, dbOut]:
            np.clip(delta, -5, 5, out=delta)

        return [dwL1, dwL2, dwRec, dwOut, dbL1, dbHid, dbOut]

    def adaGrad(self, deltaMatrices, lrMemMatrices, learnRate, epsilon):
        '''(array, array, int, int) -> None
        update the weights in the network using adagrad optimizer
        '''
        for param, delta, mem in zip([self.weightL1, self.weightL2,
                                      self.weightRecur, self.weightOut,
                                      self.biasL1, self.biasHid, self.biasOut],
                                     deltaMatrices, lrMemMatrices):
            mem += delta * delta
            param += -learnRate * delta / np.sqrt(mem + epsilon)

    def generateHiddenState(self, wordsIdx):
        '''(array) -> array
        given char2int array (e.g. 'abc'->[1,2,3]) will generate hidden state
        for each timestep and return the last hidden state
        i.e. help the network gain context of a sentence
        '''
        # set initial prevHState
        prevHState = np.zeros((self.L2Neurons, 1))
        # generate new hstate at each timestep based on prevHState and
        # current char in words
        for t in range(len(wordsIdx)):
            # 1-hot encode char at current timestep
            x = np.zeros((self.vocabSize, 1))
            x[wordsIdx[t]] = 1

            L1State = np.dot(self.weightL1, x) + self.biasL1
            temp = np.dot(self.weightL2, L1State)
            prevHState = np.tanh(
                temp + np.dot(self.weightRecur, prevHState) + self.biasHid)

        return prevHState

    def generateWords(self, wordsIdx, numChars,
                      sampling=False, prevHStat=None):
        '''
        given char2int array (e.g. 'abc'->[1,2,3]) and desired length of output
        will return the char2int array representation of predicted output,
        enable sampling during training to see how the nework is performing
        '''
        # when not sampling manually generate hidden state based in input
        if not sampling:
            prevHState = self.generateHiddenState(wordsIdx)
        if sampling:
            prevHState = np.copy(prevHStat)

        # one hot encode last char in words (this is the intial input)
        x = np.zeros((self.vocabSize, 1))
        x[wordsIdx[-1]] = 1

        charIdx = []  # output list
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

        return charIdx

    def saveModel(self, char2Idx):
        # export weight matrices and model info to a npz file
        saveArray = [self.weightL1, self.weightL2, self.weightRecur, self.weightOut,
                     self.biasL1, self.biasHid, self.biasOut, char2Idx, self.L1Neurons,
                     self.L2Neurons, self.vocabSize]
        np.save('TrainedModels/modelExport.npy', saveArray)

    def loadModel(self, fileName):
        '''
        load weight matrices and model info from npz file return the dict
        mapping characters to integers (char2Idx)
        '''
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
        return saveArray[7]


if __name__ == '__main__':

    data = open('data.txt', 'r').read()
    dataSize, vocabSize = len(data), len(set(data))
    L1Neurons = 100
    L2Neurons = 100
    charBatch = 25
    learnRate = 0.1
    epsilon = 1e-8

    neuralNet = Rnn(L1Neurons, L2Neurons, vocabSize)
