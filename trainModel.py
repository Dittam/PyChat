import numpy as np
from pyChatModel import Rnn
import trainingGraph


def train(rnn, data, iterations, charBatch, learnRate, epsilon):
    '''(Rnn, string, int, int, float, float) -> None
    given rnn object, training data, training iterations, batch size, will
    train the network 
    '''
    # generate dicts mapping chars to ints and vice versa
    char2Idx = {char: idx for idx, char in enumerate(set(data))}
    idx2Char = inv = {val: key for key, val in char2Idx.items()}

    # learning rate memory matrices used in adagrad optimizer
    lrMemMatrices = np.zeros_like(rnn.weightL1), np.zeros_like(
        rnn.weightL2), np.zeros_like(rnn.weightRecur), np.zeros_like(
        rnn.weightOut), np.zeros_like(rnn.biasL1), np.zeros_like(
        rnn.biasHid), np.zeros_like(rnn.biasOut)

    # smooth loss is better for graphing loss
    smoothLoss = -np.log(1.0 / rnn.vocabSize) * charBatch
    lossData = []  # loss data that will be written to file fro graphing
    pos = 0  # initial position in trainging data
    for n in range(iterations + 1):
        # if pos is at the start or end of training data,
        if pos + charBatch + 1 >= len(data) or n == 0:
            # set initial hidden state
            prevHState = np.zeros((rnn.L2Neurons, 1))
            pos = 0
        # generate char2int rep of input and targets, len of charBatch
        inputs = [char2Idx[ch] for ch in data[pos:pos + charBatch]]
        targets = [char2Idx[ch]
                   for ch in data[pos + 1:pos + charBatch + 1]]

        # forward prop, calculate loss, back prop, update weights with adagrad
        loss, prevHState = rnn.lossFunction(inputs, targets, prevHState)

        deltaMatrices = rnn.backpropagate(inputs, targets)

        rnn.adaGrad(deltaMatrices, lrMemMatrices, learnRate, epsilon)

        smoothLoss = smoothLoss * 0.999 + loss * 0.001

        # every 1000 trainging steps log model info
        if n % 1000 == 0:
            print('iter %d, loss: %f' % (n, smoothLoss))
            charIdx = rnn.generateWords([inputs[0]], 45, True, prevHState)
            print(''.join(idx2Char[i] for i in charIdx) + '\n')

        # every 100 trainging steps record loss data
        if n % 100 == 0:
            lossData.append((n / 100, smoothLoss))

        pos += charBatch  # move pos to next training step

    rnn.saveModel(char2Idx)  # save model and end of training

    # save loss data to txt file for graphing
    with open('lossData.txt', 'w') as file:
        for entry in lossData:
            file.write(str('{}, {}\n'.format(int(entry[0]), entry[1])))


if __name__ == '__main__':

    data = open('data.txt', 'r').read()

    vocabSize = len(set(data))
    L1Neurons = 100
    L2Neurons = 100
    charBatch = 25
    learnRate = 0.1
    epsilon = 1e-8

    neuralNet = Rnn(L1Neurons, L2Neurons, vocabSize)
    print(train(neuralNet, data, 300000, charBatch, learnRate, epsilon))
    trainingGraph.plotgraph()

# train start time 2:54, 5:00
