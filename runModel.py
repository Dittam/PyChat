from pyChatModel import Rnn
from CleanData import cleanText
from CleanData import loadWordVocab

wordVocab = loadWordVocab()

model = Rnn()
char2Idx = model.loadModel('modelExport.npy')
idx2Char = {val: key for key, val in char2Idx.items()}

while True:
    inputs = input('>>> ')

    if inputs == 'exit':
        break

    inputs = cleanText(inputs)
    output = model.generateWords([char2Idx[char] for char in inputs], 50)
    temp = ''.join(idx2Char[i] for i in output)
    response = ''
    for i in temp:
        if i in wordVocab:
            response += i
    print(response)
