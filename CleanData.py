# July 23 2018
import re


def cleanText(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"[-()’\"#/\\@;^&_*$:<>{}`+=~|,]", "", text)

    return text


def generateData():
    # generate training data as txt file from cornell movie database
    lines = open('movie_lines.txt', encoding='utf-8',
                 errors='ignore').read().split('\n')

    with open('data.txt', 'a') as file:
        for line in lines:
            _line = line.split(' +++$+++ ')
            if len(_line) == 5:
                file.write(cleanText(_line[4]) + ' ')


def generateWordVocab():
    # generates a txt file with all unique words in training data
    lines = open('data.txt', encoding='utf-8',
                 errors='ignore').read().split('\n')
    split = lines[0].split(' ')
    vocab = set(split)
    vocab.remove('')
    vocab.add(' ')
    with open('wordVocab.txt', 'w') as file:
        for word in vocab:
            file.write(word + '\n')


def loadWordVocab():
    with open('wordVocab.txt', 'r') as file:
        words = file.read().split('\n')

    return words
