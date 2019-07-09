# August 1, 2018
from fbchat import Client
from fbchat.models import *
import numpy as np
from pyChatModel import Rnn
from CleanData import cleanText
from CleanData import loadWordVocab
import time


def getResponse(inputs):
    '''(str) -> str
    given string generate a response using Rnn
    '''
    inputs = cleanText(inputs)
    output = model.generateWords([char2Idx[char] for char in inputs], 75)
    temp = ''.join(idx2Char[i] for i in output)
    response = ''
    for i in temp.split(' '):
        if i in wordVocab:
            response += i + ' '
    return response


def findUser(name):
    users = client.searchForUsers(name)
    user = users[0]
    print("User's ID: {}".format(user.uid))
    print("User's name: {}".format(user.name))
    print("User's profile picture url: {}".format(user.photo))
    print("User's main url: {}".format(user.url))


def sendMessage(thread_id='1776042969102657', thread_type=ThreadType.GROUP,
                message='<ONLINE>'):
    pyBot.send(Message(text=message, thread_id=thread_id,
                       thread_type=thread_type))


def generateNewSessionCookie(client):
    sessionCookies = client.getSession()
    with open('sessionCookies.npy', 'wb') as file:
        np.save(file, [sessionCookies])


class pyBot(Client):

    def onMessage(self, author_id, message_object,
                  thread_id, thread_type, logging_level=0, ** kwargs):
        #self.markAsDelivered(thread_id, message_object.uid)
        # self.markAsRead(thread_id)
        replyWithoutAt = False
        if 'exit()' in message_object.text:
            self.stopListening()
            self.send(Message(text='<OFFLINE>'), thread_id=thread_id,
                      thread_type=thread_type)

        elif '@PyBot Alpha' in message_object.text:
            response = getResponse(message_object.text)
            if author_id != self.uid:
                self.send(Message(text=response), thread_id=thread_id,
                          thread_type=thread_type)
        elif replyWithoutAt:
            response = getResponse(message_object.text)
            if author_id != self.uid:
                self.send(Message(text=response), thread_id=thread_id,
                          thread_type=thread_type)
                self.reactToMessage(message_object.uid, MessageReaction.LOVE)
        time.sleep(1)


if __name__ == '__main__':

    wordVocab = loadWordVocab()
    model = Rnn()
    char2Idx = model.loadModel('modelExport.npy')
    idx2Char = {val: key for key, val in char2Idx.items()}

    sessionCookie = np.load('sessionCookies.npy')[0]
    pyBot = pyBot("pybotalpha@gmail.com", "Rd-4210!",
                  session_cookies=sessionCookie)

    # threadList = pyBot.fetchThreadList()
    # for thread in threadList:
    #     sendMessage(thread.uid,thread.type,'<ONLINE>')

    pyBot.listen()
    # pyBot.logout()
