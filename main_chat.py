import re

from sklearn import preprocessing
from underthesea.pipeline.word_tokenize.regex_tokenize import LOWER
import nltk
import argparse
from underthesea import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import tensorflow.compat.v1 as tf
import numpy as np
import tflearn
import random
import json
import pickle
from types import resolve_bases
from flask import Flask, render_template, request

# from unicode import unicode
# from chatterbot import ChatBot
# from chatterbot.trainers import ChatterBotCorpusTrainer
import time
import regex as reg
# import preprocessing

app = Flask(__name__)

# Libraries needed for NLP
nltk.download('punkt')
stemmer = LancasterStemmer()


def init():
    global data, words, classes, train_x, train_y, model, intents
    data = pickle.load(open("./training_data", "rb"))
    words = data['words']
    print(len(words))
    classes = data['classes'] 
    print(len(classes))
    train_x = data['train_x']
    # print(train_x)
    train_y = data['train_y']
    # print(train_y)

    with open('./movie.json', encoding="utf8") as json_data:
        intents = json.load(json_data)
    # intents = json.load('./data.json')

    # resetting underlying graph data
    tf.reset_default_graph()

    # Building neural network
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 10)
    net = tflearn.fully_connected(net, 10)
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)

    # Defining model and setting up tensorboard
    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
    model.load('./model.tflearn')

def clean_up_sentence(sentence):
    # tokenizing the pattern
    sentence_words = word_tokenize(sentence)
    # stemming each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# returning bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenizing the pattern
    sentence_words = clean_up_sentence(sentence)
    
    # sentence_words = text_preprocess(sentence)
    # generating bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return(np.array(bag))


ERROR_THRESHOLD = 0.3


def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list


@app.route("/")
def home():
    return render_template("chatbot.html")


@app.route("/chat")
def get_bot_response():
    userText = request.args.get('msg')
    print(f"user: {userText}")
    start = time.time()
    # userText = userText.lower()
    results = classify(userText)
    print(results)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # random response from the intent
                    if i['tag'] == 'greeting' or  i['tag'] == 'goodbye'  or  i['tag'] == 'thanks'  or  i['tag'] == 'default':
                        res = random.choice(i['responses'])
                        print(len(i['responses']))
                        end = time.time()
                        print(f"bot: {res}")
                        print(f"time of getting response: {round((end - start), 5)}")
                        return res
                    else:
                        if (len(i['responses']) >= 5):
                            res = random.sample(i['responses'],5)
                            end = time.time()
                            print(f"bot: {res}")
                            print(
                                f"time of getting response: {round((end - start), 5)}")
                            return json.dumps(res)
                        else:
                            res = random.choice(i['responses'])
                            # print(len(i['responses']))
                            end = time.time()
                            print(f"bot: {res}")
                            print(f"time of getting response: {round((end - start), 5)}")
                            return json.dumps(res)
        
            results.pop(0)

                
            


if __name__ == "__main__":
    init()
    app.run(host='localhost', port=8000, debug=True)
