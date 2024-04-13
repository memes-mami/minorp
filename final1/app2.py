# Import necessary libraries
from flask import Flask, render_template, request, jsonify
import random
#from textblob import TextBlob
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)
import random
#from textblob import TextBlob

import json
import pickle

import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
from tensorflow.keras import models
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = models.load_model('chat_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words  = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words
def bag_of_words(sentence):
        sentence_words = clean_up_sentence(sentence)
        bag = [0]* len(words)
        for w in sentence_words:
            for i,word in enumerate(words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)
def predict_classes(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result =[[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key = lambda x: x[1], reverse =  True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents =  intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
# Load necessary resources for the chatbot (intents, model, etc.)
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
model = tf.keras.models.load_model('chat_model.h5')
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# Define a function to correct spelling using TextBlob
def correct_spelling(text):
    blob = TextBlob(text)
    return str(blob.correct())

# Define functions to clean up sentences and predict classes
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_classes(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in result]
    return return_list

def get_response(intents_list, all_intents):
    tag = intents_list[0]['intent']
    list_of_intents =  all_intents['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Define route for index page
@app.route('/')
def index():
    return render_template('basetemp2.html')

# Define route to process incoming messages
@app.route('/process_message', methods=['POST'])
def process_message():
    message = request.json['message']
    
    ints = predict_classes(message)
    response = get_response(ints, intents)
    print(response)
    return jsonify({'response': response})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)