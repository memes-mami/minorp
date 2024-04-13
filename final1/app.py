from flask import Flask, request, jsonify
import random
import json
import pickle
import numpy as np
from tensorflow.keras import models
import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load data and model
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = models.load_model('chat_model.h5')
lemmatizer = WordNetLemmatizer()

# Define helper functions
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
    result = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Define Flask route
@app.route('/get_response', methods=['POST'])
def get_response_route():
    data = request.json.get('message')
    intents = predict_classes(data)
    response = get_response(intents, intents_json)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
