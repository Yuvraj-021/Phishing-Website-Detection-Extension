import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from urllib.parse import urlparse,urlencode
import re
from bs4 import BeautifulSoup
import requests
from keras.preprocessing import sequence
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.utils import to_categorical
from flask_cors import CORS
import tensorflow as tf
app = Flask(__name__)
CORS(app)
all_in_one_model = load_model('model_all.h5')


tokener = Tokenizer(lower=True, char_level=True, oov_token='-n-')

# Add the following function to load the tokenizer with more data
def load_tokenizer(filename):
    tokener.load_weights(filename)

# Add the following function to preprocess the URL using the tokenizer
def preprocess_url(url):
    tokener.fit_on_texts([url])
    char_index = tokener.word_index
    x_train = np.asanyarray(tokener.texts_to_sequences([url]))
    x_train = pad_sequences(x_train, maxlen=200)
    return x_train, char_index


# Add the following function to make predictions using the Keras model
def predict_keras(x_train, char_index):
    #x_train = to_categorical(x_train, num_classes=len(char_index) + 1)
    #prediction = all_in_one_model.predict(x_train).tolist()
    #classes_x=np.argmax(prediction,axis=1)
    prediction=np.argmax(all_in_one_model.predict(x_train),axis=1).tolist()
    return prediction

@app.route('/',methods=["GET","POST"])  
def home():
    return "Hello World"

@app.route('/post',methods=['POST'])
def predict():
    url = request.form['URL']
    x_train, char_index = preprocess_url(url)
    #features = extract_features(url)
    keras_prediction = predict_keras(x_train, char_index)
    print(keras_prediction)
    #return "0"
    #svm_prediction = predict_svm(features)
    if keras_prediction == 1:
        return "-1"
    elif keras_prediction ==0:
        return "0"
    else:
        return "1"

if __name__ == "__main__":
    app.run(debug=True)