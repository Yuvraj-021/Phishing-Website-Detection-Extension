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

from sklearn import metrics 

import pickle
from feature import FeatureExtraction

app = Flask(__name__)
CORS(app)
import joblib
gbc = joblib.load('model.pkl')
joblib.dump(gbc, 'model.pkl')



@app.route('/',methods=["GET","POST"])  
def home():
    return "Hello World"

@app.route('/post',methods=['POST'])
def predict():
    # 
    url = request.form["URL"]
    obj = FeatureExtraction(url)
    x = np.array(obj.getFeaturesList()).reshape(1,30) 
    y_pred =gbc.predict(x)[0]
    y_pro_phishing = gbc.predict_proba(x)[0,0]
    
    print("Y pred Pro ",y_pro_phishing)
    print("Y pred",y_pred)
    
    if y_pred == 0:
        return "-1"
    elif y_pred ==1:
        return "0"
    else:
        return "1"

if __name__ == "__main__":
    app.run(debug=True)