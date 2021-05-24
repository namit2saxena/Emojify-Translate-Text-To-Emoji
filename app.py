from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize

IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

def init():
    global model,graph,tokenizer
    # load the pre-trained Keras model
    graph = tf.compat.v1.get_default_graph()
    tokenizer = Tokenizer()

#########################Code for Sentiment Analysis

def clean_text(data):
    
    # remove hashtags and @usernames
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    
    # tekenization using nltk
    data = word_tokenize(data)
    
    return data

@app.route('/', methods=['GET', 'POST'])
def home():

    return render_template("home.html")

@app.route('/sentiment_analysis_prediction', methods = ['POST', "GET"])
def sent_anly_prediction():
    if request.method=='POST':
        message = request.form['text']
        sentiment = ''
        
        #to train the internal vocabulary
        data_train = pd.read_csv('data_train.csv', encoding='utf-8')
        data_test = pd.read_csv('data_test.csv', encoding='utf-8')
        data = data_train.append(data_test, ignore_index=True)
        texts = [' '.join(clean_text(textT)) for textT in data.Text]
        tokenizer.fit_on_texts(texts)
        # Max input length (max number of words) 
        max_seq_len = 500
        class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
        seq = tokenizer.texts_to_sequences([message])
        padded = pad_sequences(seq, maxlen=max_seq_len)
        print(padded)
        with graph.as_default():
            json_file = open('model.json','r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            #load weights into new model
            loaded_model.load_weights("sentiment_analysis.h5")
            #compile and evaluate loaded models
            loaded_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
            # perform the prediction
            probability = np.amax(loaded_model.predict(padded)[0])
            pred = loaded_model.predict(padded)
            # print(np.argmax(pred))
            sentiment = class_names[np.argmax(pred)]
            sentiment = sentiment.capitalize()
            print(loaded_model.predict(padded))
        img_filename = os.path.join(app.config['UPLOAD_FOLDER'], '{0}_Emoji.png'.format(sentiment))
    return render_template('home.html', text=message, sentiment=sentiment, probability=probability, image=img_filename)
#########################Code for Sentiment Analysis

if __name__ == "__main__":
    init()
    app.run()
