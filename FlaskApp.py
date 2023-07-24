# -*- coding: utf-8 -*-


import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
from flask import Flask, request, jsonify, render_template
import os
cwd = os.getcwd()

app=Flask(__name__)
pickle_in = open("regressor.pkl","rb")
regressor=pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Get the input features from the form as strings
    int_features_str = [x for x in request.form.values()]

    # Convert the input features to float (numeric) format
    try:
        final_features = [float(x) for x in int_features_str]
    except ValueError:
        # Handle cases where the input cannot be converted to a float
        # For example, if it's a string or contains non-numeric characters
        # You can choose to display an error message or set a default value.
        return render_template('index.html', prediction_text='Invalid input. Please enter numeric values.')

    # Convert the list of features to a numpy array
    final_features = np.array(final_features).reshape(1, -1)

    # Make the prediction using the loaded model
    prediction = regressor.predict(final_features)

    # Format the prediction to display only two decimal places
    formatted_prediction = "{:.2f}".format(prediction[0])

    # Display the formatted prediction result on the HTML page
    return render_template('index.html', prediction_text='The weight of the fish is {}'.format(formatted_prediction))

    
    

if __name__=='__main__':
    app.run()
