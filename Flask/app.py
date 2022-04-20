# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:41:10 2020

@author: Adminr
"""
# importing the necessary dependencies
from flask import Flask,request,render_template
import numpy as np
import joblib

app=Flask(__name__)# initializing a flask app
model=joblib.load('bagging.model')
sc=joblib.load('transform.save')

@app.route('/')# route to display the home page
def home():
    return render_template('home.html') #rendering the home page
@app.route('/Prediction',methods=['POST','GET'])
def prediction(): # route which will take you to the prediction page
    return render_template('indexnew.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('home.html')

@app.route('/predict',methods=["POST","GET"])# route to show the predictions in a web UI
def predict():
    #  reading the inputs given by the user
    input_feature=[float(x) for x in request.form.values() ]  
    x=[np.array(input_feature)]
    x= sc.transform(x)
    print(x)
     # predictions using the loaded model file
    prediction=model.predict(x)  
    labels=['Dark Trap','Emo','Hiphop','Pop','Rap','Rnb','Trap Metal','Underground Rap','dnb','hardstyle','psytrance','techhouse','techno','trance','trap']
    print("Prediction is:",labels[prediction[0]])
     # showing the prediction results in a UI
    return render_template("resultnew.html",prediction=labels[prediction[0]])
if __name__=="__main__":
    
    app.run(host='0.0.0.0', port=8000,debug=False)    # running the app
    