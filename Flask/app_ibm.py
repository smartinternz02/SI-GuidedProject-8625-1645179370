# importing the necessary dependencies
from flask import Flask,request,render_template
import numpy as np
import joblib

import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "kmaQDsThwa-HkekgsBT14wu_vTKoRC5FUvVUL4HK6xB7"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

app=Flask(__name__)# initializing a flask app
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
    x=x.tolist()
    print(x)
     # predictions using the loaded model file
    payload_scoring = {"input_data": [{"fields": [["f0", "f1","f2","f3", "f4","f5","f6", "f7","f8"]], "values": x}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/9802288c-15dc-4013-9e4d-f5f830527767/predictions?version=2022-04-16', json=payload_scoring,
    headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    pred= response_scoring.json()
    print(pred)
    output= pred['predictions'][0]['values'][0][0]
    print(output)
    labels=['Dark Trap','Emo','Hiphop','Pop','Rap','Rnb','Trap Metal','Underground Rap','dnb','hardstyle','psytrance','techhouse','techno','trance','trap']
    print("Prediction is:",labels[output])
    y=labels[output]
    print(type(y))
     # showing the prediction results in a UI
    return render_template("resultnew.html",prediction=labels[output])
if __name__=="__main__":
    
    app.run(host='0.0.0.0', port=8000,debug=False)    # running the app
    