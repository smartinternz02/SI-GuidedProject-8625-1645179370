import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "kmaQDsThwa-HkekgsBT14wu_vTKoRC5FUvVUL4HK6xB7"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line
payload_scoring = {"input_data": [{"fields": [["f0", "f1","f2","f3", "f4","f5","f6", "f7","f8"]], "values": [[1999,12,23,2345,456,345,456,345,456]]}]}

response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/9802288c-15dc-4013-9e4d-f5f830527767/predictions?version=2022-04-16', json=payload_scoring,
 headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
pred= response_scoring.json()
print(pred)
output= pred['predictions'][0]['values'][0][0]
print(output)